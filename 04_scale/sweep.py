"""Chapter 04, part two — measuring what a configuration actually costs.

    python 04_scale/sweep.py
    python 04_scale/sweep.py --full

`memory.py` predicts. This runs real training steps and reports what happened,
side by side with the prediction and the ratio between them.

That ratio is the point of the chapter. A predictor that is never checked is a
story you tell yourself about your own machine, and the first time it matters
is the first time it is wrong.

The batches here are random token ids, not text. That is deliberate: this
sweep measures the machine, not the data. Loss values from it mean nothing and
are not reported.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / "00_setup"))
sys.path.insert(0, str(REPO_ROOT / "02_model"))
sys.path.insert(0, str(REPO_ROOT / "03_train"))

from check_mac import machine_info, memory_budget  # noqa: E402
from memory import GIB, estimate  # noqa: E402
from model import PRESETS, Transformer  # noqa: E402

# The loss from chapter 03, unchanged. A sweep that measured a different
# forward pass than the one you train with would be measuring the wrong thing.
from train import loss_fn  # noqa: E402

FAST_SWEEP = {"presets": ["nano", "micro"], "batch_sizes": [4, 8], "seq_lens": [128, 256]}
FULL_SWEEP = {
    "presets": ["nano", "micro", "small", "base"],
    "batch_sizes": [4, 8, 16, 32],
    "seq_lens": [128, 256, 512],
}


@dataclass
class Row:
    preset: str
    batch_size: int
    seq_len: int
    n_parameters: int
    predicted_gib: float
    measured_gib: float | None
    ratio: float | None
    tokens_per_sec: float | None
    status: str
    note: str = ""


def run_one(
    preset: str,
    batch_size: int,
    seq_len: int,
    steps: int,
    budget_gib: float,
) -> Row:
    """Train a few real steps and report throughput and peak memory."""
    config = PRESETS[preset]
    if seq_len > config.max_seq_len:
        config = replace(config, max_seq_len=seq_len)

    prediction = estimate(config, batch_size, seq_len)
    predicted_gib = prediction.as_gib("training")
    row = Row(
        preset=preset,
        batch_size=batch_size,
        seq_len=seq_len,
        n_parameters=prediction.breakdown["n_parameters"],
        predicted_gib=predicted_gib,
        measured_gib=None,
        ratio=None,
        tokens_per_sec=None,
        status="ok",
    )

    if not prediction.fits_in(budget_gib):
        # Predicted not to fit. Say so and move on rather than sending the
        # machine into swap to prove a point.
        row.status = "skipped"
        row.note = f"predicted {predicted_gib:.1f} GiB > {budget_gib:.1f} GiB budget"
        return row

    try:
        mx.random.seed(0)
        model = Transformer(config)
        optimizer = optim.AdamW(learning_rate=3e-4, weight_decay=0.1)
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        mx.eval(model.parameters())

        inputs = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = mx.random.randint(0, config.vocab_size, (batch_size, seq_len))

        def one_step() -> None:
            loss, grads = loss_and_grad(model, inputs, targets)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            # MLX is lazy. Without this the loop below times graph
            # construction, which is the mistake chapter 00 warns about.
            mx.eval(model.parameters(), optimizer.state, loss)

        # Two warm-ups, not one. At large working sets the first step through a
        # newly grown allocation pays a one-time cost — on this machine, base
        # at batch 8 x 512 measured 85 tokens/sec on its first run and 4,233 on
        # its second, with byte-identical peak memory. One warm-up step is not
        # enough to get past that.
        one_step()
        one_step()

        # Reset the high-water mark now that everything persistent is live, so
        # the measurement covers a steady-state step rather than start-up
        # churn. Parameters and optimizer state stay allocated, so they are
        # still counted.
        mx.reset_peak_memory()

        # Time each step separately and take the median. A mean over a handful
        # of steps lets one slow outlier drag the answer a long way, and the
        # slow outlier is exactly what the warm-ups above are still sometimes
        # failing to absorb. The median says what a typical step costs, which
        # is the question being asked.
        durations = []
        for _ in range(steps):
            start = time.perf_counter()
            one_step()
            durations.append(time.perf_counter() - start)

        durations.sort()
        median = durations[len(durations) // 2]

        row.measured_gib = mx.get_peak_memory() / GIB
        row.ratio = row.measured_gib / predicted_gib if predicted_gib else None
        row.tokens_per_sec = batch_size * seq_len / median

    except Exception as exc:  # noqa: BLE001 - an allocation failure must not end the sweep
        row.status = "failed"
        row.note = f"{type(exc).__name__}: {str(exc)[:80]}"
    finally:
        # Drop everything before the next combination, or the allocator's cache
        # carries into the next measurement and the numbers drift upward.
        model = optimizer = loss_and_grad = inputs = targets = None
        mx.clear_cache()

    return row


def format_table(rows: list[Row]) -> str:
    header = (
        "| preset | params | batch | seq | tokens/step | predicted | measured | "
        "measured/predicted | tokens/sec | status |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for row in rows:
        measured = f"{row.measured_gib:.2f} GiB" if row.measured_gib is not None else "—"
        ratio = f"{row.ratio:.2f}x" if row.ratio is not None else "—"
        throughput = f"{row.tokens_per_sec:,.0f}" if row.tokens_per_sec is not None else "—"
        status = row.status if not row.note else f"{row.status} ({row.note})"
        lines.append(
            f"| {row.preset} | {row.n_parameters / 1e6:.1f}M | {row.batch_size} | "
            f"{row.seq_len} | {row.batch_size * row.seq_len:,} | "
            f"{row.predicted_gib:.2f} GiB | {measured} | {ratio} | {throughput} | {status} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="sweep every preset and shape")
    parser.add_argument("--steps", type=int, default=6, help="timed steps per combination")
    parser.add_argument(
        "--budget-gib",
        type=float,
        default=None,
        help="usable memory; defaults to this machine's, from chapter 00",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "sweep.json")
    args = parser.parse_args()

    info = machine_info()
    budget = args.budget_gib
    if budget is None:
        if info["memory_gib"] is None:
            print("Could not read this machine's memory; pass --budget-gib.", file=sys.stderr)
            return 1
        budget = memory_budget(info["memory_gib"])["usable_gib"]

    plan = FULL_SWEEP if args.full else FAST_SWEEP
    print(f"{info['chip']}, {info['memory_gib']:.0f} GiB unified")
    print(f"Budget for training: {budget:.1f} GiB (chapter 00's split)")
    print(f"{args.steps} timed steps per combination, random tokens\n")

    rows: list[Row] = []
    for preset in plan["presets"]:
        for batch_size in plan["batch_sizes"]:
            for seq_len in plan["seq_lens"]:
                if seq_len > PRESETS[preset].max_seq_len and not args.full:
                    continue
                row = run_one(preset, batch_size, seq_len, args.steps, budget)
                rows.append(row)
                marker = {"ok": " ", "skipped": "-", "failed": "!"}[row.status]
                measured = f"{row.measured_gib:.2f}" if row.measured_gib is not None else "  — "
                throughput = f"{row.tokens_per_sec:>9,.0f}" if row.tokens_per_sec else "        —"
                print(
                    f" {marker} {preset:<6} batch {batch_size:>3} x {seq_len:>4}  "
                    f"predicted {row.predicted_gib:>5.2f} GiB  "
                    f"measured {measured:>5} GiB  "
                    f"{throughput} tok/s  {row.note}"
                )

    print("\n" + format_table(rows))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "machine": {k: v for k, v in info.items()},
                "budget_gib": budget,
                "steps": args.steps,
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {args.out}")

    measured = [r for r in rows if r.ratio is not None]
    if measured:
        ratios = [r.ratio for r in measured]
        print(
            f"measured/predicted across {len(ratios)} combinations: "
            f"{min(ratios):.2f}x to {max(ratios):.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
