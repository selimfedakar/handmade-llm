"""Chapter 07, part two — what four bits gain you, and what they cost.

    python 07_quantize/compare.py            # the headline: float32 against 4-bit
    python 07_quantize/compare.py --sweep    # every bit width and group size

`quantize.py` does the arithmetic. This runs it against the real checkpoint and
answers the only three questions that matter:

    how much smaller     bytes, counted from the arrays themselves
    how much faster      tokens per second, two warm-ups and a median
    how much worse       chapter 06's metrics, before and against after

The third one is why this file imports from `06_eval` instead of measuring
quality its own way. A chapter that graded quantization with a metric it had
just invented would be marking its own homework — chapter 06 already argued for
bits per byte and already has the tests. Same metric, same split, same code.

The untrained control runs too, and it is not decoration. "Bits per byte went
up by 0.02" means nothing on its own. Against the 1.96 bits per byte the model
gained by training at all, it means the quantized model kept 99% of everything
the training run bought.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / "01_tokenizer"))
sys.path.insert(0, str(REPO_ROOT / "02_model"))
sys.path.insert(0, str(REPO_ROOT / "03_train"))
sys.path.insert(0, str(REPO_ROOT / "06_eval"))

from bpe import BPETokenizer  # noqa: E402
from data import Batcher  # noqa: E402

# Chapter 06's grader, unchanged. Grading quantization with a different metric
# than the one the repository already argued for would be measuring the wrong
# thing — the same reason chapter 04's sweep imports chapter 03's loss.
from evaluate import measure  # noqa: E402
from model import Transformer  # noqa: E402
from probes import context_ablation, induction, memorisation_gap  # noqa: E402
from quantize import (  # noqa: E402
    BITS,
    GROUP_SIZE,
    array_bytes,
    dequantize_groupwise,
    quantize_groupwise,
    quantize_model,
    round_trip_error,
    save_quantized,
)
from train import load_checkpoint  # noqa: E402

MIB = 1024**2


# -- size -----------------------------------------------------------------


def size_breakdown(model: Transformer) -> dict:
    """Bytes held by the model, split into the three things it is made of.

    Read straight off the arrays, so a packed uint32 counts as four bytes and
    not as the eight weights it stands for. Anything else and the compression
    ratio is a ratio of two numbers you made up.
    """
    embedding = array_bytes(model.embed.parameters())
    norms = array_bytes(model.norm.parameters()) + sum(
        array_bytes(block.attn_norm.parameters()) + array_bytes(block.ffn_norm.parameters())
        for block in model.layers
    )
    total = array_bytes(model.parameters())
    return {
        "embedding": embedding,
        "norms": norms,
        "layers": total - embedding - norms,
        "total": total,
    }


# -- weight error ---------------------------------------------------------


def weight_errors(model: Transformer, group_size: int, bits: int) -> list[tuple[str, dict]]:
    """Round-trip error for every matrix that would be quantized, by name.

    Run on the float32 model before anything is replaced, so this is the error
    the quantized model is *about* to carry — not a measurement taken after the
    fact from weights that have already lost the information.
    """
    matrices: list[tuple[str, mx.array]] = [("embed", model.embed.weight)]
    for i, block in enumerate(model.layers):
        for holder_name, holder in (("attn", block.attn), ("ffn", block.ffn)):
            for name in sorted(holder):
                layer = holder[name]
                if isinstance(layer, nn.Linear):
                    matrices.append((f"layers.{i}.{holder_name}.{name}", layer.weight))

    rows = []
    for name, weight in matrices:
        codes, scales, biases = quantize_groupwise(weight, group_size, bits)
        restored = dequantize_groupwise(codes, scales, biases, group_size)
        mx.eval(restored)
        rows.append((name, round_trip_error(weight, restored)))
    return rows


def summarise_errors(rows: list[tuple[str, dict]]) -> dict:
    """The worst layer and the typical one. Both, because either alone lies."""
    worst = max(rows, key=lambda row: row[1]["rel_rms"])
    best = min(rows, key=lambda row: row[1]["rel_rms"])
    mean_rel = sum(row[1]["rel_rms"] for row in rows) / len(rows)
    return {
        "worst_name": worst[0],
        "worst": worst[1],
        "best_name": best[0],
        "best": best[1],
        "mean_rel_rms": mean_rel,
        "max_abs": max(row[1]["max_abs"] for row in rows),
    }


# -- probes ---------------------------------------------------------------


def probe_comparison(
    model: Transformer, val: Batcher, train: Batcher, seq_len: int, seeds: int = 6
) -> dict:
    """Chapter 06's probes, run over several seeds, before and after quantizing.

    Bits per byte says the model is still as surprised as it was. It does not
    say the model is still doing the same *thing* — a circuit could be gone and
    the average could close over the hole. So the probes come too.

    Six seeds, not one. Chapter 06 established that a single probe number is not
    a result, because the untrained control's own spread is wider than the
    effect being measured. Quantization is a smaller intervention than training,
    so the margin matters more here, not less. What is being compared is the
    *sign and its consistency*, never the magnitude.
    """
    context = [
        context_ablation(model, val.tokens, seq_len, seed=seed)["gain"] for seed in range(seeds)
    ]
    induct = [
        induction(model, model.config.vocab_size, seed=seed)["gain"] for seed in range(seeds)
    ]
    gap = memorisation_gap(model, train.tokens, val.tokens, seq_len)

    return {
        "context": context,
        "context_positive": sum(1 for value in context if value > 0),
        "induction": induct,
        "induction_positive": sum(1 for value in induct if value > 0),
        "memorisation_gap": gap["gap"],
    }


def format_probe(values: list[float]) -> str:
    return f"{sum(values) / len(values):+.4f}"


def format_range(values: list[float]) -> str:
    return f"[{min(values):+.4f}, {max(values):+.4f}]"


# -- speed ----------------------------------------------------------------


def interleaved_speed(
    models: dict, prompt: mx.array, max_tokens: int = 64, rounds: int = 7
) -> dict:
    """Time both models in the same loop, alternating, and take medians.

    This exists because the obvious way — measure one, then measure the other —
    produced a different answer every time it ran. Seven of those runs, on the
    same two models, same machine, nothing else changed:

        float32   379.0  379.5  351.4  249.8  244.6  309.1  255.8
        quantized 380.4  376.0  367.7  338.3  358.8  278.3  260.0

    The ordering flipped four times and the spread inside one model was larger
    than any gap between the two. Peak memory, taken in those same runs, was
    byte-identical every single time — 97 MiB and 18 MiB, seven for seven. That
    is `docs/LESSONS.md` L5 arriving on schedule: memory is repeatable, speed is
    not.

    Two warm-ups and a median were never going to fix this, because the problem
    is not noise inside a run. It is **drift across** one: whichever model is
    measured second is measured on a different machine than the first. So both
    are measured in the same loop, one round each, and the drift lands on both.
    """

    def one_run(model: Transformer) -> None:
        for _ in model.generate(prompt, max_tokens=max_tokens, temperature=0.8, seed=0):
            pass

    for model in models.values():
        one_run(model)
        one_run(model)

    durations: dict[str, list[float]] = {name: [] for name in models}
    for _ in range(rounds):
        for name, model in models.items():
            start = time.perf_counter()
            one_run(model)
            durations[name].append(time.perf_counter() - start)

    results = {}
    for name, values in durations.items():
        values.sort()
        median = values[len(values) // 2]
        results[name] = {
            "tokens_per_sec": max_tokens / median,
            "spread": (max_tokens / values[-1], max_tokens / values[0]),
        }
    return results


def generation_speed(
    model: Transformer, prompt: mx.array, max_tokens: int = 64, runs: int = 5
) -> dict:
    """Tokens per second while generating, and the peak memory it took.

    Two warm-up runs and a **median**, not a mean. Chapter 04 nearly published a
    number that was fifty times wrong because the first pass through a new
    working set pays a one-time allocator cost and a mean let that single
    outlier set the answer. `docs/LESSONS.md` L6.

    `model.generate` calls `mx.eval` on every token it yields, so the loop below
    is timing real work rather than graph construction — but draining the
    generator is still required, because a generator that is never iterated
    computes nothing at all.
    """

    def one_run() -> None:
        for _ in model.generate(prompt, max_tokens=max_tokens, temperature=0.8, seed=0):
            pass

    one_run()
    one_run()

    mx.reset_peak_memory()
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        one_run()
        durations.append(time.perf_counter() - start)

    durations.sort()
    median = durations[len(durations) // 2]
    return {
        "tokens_per_sec": max_tokens / median,
        "seconds": median,
        "peak_mib": mx.get_peak_memory() / MIB,
        "spread": (max_tokens / durations[-1], max_tokens / durations[0]),
    }


# -- the comparison -------------------------------------------------------


def fresh_model(checkpoint: Path) -> tuple[Transformer, int]:
    model, _, step, _ = load_checkpoint(checkpoint)
    mx.eval(model.parameters())
    model.eval()
    return model, step


def quality(model: Transformer, batcher: Batcher, tokenizer: BPETokenizer, batch_size: int) -> dict:
    return measure(model, batcher, tokenizer, batch_size)


def print_headline(args, step: int, before: dict, after: dict, control: dict,
                   sizes_before: dict, sizes_after: dict, errors: dict,
                   speed_before: dict, speed_after: dict) -> None:
    scale = sizes_before["total"] / sizes_after["total"]

    print(f"Checkpoint at step {step}, quantized to {args.bits} bits "
          f"in groups of {args.group_size}"
          f"{'' if args.quantize_embedding else ', embedding left at float32'}\n")

    print("Size")
    print(f"{'':22}{'float32':>12}{'quantized':>12}")
    for key in ("embedding", "layers", "norms", "total"):
        print(f"  {key:<20}{sizes_before[key] / MIB:>10.2f} MiB"
              f"{sizes_after[key] / MIB:>9.2f} MiB")
    print(f"  {'compression':<20}{'':>14}{scale:>9.2f}x")

    print("\nWeight error, before any of it reaches a forward pass")
    print(f"  mean relative RMS    {errors['mean_rel_rms'] * 100:.2f}%")
    print(f"  worst matrix         {errors['worst_name']} "
          f"({errors['worst']['rel_rms'] * 100:.2f}%, "
          f"{errors['worst']['snr_db']:.1f} dB)")
    print(f"  best matrix          {errors['best_name']} "
          f"({errors['best']['rel_rms'] * 100:.2f}%, "
          f"{errors['best']['snr_db']:.1f} dB)")
    print(f"  largest single error {errors['max_abs']:.4f}")

    print("\nQuality on the held-out split")
    print(f"{'':22}{'float32':>12}{'quantized':>12}{'untrained':>12}")
    for label, key, fmt in (
        ("loss", "loss", "{:.4f}"),
        ("perplexity", "perplexity", "{:.1f}"),
        ("bits per byte", "bits_per_byte", "{:.3f}"),
    ):
        print(f"  {label:<20}{fmt.format(before[key]):>12}"
              f"{fmt.format(after[key]):>12}{fmt.format(control[key]):>12}")
    for label, key in (("top-1 accuracy", "top1"), ("top-5 accuracy", "top5")):
        print(f"  {label:<20}{before[key] * 100:>11.1f}%"
              f"{after[key] * 100:>11.1f}%{control[key] * 100:>11.1f}%")

    # The number the raw delta cannot give you: how much of the distance the
    # model travelled from knowing nothing is still there afterwards.
    travelled = control["bits_per_byte"] - before["bits_per_byte"]
    lost = after["bits_per_byte"] - before["bits_per_byte"]
    if travelled > 0:
        print(f"\n  Training moved this model {travelled:.3f} bits per byte from the "
              f"untrained control.\n  Quantizing gives back {lost:.3f} of that — "
              f"{100 * (1 - lost / travelled):.1f}% of the learning survives.")

    print(f"\nGeneration, {args.tokens} tokens from a one-token prompt")
    print(f"{'':22}{'float32':>12}{'quantized':>12}")
    print(f"  {'tokens/sec':<20}{speed_before['tokens_per_sec']:>12,.1f}"
          f"{speed_after['tokens_per_sec']:>12,.1f}")
    print(f"  {'peak memory':<20}{speed_before['peak_mib']:>8,.0f} MiB"
          f"{speed_after['peak_mib']:>8,.0f} MiB")
    print(f"  {'slowest–fastest':<20}"
          f"{speed_before['spread'][0]:>5,.0f}–{speed_before['spread'][1]:<6,.0f}"
          f"{speed_after['spread'][0]:>6,.0f}–{speed_after['spread'][1]:<6,.0f}")
    print(f"  Throughput is the paired measurement: {args.rounds} alternating rounds, "
          "medians.\n  Peak memory is the isolated one, and it is the only figure here "
          "that repeats to the byte.")


def run_sweep(args) -> int:
    """Every bit width against every group size, on the real checkpoint."""
    tokenizer = BPETokenizer.load(args.tokenizer)
    val = Batcher.from_file(args.data_dir / "val.npy", args.seq_len)

    reference, step = fresh_model(args.checkpoint)
    baseline = quality(reference, val, tokenizer, args.batch_size)
    baseline_bytes = array_bytes(reference.parameters())
    print(f"Checkpoint at step {step}: {baseline_bytes / MIB:.1f} MiB, "
          f"{baseline['bits_per_byte']:.3f} bits per byte, "
          f"top-1 {baseline['top1'] * 100:.1f}%\n")
    del reference
    mx.clear_cache()

    header = ("| bits | group | MiB | compression | weight rel RMS | bits/byte | "
              "top-1 | top-5 |")
    lines = [header, "|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for bits in args.sweep_bits:
        for group_size in args.sweep_groups:
            model, _ = fresh_model(args.checkpoint)
            try:
                errors = summarise_errors(weight_errors(model, group_size, bits))
                quantize_model(model, group_size, bits, args.quantize_embedding)
                mx.eval(model.parameters())
                scores = quality(model, val, tokenizer, args.batch_size)
                total = array_bytes(model.parameters())
                lines.append(
                    f"| {bits} | {group_size} | {total / MIB:.1f} | "
                    f"{baseline_bytes / total:.2f}x | "
                    f"{errors['mean_rel_rms'] * 100:.2f}% | "
                    f"{scores['bits_per_byte']:.3f} | "
                    f"{scores['top1'] * 100:.1f}% | {scores['top5'] * 100:.1f}% |"
                )
                print(f"  {bits} bits, group {group_size:>3}: "
                      f"{total / MIB:>5.1f} MiB, "
                      f"weight error {errors['mean_rel_rms'] * 100:>5.2f}%, "
                      f"{scores['bits_per_byte']:.3f} bits/byte, "
                      f"top-1 {scores['top1'] * 100:.1f}%")
            except ValueError as exc:
                # A group size that does not divide every matrix is not an
                # error in the sweep, it is the answer for that combination.
                lines.append(f"| {bits} | {group_size} | — | — | — | — | — | — |")
                print(f"  {bits} bits, group {group_size:>3}: skipped — {exc}")
            finally:
                model = None
                mx.clear_cache()

    print("\n" + "\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO_ROOT / "runs" / "latest" / "checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer.json")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "tokens")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--tokens", type=int, default=64, help="tokens to generate when timing")
    parser.add_argument("--runs", type=int, default=5, help="timed generation runs")
    parser.add_argument(
        "--rounds", type=int, default=7, help="alternating rounds in the paired timing"
    )
    parser.add_argument(
        "--keep-embedding-float32",
        dest="quantize_embedding",
        action="store_false",
        help="leave the token table at float32 — it is also the output head",
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="skip chapter 06's probes; bits per byte alone will not tell you whether "
        "a circuit survived",
    )
    parser.add_argument("--sweep", action="store_true", help="every bit width and group size")
    parser.add_argument("--sweep-bits", type=int, nargs="+", default=[8, 6, 5, 4, 3, 2])
    parser.add_argument("--sweep-groups", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "latest" / "quantized")
    parser.add_argument("--no-save", action="store_true", help="do not write the quantized model")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
        print("Run chapter 03 first: python 03_train/train.py --steps 300", file=sys.stderr)
        return 1

    if args.sweep:
        return run_sweep(args)

    tokenizer = BPETokenizer.load(args.tokenizer)
    val = Batcher.from_file(args.data_dir / "val.npy", args.seq_len)
    train = Batcher.from_file(args.data_dir / "train.npy", args.seq_len)
    prompt = mx.array([[tokenizer.encode("The ")[0]]])

    model, step = fresh_model(args.checkpoint)
    sizes_before = size_breakdown(model)
    before = quality(model, val, tokenizer, args.batch_size)
    speed_before = generation_speed(model, prompt, args.tokens, args.runs)
    probes_before = None if args.no_probes else probe_comparison(model, val, train, args.seq_len)
    error_rows = weight_errors(model, args.group_size, args.bits)
    errors = summarise_errors(error_rows)
    del model
    mx.clear_cache()

    # The control, measured in this run rather than remembered from chapter 06.
    # It is what makes "0.02 bits per byte worse" readable as a fraction of
    # something instead of as a number with no scale attached.
    control_model, _ = fresh_model(args.checkpoint)
    mx.random.seed(0)
    control_model = Transformer(control_model.config)
    mx.eval(control_model.parameters())
    control_model.eval()
    control = quality(control_model, val, tokenizer, args.batch_size)
    probes_control = (
        None if args.no_probes else probe_comparison(control_model, val, train, args.seq_len)
    )
    del control_model
    mx.clear_cache()

    model, _ = fresh_model(args.checkpoint)
    quantize_model(model, args.group_size, args.bits, args.quantize_embedding)
    mx.eval(model.parameters())
    sizes_after = size_breakdown(model)
    after = quality(model, val, tokenizer, args.batch_size)
    speed_after = generation_speed(model, prompt, args.tokens, args.runs)
    probes_after = None if args.no_probes else probe_comparison(model, val, train, args.seq_len)

    # Both models resident, timed in one alternating loop. The isolated
    # measurements above stay, because peak memory has to be read with only one
    # model in memory — but the throughput number that gets quoted is this one.
    reference, _ = fresh_model(args.checkpoint)
    paired = interleaved_speed(
        {"float32": reference, "quantized": model}, prompt, args.tokens, args.rounds
    )
    speed_before["tokens_per_sec"] = paired["float32"]["tokens_per_sec"]
    speed_before["spread"] = paired["float32"]["spread"]
    speed_after["tokens_per_sec"] = paired["quantized"]["tokens_per_sec"]
    speed_after["spread"] = paired["quantized"]["spread"]
    del reference
    mx.clear_cache()

    print_headline(args, step, before, after, control, sizes_before, sizes_after,
                   errors, speed_before, speed_after)

    if probes_before is not None:
        runs = (
            ("float32", probes_before),
            ("quantized", probes_after),
            ("untrained", probes_control),
        )
        print("\nProbes, six seeds each — chapter 06's, unchanged")
        print(f"  {'':<12}{'mean':>10}  {'range':<22}{'seeds positive':>15}")
        for label, key, positive in (
            ("context gain", "context", "context_positive"),
            ("induction gain", "induction", "induction_positive"),
        ):
            print(f"  {label}")
            for name, probes in runs:
                print(f"    {name:<10}{format_probe(probes[key]):>10}  "
                      f"{format_range(probes[key]):<22}"
                      f"{probes[positive]:>12}/6")
        print("  memorisation gap")
        for name, probes in runs:
            print(f"    {name:<10}{probes['memorisation_gap']:>+10.4f}")
        print("\n  The claim chapter 06 makes is the consistency of the *sign*, never "
              "the magnitude —\n  the effect is smaller than the untrained control's own "
              "spread. That is the claim\n  quantization has to leave standing, and it "
              "is what this block checks.")

    print("\nPer-matrix relative RMS error")
    for name, error in error_rows:
        print(f"  {name:<28}{error['rel_rms'] * 100:>7.2f}%   {error['snr_db']:>6.1f} dB")

    if not args.no_save:
        save_quantized(args.out, model, args.group_size, args.bits,
                       args.quantize_embedding, step)
        weights = args.out / "weights.safetensors"
        print(f"\nWritten to {args.out}  ({weights.stat().st_size / MIB:.1f} MiB on disk)")
        print("Chapter 08 loads this file from Swift.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
