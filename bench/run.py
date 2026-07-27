"""The community benchmark — one command, one row, no setup.

    python bench/run.py

It builds a fixed model, trains it for a fixed number of steps on synthetic
tokens, generates a fixed number of tokens, and prints one Markdown row. No
corpus download, no tokenizer, nothing to configure — because a benchmark
people have to prepare for is a benchmark nobody runs.

The configuration is frozen on purpose. Chapter 04 is where you explore what
your machine can do; this is where every machine does the *same* thing, so the
numbers can sit in one table and mean something.

If you run it, send the row back. I own one Mac. The table is only worth
anything with more than one Mac in it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "02_model"))
sys.path.insert(0, str(REPO_ROOT / "03_train"))

from model import ModelConfig, Transformer  # noqa: E402
from train import loss_fn  # noqa: E402

# Frozen. Changing any of these makes the table incomparable, so if it ever has
# to change, the version below changes with it and old rows keep their version.
BENCH_VERSION = 1
BENCH_CONFIG = ModelConfig(
    vocab_size=4096,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=512,
)
BATCH_SIZE = 8
SEQ_LEN = 256
WARMUP_STEPS = 5
MEASURED_STEPS = 25
GENERATE_TOKENS = 64
MIB = 1024**2


def _sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def machine() -> dict:
    memory = _sysctl("hw.memsize")
    return {
        "chip": _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
        "memory_gib": round(int(memory) / 1024**3) if memory else None,
        "os": platform.mac_ver()[0] or platform.release(),
        "mlx": mx.__version__,
        "python": platform.python_version(),
    }


def measure_training() -> dict:
    """Steady-state training throughput and peak memory."""
    mx.random.seed(0)
    model = Transformer(BENCH_CONFIG)
    optimizer = optim.AdamW(learning_rate=3e-4)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    mx.eval(model.parameters())

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, BENCH_CONFIG.vocab_size, size=(BATCH_SIZE, SEQ_LEN + 1))
    inputs = mx.array(tokens[:, :-1].astype(np.int32))
    targets = mx.array(tokens[:, 1:].astype(np.int32))

    def step() -> None:
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        # MLX is lazy. Without this the loop below times nothing at all.
        mx.eval(model.parameters(), optimizer.state, loss)

    for _ in range(WARMUP_STEPS):
        step()

    mx.reset_peak_memory()
    start = time.perf_counter()
    for _ in range(MEASURED_STEPS):
        step()
    elapsed = time.perf_counter() - start

    return {
        "train_tokens_per_sec": MEASURED_STEPS * BATCH_SIZE * SEQ_LEN / elapsed,
        "peak_memory_mib": mx.get_peak_memory() / MIB,
        "parameters_m": model.num_parameters / 1e6,
    }


def measure_generation() -> dict:
    """Single-stream generation throughput with a warm KV-cache."""
    mx.random.seed(0)
    model = Transformer(BENCH_CONFIG)
    mx.eval(model.parameters())

    prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
    list(model.generate(prompt, max_tokens=8, temperature=0.0))  # warm up

    start = time.perf_counter()
    produced = list(model.generate(prompt, max_tokens=GENERATE_TOKENS, temperature=0.0))
    elapsed = time.perf_counter() - start

    return {"generate_tokens_per_sec": len(produced) / elapsed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="print the row plus what to do with it",
    )
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("This benchmark measures Apple Silicon.", file=sys.stderr)
        return 1

    info = machine()
    print(f"Benchmark v{BENCH_VERSION} on {info['chip']}, {info['memory_gib']} GiB")
    print(f"{BENCH_CONFIG.n_layers} layers x {BENCH_CONFIG.d_model}, "
          f"batch {BATCH_SIZE} x {SEQ_LEN} tokens\n")

    print("Training ...", flush=True)
    training = measure_training()
    print("Generating ...", flush=True)
    generation = measure_generation()

    result = {"version": BENCH_VERSION, **info, **training, **generation}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    row = (
        f"| {info['chip']} | {info['memory_gib']} | "
        f"{training['train_tokens_per_sec']:,.0f} | "
        f"{generation['generate_tokens_per_sec']:,.1f} | "
        f"{training['peak_memory_mib']:,.0f} | "
        f"{info['mlx']} | {info['os']} |"
    )

    print(f"\n  {training['parameters_m']:.1f}M parameters")
    print(f"  training     {training['train_tokens_per_sec']:,.0f} tokens/sec")
    print(f"  generation   {generation['generate_tokens_per_sec']:,.1f} tokens/sec")
    print(f"  peak memory  {training['peak_memory_mib']:,.0f} MiB")
    print(f"\nYour row:\n\n{row}\n")

    if args.submit:
        print("Add it to bench/results.md, keeping the table sorted by chip, then:")
        print("\n  git checkout -b bench/your-machine")
        print("  git add bench/results.md")
        print('  git commit -m "Add <your chip> to the benchmark table"')
        print("  git push -u origin bench/your-machine")
        print("\nOpen the pull request. That is the whole process.")
    else:
        print("Run again with --submit for how to add it to the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
