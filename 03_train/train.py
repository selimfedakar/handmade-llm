"""Chapter 03 — the training loop.

    python 03_train/train.py --preset small --steps 2000

This is where the model stops being a shape and starts being a model. Run it
and watch the loss fall; that curve is the only thing in this repository that
you cannot get from reading.

What is in here beyond the obvious:

    A schedule that warms up, then decays. Starting at full learning rate on
    freshly initialised weights is the classic way to lose a run in the first
    fifty steps — the gradients are enormous and the model walks somewhere it
    never recovers from.

    Gradient clipping. One bad batch produces one enormous gradient, and
    without a clip that single step can undo an hour.

    Checkpoints that actually resume. Weights, optimizer moments, and step
    number. Resuming without the optimizer state restarts Adam's estimates
    from zero, which shows up as a visible bump in the loss curve — a bump
    people usually blame on the data.

    Honest throughput. MLX is lazy, so anything timed without `mx.eval()`
    measures graph construction. Chapter 00 makes the same point; it matters
    more here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / "02_model"))

from data import Batcher, prepare  # noqa: E402
from model import PRESETS, ModelConfig, Transformer  # noqa: E402


@dataclass
class TrainConfig:
    preset: str = "small"
    steps: int = 2000
    batch_size: int = 16
    seq_len: int = 256
    learning_rate: float = 3e-4
    min_lr_fraction: float = 0.1  # the floor the cosine decays to
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    checkpoint_every: int = 500
    log_every: int = 10
    seed: int = 0
    # Stop this run after N steps, checkpoint, and exit — `steps` still
    # describes the whole run, so the schedule does not change. This is how
    # you train a model on a laptop you also need for other things: an hour
    # tonight, an hour tomorrow, one curve.
    stop_after: int | None = None


def learning_rate_at(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay to a floor.

    Warmup exists because the first gradients are the largest and the least
    informative. Cosine decay exists because late training wants small,
    careful steps. The floor keeps the model learning at the end instead of
    freezing at exactly zero.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = max(1, config.steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / decay_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    floor = config.learning_rate * config.min_lr_fraction
    return floor + (config.learning_rate - floor) * cosine


def loss_fn(model: Transformer, inputs: mx.array, targets: mx.array) -> mx.array:
    """Mean cross entropy over every position in the batch."""
    logits, _ = model(inputs)
    return nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
        targets.reshape(-1),
        reduction="mean",
    )


def evaluate(model: Transformer, batcher: Batcher, max_batches: int, batch_size: int) -> float:
    """Average loss over fixed, non-overlapping windows of the validation split."""
    model.eval()
    total, count = 0.0, 0
    for inputs, targets in batcher.sequential_batches(batch_size):
        total += loss_fn(model, inputs, targets).item()
        count += 1
        if count >= max_batches:
            break
    model.train()
    return total / max(1, count)


# -- checkpoints ----------------------------------------------------------


def save_checkpoint(
    directory: Path,
    model: Transformer,
    optimizer: optim.Optimizer,
    step: int,
    train_config: TrainConfig,
) -> None:
    """Write weights, optimizer state and enough metadata to resume exactly."""
    directory.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(directory / "weights.safetensors"), dict(tree_flatten(model.parameters())))
    mx.save_safetensors(
        str(directory / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "step": step,
                "train_config": asdict(train_config),
                "model_config": asdict(model.config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_checkpoint(directory: Path) -> tuple[Transformer, dict, int, TrainConfig]:
    """Rebuild the model and optimizer state from a checkpoint directory.

    All three files are required, and the optimizer is the one that goes missing.
    Weights are what everything else in this repository reads, so a directory can
    end up with current weights and no optimizer state — and resuming from that
    is not "resuming with a fresh optimizer", it is restarting Adam's moment
    estimates at zero underneath a model that is three hundred steps in. That
    diverges quietly rather than failing, which is the worst available outcome,
    so this refuses instead of coping.
    """
    for name in ("meta.json", "weights.safetensors", "optimizer.safetensors"):
        if not (directory / name).exists():
            raise FileNotFoundError(
                f"{directory / name} is missing, so this checkpoint cannot be resumed from.\n"
                f"Present: {sorted(p.name for p in directory.iterdir()) if directory.exists() else 'nothing'}\n"
                "Weights alone are enough to evaluate, quantize or ship a model — see "
                "chapters 06, 07 and 08 — but not to continue training it. Train a new "
                "run with --out-dir instead of resuming this one."
            )
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    model = Transformer(ModelConfig(**meta["model_config"]))
    model.update(tree_unflatten(list(mx.load(str(directory / "weights.safetensors")).items())))
    optimizer_state = tree_unflatten(
        list(mx.load(str(directory / "optimizer.safetensors")).items())
    )
    return model, optimizer_state, meta["step"], TrainConfig(**meta["train_config"])


# -- the loop -------------------------------------------------------------


def train(config: TrainConfig, data_dir: Path, out_dir: Path, resume: Path | None = None) -> dict:
    mx.random.seed(config.seed)

    train_batcher = Batcher.from_file(data_dir / "train.npy", config.seq_len, config.seed)
    val_batcher = Batcher.from_file(data_dir / "val.npy", config.seq_len, config.seed)

    start_step = 0
    if resume is not None:
        model, optimizer_state, start_step, _ = load_checkpoint(resume)
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        optimizer.state = optimizer_state
        print(f"Resuming from {resume} at step {start_step}")
    else:
        model_config = PRESETS[config.preset]
        model_config.max_seq_len = max(model_config.max_seq_len, config.seq_len)
        # The tokenizer decides the vocabulary, not the preset. `prepare()`
        # writes it next to the tokens for exactly this reason.
        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            model_config.vocab_size = json.loads(meta_path.read_text(encoding="utf-8"))[
                "vocab_size"
            ]
        model = Transformer(model_config)
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    mx.eval(model.parameters())

    # Checked once, before a single step: an id the embedding has no row for
    # does not raise, it silently corrupts the loss.
    for split, batcher in (("train", train_batcher), ("val", val_batcher)):
        largest = batcher.max_token_id
        if largest >= model.config.vocab_size:
            raise ValueError(
                f"{split} split contains token id {largest} but the model's vocabulary is "
                f"{model.config.vocab_size}. Rebuild the model with the tokenizer's "
                "vocab_size, special tokens included."
            )

    print(f"Model: {config.preset}, {model.num_parameters / 1e6:.1f}M parameters")
    print(f"Data: {len(train_batcher):,} train windows, {len(val_batcher):,} val windows")
    print(f"Tokens per step: {config.batch_size * config.seq_len:,}\n")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    history = []

    tokens_per_step = config.batch_size * config.seq_len
    window_start = time.perf_counter()
    last_step = start_step

    stop_at = config.steps
    if config.stop_after is not None:
        stop_at = min(config.steps, start_step + config.stop_after)
        if stop_at < config.steps:
            print(f"This run stops at step {stop_at} of {config.steps}.\n")

    for step in range(start_step, stop_at):
        optimizer.learning_rate = learning_rate_at(step, config)

        inputs, targets = train_batcher.batch(config.batch_size, step)
        loss, grads = loss_and_grad(model, inputs, targets)
        grads, grad_norm = optim.clip_grad_norm(grads, config.grad_clip)
        optimizer.update(model, grads)

        # Nothing above has actually run yet. This is where it does.
        mx.eval(model.parameters(), optimizer.state, loss)
        last_step = step + 1

        if (step + 1) % config.log_every == 0:
            elapsed = time.perf_counter() - window_start
            throughput = config.log_every * tokens_per_step / elapsed
            record = {
                "step": step + 1,
                "loss": loss.item(),
                "lr": float(optimizer.learning_rate.item()),
                "grad_norm": grad_norm.item(),
                "tokens_per_sec": throughput,
            }
            history.append(record)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step {step + 1:>6}/{config.steps}  "
                f"loss {record['loss']:.4f}  "
                f"lr {record['lr']:.2e}  "
                f"|grad| {record['grad_norm']:.2f}  "
                f"{throughput:,.0f} tok/s"
            )
            window_start = time.perf_counter()

        if (step + 1) % config.eval_every == 0:
            val_loss = evaluate(model, val_batcher, config.eval_batches, config.batch_size)
            print(f"{'':>13}  val loss {val_loss:.4f}  (perplexity {math.exp(val_loss):.1f})")
            window_start = time.perf_counter()

        if (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(out_dir / "checkpoint", model, optimizer, step + 1, config)

    save_checkpoint(out_dir / "checkpoint", model, optimizer, last_step, config)
    final_val = evaluate(model, val_batcher, config.eval_batches, config.batch_size)
    print(f"\nValidation loss at step {last_step}: {final_val:.4f} "
          f"(perplexity {math.exp(final_val):.1f})")
    print(f"Checkpoint written to {out_dir / 'checkpoint'}")
    if last_step < config.steps:
        print(f"Continue with:  --resume {out_dir / 'checkpoint'} --steps {config.steps}")
    return {"history": history, "val_loss": final_val, "last_step": last_step}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "tokens")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "latest")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="stop this run after N steps and checkpoint, leaving --steps unchanged",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="tokenize the corpus into --data-dir before training",
    )
    args = parser.parse_args()

    if args.prepare or not (args.data_dir / "train.npy").exists():
        text_path = REPO_ROOT / "data" / "tinyshakespeare.txt"
        tokenizer_path = REPO_ROOT / "data" / "tokenizer.json"
        if not text_path.exists() or not tokenizer_path.exists():
            print("Run `python data/download.py` and chapter 01 first.", file=sys.stderr)
            return 1
        stats = prepare(text_path, tokenizer_path, args.data_dir)
        print(
            f"Prepared {stats['total_tokens']:,} tokens "
            f"({stats['train_tokens']:,} train / {stats['val_tokens']:,} val), "
            f"dtype {stats['dtype']}\n"
        )

    config = TrainConfig(
        preset=args.preset,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        learning_rate=args.learning_rate,
        seed=args.seed,
        stop_after=args.stop_after,
    )
    train(config, args.data_dir, args.out_dir, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
