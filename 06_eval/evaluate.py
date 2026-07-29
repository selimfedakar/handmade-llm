"""Chapter 06 — did it actually learn anything?

    python 06_eval/evaluate.py

Loads a checkpoint, measures it three ways, then runs three probes and prints
what each one means. Every number comes with the value you would expect from a
model that learned nothing, so you can tell the difference without having to
remember what good looks like.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / "01_tokenizer"))
sys.path.insert(0, str(REPO_ROOT / "02_model"))
sys.path.insert(0, str(REPO_ROOT / "03_train"))

from bpe import BPETokenizer  # noqa: E402
from data import Batcher  # noqa: E402
from metrics import bits_per_byte, perplexity, token_nll, top_k_accuracy  # noqa: E402
from model import Transformer  # noqa: E402
from probes import context_ablation, induction, memorisation_gap  # noqa: E402
from train import load_checkpoint  # noqa: E402


def measure(model: Transformer, batcher: Batcher, tokenizer: BPETokenizer, batch_size: int) -> dict:
    """Perplexity, bits per byte and accuracy over fixed windows of the split."""
    total_nll, token_count, byte_count = 0.0, 0, 0
    top1_sum, topk_sum, batches = 0.0, 0.0, 0

    for inputs, targets in batcher.sequential_batches(batch_size):
        nll, count = token_nll(model, inputs, targets)
        total_nll += nll
        token_count += count
        # The bytes those targets stand for in the original text. Decoding is
        # what makes the number tokenizer-independent, so it is worth the cost.
        byte_count += sum(
            len(tokenizer.decode(row).encode("utf-8")) for row in np.array(targets).tolist()
        )

        top1, topk = top_k_accuracy(model, inputs, targets, k=5)
        top1_sum += top1
        topk_sum += topk
        batches += 1

    return {
        "tokens": token_count,
        "bytes": byte_count,
        "loss": total_nll / max(1, token_count),
        "perplexity": perplexity(total_nll, token_count),
        "bits_per_byte": bits_per_byte(total_nll, byte_count),
        "top1": top1_sum / max(1, batches),
        "top5": topk_sum / max(1, batches),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "runs" / "latest" / "checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer.json")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "tokens")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="evaluate a freshly initialised model instead — the control run",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
        print("Run chapter 03 first: python 03_train/train.py --steps 300", file=sys.stderr)
        return 1

    model, _, step, _ = load_checkpoint(args.checkpoint)
    if args.untrained:
        mx.random.seed(0)
        model = Transformer(model.config)
        step = 0
    mx.eval(model.parameters())
    model.eval()

    tokenizer = BPETokenizer.load(args.tokenizer)
    val = Batcher.from_file(args.data_dir / "val.npy", args.seq_len)
    train = Batcher.from_file(args.data_dir / "train.npy", args.seq_len)

    label = "untrained control" if args.untrained else f"checkpoint at step {step}"
    print(f"{label}: {model.num_parameters / 1e6:.1f}M parameters, "
          f"vocabulary {model.config.vocab_size:,}\n")

    scores = measure(model, val, tokenizer, args.batch_size)
    uniform = math.log(model.config.vocab_size)

    print("Held-out split")
    print(f"  tokens / bytes       {scores['tokens']:,} / {scores['bytes']:,}")
    print(f"  loss                 {scores['loss']:.4f}   (a coin-flip model: {uniform:.2f})")
    print(f"  perplexity           {scores['perplexity']:.1f}   "
          f"(not comparable across tokenizers)")
    print(f"  bits per byte        {scores['bits_per_byte']:.3f}   (this one is)")
    print(f"  top-1 accuracy       {scores['top1'] * 100:.1f}%")
    print(f"  top-5 accuracy       {scores['top5'] * 100:.1f}%")

    print("\nProbes")
    ctx = context_ablation(model, val.tokens, args.seq_len)
    print(f"  context ablation     {ctx['with_context']:.4f} with real context, "
          f"{ctx['with_shuffled_context']:.4f} with it shuffled")
    print(f"                       gain {ctx['gain']:+.4f}  "
          f"(0 means the context is not being used)")

    ind = induction(model, model.config.vocab_size)
    print(f"  induction            {ind['first_copy']:.4f} first copy, "
          f"{ind['second_copy']:.4f} second")
    print(f"                       gain {ind['gain']:+.4f}  "
          f"(0 means it cannot copy from earlier in the sequence)")

    gap = memorisation_gap(model, train.tokens, val.tokens, args.seq_len)
    print(f"  memorisation         {gap['trained_on']:.4f} on trained text, "
          f"{gap['held_out']:.4f} on held out")
    print(f"                       gap {gap['gap']:+.4f}  "
          f"(large means it learned the corpus, not the language)")

    print(
        "\nRun with --untrained to see the same numbers from a model that has "
        "learned nothing.\nThat comparison is the only thing that makes any of "
        "these figures readable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
