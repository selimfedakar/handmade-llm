"""Chapter 02 — build the model and watch it generate.

    python 02_model/demo.py --preset small

It has not been trained, so what comes out is noise. That is the point of
running it now: you see the machine working end to end — tokenizer in,
transformer, sampling, tokenizer out — before any of it means anything. Then
chapter 03 gives it something to say.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_tokenizer"))

from bpe import BPETokenizer  # noqa: E402
from model import PRESETS, Transformer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizer.json",
        help="tokenizer from chapter 01",
    )
    args = parser.parse_args()

    if not args.tokenizer.exists():
        print(f"No tokenizer at {args.tokenizer}", file=sys.stderr)
        print("Run `python 01_tokenizer/train_tokenizer.py` first.", file=sys.stderr)
        return 1

    tokenizer = BPETokenizer.load(args.tokenizer)

    config = PRESETS[args.preset]
    config.vocab_size = tokenizer.vocab_size
    mx.random.seed(args.seed)
    model = Transformer(config)
    mx.eval(model.parameters())

    print(f"Model: {args.preset}")
    print(f"  layers x d_model     {config.n_layers} x {config.d_model}")
    print(f"  heads (query / kv)   {config.n_heads} / {config.n_kv_heads}")
    print(f"  context              {config.max_seq_len}")
    print(f"  vocabulary           {config.vocab_size:,}")
    print(f"  parameters           {model.num_parameters / 1e6:.1f}M")
    print(f"  weights at float32   {model.num_parameters * 4 / 1024**2:.0f} MiB")

    prompt_ids = mx.array(tokenizer.encode(args.prompt))
    print(f"\nPrompt: {args.prompt!r} -> {prompt_ids.size} tokens")

    print("\nGenerating (untrained — this is supposed to be nonsense):\n")
    start = time.perf_counter()
    generated = list(
        model.generate(
            prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=40,
            seed=args.seed,
        )
    )
    elapsed = time.perf_counter() - start

    print(f"  {args.prompt}{tokenizer.decode(generated)}")
    print(f"\n{len(generated)} tokens in {elapsed:.2f}s -> {len(generated) / elapsed:.1f} tokens/sec")
    print("\nThat rate is the untrained model with a cold cache. Chapter 04 measures it properly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
