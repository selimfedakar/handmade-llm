"""Chapter 01 — train a tokenizer on your own text.

    python 01_tokenizer/train_tokenizer.py --vocab-size 4096

By default it trains on `data/tinyshakespeare.txt`. Point `--input` at your own
text and the vocabulary becomes yours: your language, your domain, your code.
That is the part people skip, and it is the part that changes the numbers.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bpe import BPETokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
END_OF_TEXT = "<|endoftext|>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "tinyshakespeare.txt",
        help="text file to train on",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "tokenizer.json",
        help="where to write the trained tokenizer",
    )
    parser.add_argument("--vocab-size", type=int, default=4096, help="target vocabulary size")
    parser.add_argument("--verbose", action="store_true", help="print every merge as it is learned")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"No such file: {args.input}", file=sys.stderr)
        print("Run `python data/download.py` first, or pass --input.", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8")
    raw_bytes = len(text.encode("utf-8"))
    print(f"Training on {args.input.name}: {len(text):,} characters, {raw_bytes:,} bytes")

    tokenizer = BPETokenizer()
    start = time.perf_counter()
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    elapsed = time.perf_counter() - start

    # A special token so chapter 03 can tell one document from the next.
    tokenizer.register_special_tokens({END_OF_TEXT: args.vocab_size})

    tokens = tokenizer.encode(text)
    ratio = raw_bytes / len(tokens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output)

    print(f"\nLearned {len(tokenizer.merges):,} merges in {elapsed:.1f}s")
    print(f"Corpus is now {len(tokens):,} tokens")
    print(f"Compression: {ratio:.2f} bytes per token")
    print(f"Saved to {args.output}")

    # Look at what it learned. The longest tokens tell you what your corpus is
    # actually made of — and that is worth ten minutes of your attention.
    learned = sorted(
        (tokenizer.vocab[i] for i in range(256, 256 + len(tokenizer.merges))),
        key=len,
        reverse=True,
    )
    print("\nLongest tokens it decided were worth keeping:")
    for piece in learned[:12]:
        print(f"  {piece.decode('utf-8', errors='replace')!r}")

    sample = text[:120]
    print("\nRound-trip check on the first 120 characters:")
    print(f"  {'exact match' if tokenizer.decode(tokenizer.encode(sample)) == sample else 'MISMATCH'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
