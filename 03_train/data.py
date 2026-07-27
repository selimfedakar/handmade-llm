"""Chapter 03, part one — turning a text file into batches.

The dataloader is the part everyone rushes, and then spends a week debugging
because two runs with the same seed produced different curves. So it gets its
own file and its own tests.

Two ideas here:

    Tokenize once, to disk. Encoding a corpus is slow and the answer never
    changes. Chapter 01's tokenizer runs once and the token ids land in a
    `.npy` file that later runs memory-map instead of re-reading.

    Sample deterministically. Given a seed and a step number, the batch is
    fixed. Not "fixed if you do not restart" — actually fixed, so resuming
    from a checkpoint continues the same run instead of a similar one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_tokenizer"))

from bpe import BPETokenizer  # noqa: E402


def encode_corpus(text: str, tokenizer: BPETokenizer) -> np.ndarray:
    """Encode text to a compact array of token ids.

    uint16 holds vocabularies up to 65,536, which covers everything this
    repository trains. Half the memory of int32, and the corpus is the largest
    thing on disk.
    """
    ids = tokenizer.encode(text)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32
    return np.asarray(ids, dtype=dtype)


def prepare(
    text_path: Path,
    tokenizer_path: Path,
    out_dir: Path,
    val_fraction: float = 0.1,
) -> dict:
    """Tokenize a text file once and write train/val splits next to it.

    The split is by position, not shuffled: the validation set is the tail of
    the document. Shuffling first would put sentences from the same paragraph
    on both sides, and the validation loss would then be measuring memorisation
    of context it has already seen.
    """
    tokenizer = BPETokenizer.load(tokenizer_path)
    tokens = encode_corpus(text_path.read_text(encoding="utf-8"), tokenizer)

    split_at = int(len(tokens) * (1 - val_fraction))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train.npy", tokens[:split_at])
    np.save(out_dir / "val.npy", tokens[split_at:])

    stats = {
        "total_tokens": len(tokens),
        "train_tokens": split_at,
        "val_tokens": len(tokens) - split_at,
        "vocab_size": tokenizer.vocab_size,
        "dtype": str(tokens.dtype),
    }
    # The vocabulary travels with the data. The model's embedding table has to
    # be as wide as the tokenizer, special tokens included — and a preset's
    # default vocab_size has no way of knowing what you trained the tokenizer
    # on. See docs/LESSONS.md L7 for what happens when those two disagree.
    (out_dir / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class Batcher:
    """Deterministic batches of (inputs, targets) for next-token prediction.

    Targets are inputs shifted by one. Every position in the window is a
    training example, which is why a language model gets so much signal out of
    so little text — a 256-token window is 256 predictions, not one.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, seed: int = 0) -> None:
        if len(tokens) < seq_len + 1:
            raise ValueError(
                f"need at least {seq_len + 1} tokens for seq_len {seq_len}, got {len(tokens)}"
            )
        self.tokens = tokens
        self.seq_len = seq_len
        self.seed = seed

    @classmethod
    def from_file(cls, path: Path, seq_len: int, seed: int = 0) -> "Batcher":
        return cls(np.load(path, mmap_mode="r"), seq_len, seed)

    def __len__(self) -> int:
        """How many distinct windows exist in this split."""
        return len(self.tokens) - self.seq_len

    @property
    def max_token_id(self) -> int:
        """The largest id in this split, for checking it fits the model.

        One full pass over the tokens. Paid once per run, and worth it: MLX
        does not raise when a target id falls outside the vocabulary — it
        returns a number that changes with the batch shape, and training
        continues with a loss that is quietly wrong. See docs/LESSONS.md L7.
        """
        return int(np.max(self.tokens))

    def batch(self, batch_size: int, step: int) -> tuple[mx.array, mx.array]:
        """The batch for a given step. Same seed and step, same batch, always."""
        # A fresh generator per step rather than a running one: that is what
        # makes resuming at step 5,000 identical to never having stopped.
        rng = np.random.default_rng((self.seed, step))
        starts = rng.integers(0, len(self), size=batch_size)

        window = self.seq_len + 1
        chunks = np.stack([np.asarray(self.tokens[s : s + window]) for s in starts])
        inputs = mx.array(chunks[:, :-1].astype(np.int32))
        targets = mx.array(chunks[:, 1:].astype(np.int32))
        return inputs, targets

    def sequential_batches(self, batch_size: int):
        """Walk the split once, in order, without overlap — for evaluation.

        Validation loss has to be comparable between runs, so it is measured on
        fixed windows rather than random ones.
        """
        window = self.seq_len + 1
        usable = (len(self.tokens) - 1) // self.seq_len
        for start in range(0, usable, batch_size):
            count = min(batch_size, usable - start)
            chunks = np.stack(
                [
                    np.asarray(self.tokens[i * self.seq_len : i * self.seq_len + window])
                    for i in range(start, start + count)
                ]
            )
            yield (
                mx.array(chunks[:, :-1].astype(np.int32)),
                mx.array(chunks[:, 1:].astype(np.int32)),
            )
