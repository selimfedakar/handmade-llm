"""Byte-level BPE tokenizer, written from scratch.

No dependencies beyond the standard library. This is chapter 1 of handmade-llm:
before a model can learn anything, text has to become integers, and *how* you
turn text into integers decides how much your model has to learn.

The algorithm is the original one from Sennrich et al. (2016), applied to raw
UTF-8 bytes the way GPT-2 does it:

    1. Start with a vocabulary of the 256 possible bytes.
    2. Count every adjacent pair of tokens in the corpus.
    3. Merge the most frequent pair into one new token.
    4. Repeat until the vocabulary is as large as you asked for.

Everything a byte-level tokenizer needs falls out of that loop: it can encode
any string in any language, it never emits an unknown token, and decoding is
exact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# How text is chopped up before merging. Merges never cross these boundaries,
# which is what stops the tokenizer from learning tokens like "dog." or " the("
# that glue words to punctuation.
#
# This is GPT-4's split in spirit. GPT-4 uses \p{L} / \p{N} from the PCRE
# `regex` package; we stay in the standard library, so letters are spelled
# [^\W\d_] (word character that is not a digit and not an underscore), which is
# Unicode-aware in Python 3 and behaves the same on the text you are likely to
# train on.
SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"  # common English contractions, kept whole
    r"| ?[^\W\d_]+"  # a run of letters, with an optional leading space
    r"| ?\d{1,3}"  # digits in groups of at most three
    r"| ?[^\s\w]+"  # punctuation and symbols
    r"|\s+(?!\S)"  # trailing whitespace at end of input
    r"|\s+"  # any remaining whitespace
)


def count_pairs(ids: list[int], counts: Counter | None = None) -> Counter:
    """Count every adjacent pair in `ids`, accumulating into `counts`."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with the single `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer you can train on your own text."""

    def __init__(self, pattern: str = SPLIT_PATTERN) -> None:
        self.pattern = pattern
        self._compiled = re.compile(pattern)
        self.merges: dict[tuple[int, int], int] = {}
        self.special_tokens: dict[str, int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # -- training ---------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merges from `text`.

        The textbook version of this loop re-counts every pair in the whole
        corpus before every single merge. It is the clearest way to write the
        algorithm and it is unusably slow: a megabyte of text and four thousand
        merges means several billion pair visits, which on a laptop is a coffee
        break turning into an afternoon. I tried it. Do not.

        Two changes fix it, and neither one changes a single learned merge:

        1. **Deduplicate chunks.** " the" appears thousands of times in any
           English corpus, and every copy merges identically. Keep one copy and
           a count.
        2. **Only touch what changed.** Keep a running pair counter and an index
           from each pair to the chunks containing it. After a merge, update
           just those chunks.

        Same output, and now it finishes while you are still looking at it.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        num_merges = vocab_size - 256

        # Each chunk is trained independently; merges never cross chunk borders.
        chunk_counts = Counter(self._compiled.findall(text))
        chunks = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        weights = list(chunk_counts.values())

        # Global pair counts, weighted by how often each chunk occurs, plus the
        # reverse lookup that lets us skip untouched chunks.
        pair_counts: Counter = Counter()
        pair_index: dict[tuple[int, int], set[int]] = {}
        for i, chunk in enumerate(chunks):
            for pair in zip(chunk, chunk[1:]):
                pair_counts[pair] += weights[i]
                pair_index.setdefault(pair, set()).add(i)

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for step in range(num_merges):
            if not pair_counts:
                break  # corpus fully merged; nothing left to learn

            # Most frequent pair wins. Ties are broken toward the lower token
            # ids so that training the same text twice gives the same merges.
            best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            if pair_counts[best] < 2:
                break  # a pair seen once is not a pattern

            new_id = 256 + step
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            occurrences = pair_counts[best]

            for i in sorted(pair_index[best]):
                old, weight = chunks[i], weights[i]
                for pair in set(zip(old, old[1:])):
                    pair_index[pair].discard(i)

                new = merge(old, best, new_id)
                chunks[i] = new

                # Recount this chunk from scratch. It is short, and doing the
                # arithmetic by difference is where the subtle bugs live.
                for pair, count in count_pairs(old).items():
                    pair_counts[pair] -= count * weight
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                for pair, count in count_pairs(new).items():
                    pair_counts[pair] += count * weight
                    pair_index.setdefault(pair, set()).add(i)

            pair_index.pop(best, None)

            if verbose:
                piece = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"merge {step + 1}/{num_merges}: "
                    f"{best} -> {new_id} ({piece!r}) had {occurrences} occurrences"
                )

    # -- special tokens ---------------------------------------------------

    def register_special_tokens(self, tokens: dict[str, int]) -> None:
        """Reserve ids for markers like `<|endoftext|>` that never get merged."""
        self.special_tokens = dict(tokens)
        for token, idx in tokens.items():
            self.vocab[idx] = token.encode("utf-8")

    # -- encoding / decoding ----------------------------------------------

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating any special-token text as ordinary characters."""
        ids: list[int] = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set[str] = "none") -> list[int]:
        """Encode text.

        `allowed_special` is "none" (specials are just text), "all" (every
        registered special is recognised), or an explicit set of strings.
        """
        if allowed_special == "none":
            special = {}
        elif allowed_special == "all":
            special = self.special_tokens
        else:
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

        if not special:
            return self.encode_ordinary(text)

        # Split on the special tokens, keeping them, then encode the gaps.
        splitter = "(" + "|".join(re.escape(k) for k in special) + ")"
        ids: list[int] = []
        for part in re.split(splitter, text):
            if part in special:
                ids.append(special[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def _encode_chunk(self, raw: bytes) -> list[int]:
        ids = list(raw)
        while len(ids) >= 2:
            # Apply the earliest-learned merge that is still present. Merge
            # order is the whole point: it is what makes encoding deterministic
            # and consistent with training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        parts = []
        for idx in ids:
            if idx not in self.vocab:
                raise ValueError(f"token id {idx} is not in the vocabulary")
            parts.append(self.vocab[idx])
        # errors="replace" because a slice of a token stream can cut a
        # multi-byte character in half; that is normal during generation.
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        """Write the tokenizer to a JSON file."""
        path = Path(path)
        payload = {
            "version": 1,
            "pattern": self.pattern,
            "merges": [[p0, p1, idx] for (p0, p1), idx in self.merges.items()],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Read a tokenizer back from a JSON file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=payload["pattern"])
        # Merges are replayed in order so the vocabulary rebuilds exactly.
        for p0, p1, idx in payload["merges"]:
            tok.merges[(p0, p1)] = idx
            tok.vocab[idx] = tok.vocab[p0] + tok.vocab[p1]
        tok.register_special_tokens(payload["special_tokens"])
        return tok
