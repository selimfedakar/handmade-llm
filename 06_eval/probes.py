"""Chapter 06, part two — probes that ask specific questions.

A single number cannot tell you *what* a model learned. Perplexity going from
120 to 100 is real, and it does not say whether the model started using
context, started memorising the training set, or simply got better at guessing
that the next character is a space.

So: three probes. Each one is a controlled comparison — the same model, the
same amount of text, one thing changed — and each one has an answer you can
predict for a model that learned nothing, which is what makes it a measurement
rather than a vibe.

    context ablation    does it use the context, or just the last few tokens?
    induction           can it notice a repeat and copy from it?
    memorisation gap    did it learn the language or the corpus?
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from metrics import nll_from_logits, token_nll  # noqa: E402


def context_ablation(
    model, tokens: np.ndarray, seq_len: int, batch_size: int = 8, seed: int = 0
) -> dict:
    """Score the tail of each window with a real prefix, then with a shuffled one.

    If the model is genuinely using its context, destroying the prefix must
    make the tail harder to predict. If the two numbers are the same, the model
    is a bigram table with extra steps — whatever the loss curve looked like.

    Only the second half of each window is scored, so the two runs are graded
    on identical positions and identical targets.
    """
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
    windows = np.stack([np.asarray(tokens[s : s + seq_len + 1]) for s in starts])

    half = seq_len // 2
    shuffled = windows.copy()
    for row in shuffled:
        # Shuffle the prefix only. The scored half is untouched.
        rng.shuffle(row[:half])

    def score(batch: np.ndarray) -> float:
        # The model reads the whole window — that is the point, the prefix has
        # to reach it. Only the grading is restricted to the second half.
        inputs = mx.array(batch[:, :-1].astype(np.int32))
        targets = mx.array(batch[:, 1:].astype(np.int32))
        logits, _ = model(inputs)
        total, count = nll_from_logits(logits[:, half:, :], targets[:, half:])
        return total / max(1, count)

    intact = score(windows)
    broken = score(shuffled)
    return {
        "with_context": intact,
        "with_shuffled_context": broken,
        "gain": broken - intact,
    }


def induction(
    model, vocab_size: int, length: int = 48, batch_size: int = 8, seed: int = 0
) -> dict:
    """Show the model a random sequence twice and see if the second time is cheaper.

    Random tokens are unpredictable by construction — there is no language to
    fall back on. The only way the second copy can be cheaper than the first is
    if the model looks back, finds where the current token appeared before, and
    copies what followed it. That circuit has a name in the literature and it
    is one of the first things a transformer learns.

    A model that has not learned it scores both halves the same, which is what
    makes this a real test rather than a demonstration.
    """
    rng = np.random.default_rng(seed)
    # Leave the special-token id alone; it never appears in ordinary text.
    pattern = rng.integers(0, max(2, vocab_size - 1), size=(batch_size, length))
    doubled = np.concatenate([pattern, pattern], axis=1)

    inputs = mx.array(doubled[:, :-1].astype(np.int32))
    targets = mx.array(doubled[:, 1:].astype(np.int32))

    # One forward pass over both copies. Scoring the second copy from its own
    # forward pass would hide the first one from the model, and hiding the
    # thing it is supposed to copy from turns the probe into a coin flip.
    logits, _ = model(inputs)
    first_total, first_count = nll_from_logits(logits[:, :length, :], targets[:, :length])
    second_total, second_count = nll_from_logits(logits[:, length:, :], targets[:, length:])

    first = first_total / max(1, first_count)
    second = second_total / max(1, second_count)
    return {"first_copy": first, "second_copy": second, "gain": first - second}


def memorisation_gap(
    model, train_tokens: np.ndarray, val_tokens: np.ndarray, seq_len: int, batch_size: int = 8
) -> dict:
    """Loss on text it trained on against text it has never seen.

    A small gap means it learned the language. A large one means it learned the
    corpus. Neither is automatically wrong — but you should know which you
    bought, and a single validation number will not tell you.
    """

    def score(tokens: np.ndarray, seed: int) -> float:
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, len(tokens) - seq_len - 1, size=batch_size)
        windows = np.stack([np.asarray(tokens[s : s + seq_len + 1]) for s in starts])
        inputs = mx.array(windows[:, :-1].astype(np.int32))
        targets = mx.array(windows[:, 1:].astype(np.int32))
        total, count = token_nll(model, inputs, targets)
        return total / max(1, count)

    seen = score(train_tokens, seed=1)
    unseen = score(val_tokens, seed=2)
    return {"trained_on": seen, "held_out": unseen, "gap": unseen - seen}
