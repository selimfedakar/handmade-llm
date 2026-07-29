"""Chapter 06, part one — numbers that mean something.

The loss fell. That is not the same as the model having learned anything you
care about, and the gap between those two statements is where this chapter
lives.

Three measurements, in increasing order of how much they tell you:

    perplexity        how surprised the model is, per token
    bits per byte     the same thing, but comparable between models
    top-k accuracy    how often the right token is in the shortlist

**Perplexity is not comparable across tokenizers**, and almost everyone quotes
it as if it were. A model with a larger vocabulary uses fewer tokens for the
same text, so each token carries more information and perplexity goes up —
without the model being any worse. Change the tokenizer, change the number,
learn nothing.

Bits per byte fixes that by dividing by the size of the original text instead
of by the token count. Same text, same denominator, whatever the tokenizer did
in between. If you report one number from this repository, report that one.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def nll_from_logits(logits: mx.array, targets: mx.array) -> tuple[float, int]:
    """Total negative log likelihood in nats for logits you already have.

    Separate from `token_nll` because scoring a *slice* of a sequence is not
    the same as running the model on that slice. The model has to see the whole
    window; only the grading is restricted. Getting that backwards produces a
    probe that measures nothing and says so with suspicious precision —
    `docs/LESSONS.md` L9.
    """
    losses = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
        targets.reshape(-1),
        reduction="none",
    )
    return float(losses.sum().item()), int(targets.size)


def token_nll(model, inputs: mx.array, targets: mx.array) -> tuple[float, int]:
    """Total negative log likelihood in nats, and how many tokens it covers.

    Summed rather than averaged: averages of averages over batches of different
    sizes are quietly wrong, and this repository has already been bitten once
    by a denominator (chapter 05, and `docs/LESSONS.md` L8).
    """
    logits, _ = model(inputs)
    return nll_from_logits(logits, targets)


def perplexity(total_nll: float, token_count: int) -> float:
    """exp(mean negative log likelihood).

    Read it as: on average the model is as unsure as if it were choosing
    uniformly among this many tokens. Only comparable between models that
    share a tokenizer.
    """
    if token_count <= 0:
        raise ValueError("perplexity over zero tokens is not defined")
    return math.exp(total_nll / token_count)


def bits_per_byte(total_nll: float, byte_count: int) -> float:
    """Cross entropy in bits per byte of the original UTF-8 text.

    Divide nats by ln(2) to get bits, then by the byte count rather than the
    token count. That denominator is a property of the text, not of the
    tokenizer — which is exactly what makes the number portable.

    For scale: raw English is around 1 bit per byte for a strong model, and 8
    bits per byte is a model that has learned nothing at all.
    """
    if byte_count <= 0:
        raise ValueError("bits per byte over zero bytes is not defined")
    return total_nll / math.log(2) / byte_count


def accuracy_from_logits(
    logits: mx.array, targets: mx.array, k: int = 5
) -> tuple[float, float]:
    """Fraction of positions where the true token is the argmax, and in the top k.

    Accuracy answers a question perplexity does not: perplexity is dominated by
    the positions the model finds hardest, and a model can carry a bad
    perplexity while getting most tokens right. Both numbers, always.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    flat_logits = logits.reshape(-1, logits.shape[-1]).astype(mx.float32)
    flat_targets = targets.reshape(-1)

    # A token is in the top k when fewer than k tokens score strictly above it.
    # Both numbers come from that one rule, k=1 included, so "top-1" is exactly
    # "top-k at k=1" and cannot disagree with it.
    #
    # The obvious alternative for top-1 is `argmax(logits) == target`, and it is
    # subtly a different question: argmax breaks ties by taking the lowest
    # index, so of two tokens sharing the top score one would count as right and
    # the other as wrong, while at k=2 both count. That makes k=1 mean something
    # its neighbours do not. Here a tie for first place counts as top-1 — one
    # convention, stated, at every k.
    true_scores = mx.take_along_axis(flat_logits, flat_targets[:, None], axis=-1)
    better = (flat_logits > true_scores).sum(axis=-1)
    top1 = mx.mean(better < 1).item()
    topk = mx.mean(better < k).item()
    return float(top1), float(topk)


def top_k_accuracy(model, inputs: mx.array, targets: mx.array, k: int = 5) -> tuple[float, float]:
    """`accuracy_from_logits`, with the forward pass done for you."""
    logits, _ = model(inputs)
    return accuracy_from_logits(logits, targets, k)
