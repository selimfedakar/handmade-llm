"""Tests for chapter 06 — metrics and probes.

Three carry the weight:

    test_bits_per_byte_survives_a_change_of_tokenizer
                                the chapter claims perplexity is not comparable
                                across tokenizers and bits-per-byte is; this is
                                the arithmetic that makes that true

    test_the_context_probe_feeds_the_whole_window_to_the_model
    test_the_induction_probe_lets_the_second_copy_see_the_first
                                both probes once sliced the sequence *before*
                                the forward pass, so the model never received
                                the thing being perturbed

Those last two are regression guards for `docs/LESSONS.md` L9. A probe *grades*
a slice; it must not *run on* one. When that was wrong, `context_ablation`
returned a gain of exactly 0.0 for every model that would ever exist, and
`induction` scored the second copy as a fresh sequence — measuring whether a
model can copy from text it cannot see. Both tests compare the probe against a
reference computation done the correct way, so reintroducing the slice turns
them red instead of silently producing a publishable number.

Run with:  python -m pytest 06_eval -q
"""

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from metrics import (  # noqa: E402
    accuracy_from_logits,
    bits_per_byte,
    nll_from_logits,
    perplexity,
    token_nll,
)
from model import ModelConfig, Transformer  # noqa: E402
from probes import context_ablation, induction, memorisation_gap  # noqa: E402

TINY = ModelConfig(
    vocab_size=128, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128
)


@pytest.fixture
def model():
    mx.random.seed(0)
    m = Transformer(TINY)
    mx.eval(m.parameters())
    m.eval()
    return m


@pytest.fixture(scope="module")
def corpus():
    # Random ids, not text: these tests measure the plumbing, and real language
    # would only make the numbers harder to reason about.
    return np.random.default_rng(7).integers(0, TINY.vocab_size, size=4000).astype(np.uint16)


# -- perplexity -----------------------------------------------------------


def test_a_model_spending_ln_k_nats_per_token_has_perplexity_k():
    # The definition, stated as a test: uniform uncertainty over k options
    # costs ln(k) nats and reads as a perplexity of exactly k.
    for k in (2, 10, 4096):
        assert perplexity(100 * math.log(k), 100) == pytest.approx(k)


def test_a_perfect_model_has_perplexity_one():
    assert perplexity(0.0, 50) == pytest.approx(1.0)


def test_perplexity_over_zero_tokens_is_refused():
    with pytest.raises(ValueError):
        perplexity(1.0, 0)


# -- bits per byte --------------------------------------------------------


def test_ln_two_nats_per_byte_is_exactly_one_bit_per_byte():
    assert bits_per_byte(1000 * math.log(2), 1000) == pytest.approx(1.0)


def test_eight_bits_per_byte_is_the_learned_nothing_reference():
    # A model with no idea at all spreads its mass over all 256 byte values,
    # which costs ln(256) nats per byte — eight bits, the size of the byte.
    assert bits_per_byte(500 * math.log(256), 500) == pytest.approx(8.0)


def test_bits_per_byte_over_zero_bytes_is_refused():
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0)


def test_bits_per_byte_survives_a_change_of_tokenizer():
    """The claim the chapter makes, as arithmetic.

    One text of 1,000 bytes, and one model that is exactly as surprised by it
    either way — the same total negative log likelihood. Tokenizer A cuts it
    into 250 tokens, tokenizer B into 500. Perplexity moves a long way. Bits
    per byte does not move at all, because the token count never enters the
    formula.
    """
    total_nll = 700.0
    byte_count = 1000

    coarse = perplexity(total_nll, 250)
    fine = perplexity(total_nll, 500)
    assert coarse > fine * 1.5, "the two tokenizers should disagree loudly"

    assert bits_per_byte(total_nll, byte_count) == pytest.approx(
        total_nll / math.log(2) / byte_count
    )


# -- accuracy -------------------------------------------------------------


def test_accuracy_reads_a_hand_built_case():
    # Four positions. The model gets position 0 right outright, has the truth
    # second at position 1, second at position 2, and last at position 3.
    logits = mx.array(
        [
            [5.0, 1.0, 0.0, -1.0],
            [1.0, 5.0, 0.0, -1.0],
            [5.0, 1.0, 0.0, -1.0],
            [5.0, 1.0, 0.0, -1.0],
        ]
    )
    targets = mx.array([0, 0, 1, 3])

    top1, top2 = accuracy_from_logits(logits, targets, k=2)
    assert top1 == pytest.approx(0.25)  # only position 0
    assert top2 == pytest.approx(0.75)  # positions 0, 1 and 2


def test_top1_never_exceeds_topk():
    logits = mx.random.normal((40, 16))
    targets = mx.array(np.random.default_rng(3).integers(0, 16, size=40).astype(np.int32))
    top1, topk = accuracy_from_logits(logits, targets, k=4)
    assert top1 <= topk


def test_a_k_as_wide_as_the_vocabulary_catches_everything():
    logits = mx.random.normal((25, 12))
    targets = mx.array(np.random.default_rng(4).integers(0, 12, size=25).astype(np.int32))
    _, topk = accuracy_from_logits(logits, targets, k=12)
    assert topk == pytest.approx(1.0)


def test_top_1_is_exactly_top_k_at_k_of_one():
    """Both numbers come from one rule, so k=1 cannot mean two things.

    An earlier version computed top-1 with `argmax` and top-k by counting what
    scores strictly above the true token. Those are different questions when
    two tokens tie for first: argmax breaks the tie by lowest index, counting
    does not, so the same position was wrong by one measure and right by the
    other — at k=1, where they ought to be the same measure.

    Exact float ties over a four-thousand-token vocabulary do not happen in
    practice, so nothing that has been reported changes. It is fixed and
    pinned because a metric that answers one question two ways is the kind of
    thing that is only ever discovered at the worst moment.
    """
    logits = mx.array([[3.0, 3.0, 0.0]])

    top1_picked, topk_picked = accuracy_from_logits(logits, mx.array([0]), k=1)
    assert top1_picked == pytest.approx(1.0)
    assert topk_picked == pytest.approx(1.0)

    # The token argmax would not have picked. Tied for first, so: still top-1.
    top1_tied, topk_tied = accuracy_from_logits(logits, mx.array([1]), k=1)
    assert top1_tied == pytest.approx(1.0)
    assert top1_tied == pytest.approx(topk_tied)

    # And a token that genuinely loses is still counted as losing.
    top1_lost, _ = accuracy_from_logits(logits, mx.array([2]), k=1)
    assert top1_lost == pytest.approx(0.0)


def test_a_k_below_one_is_refused():
    with pytest.raises(ValueError):
        accuracy_from_logits(mx.random.normal((3, 5)), mx.array([0, 1, 2]), k=0)


# -- the nll helpers ------------------------------------------------------


def test_token_nll_counts_every_target(model):
    inputs = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    targets = mx.array([[2, 3, 4, 5], [6, 7, 8, 9]])
    total, count = token_nll(model, inputs, targets)
    assert count == targets.size == 8
    assert math.isfinite(total)


def test_token_nll_sums_rather_than_averages(model):
    # Doubling the batch with the same rows must double the total, which is
    # what makes it safe to accumulate across batches of different sizes.
    row_in, row_out = mx.array([[1, 2, 3, 4]]), mx.array([[2, 3, 4, 5]])
    one, _ = token_nll(model, row_in, row_out)
    two, _ = token_nll(
        model, mx.concatenate([row_in, row_in]), mx.concatenate([row_out, row_out])
    )
    assert two == pytest.approx(2 * one, rel=1e-4)


def test_nll_from_logits_grades_only_what_it_is_given(model):
    # The helper that exists so probes can run the model on a whole window and
    # restrict the grading afterwards. Handed a slice, it must count exactly
    # that slice and nothing else.
    inputs = mx.array([[1, 2, 3, 4, 5, 6]])
    targets = mx.array([[2, 3, 4, 5, 6, 7]])
    logits, _ = model(inputs)

    whole, whole_count = nll_from_logits(logits, targets)
    tail, tail_count = nll_from_logits(logits[:, 3:, :], targets[:, 3:])
    head, head_count = nll_from_logits(logits[:, :3, :], targets[:, :3])

    assert whole_count == 6 and tail_count == 3 and head_count == 3
    assert whole == pytest.approx(head + tail, rel=1e-4)


# -- memorisation gap -----------------------------------------------------


def test_memorisation_gap_is_the_difference_it_reports(model, corpus):
    result = memorisation_gap(model, corpus[:3000], corpus[3000:], seq_len=32)
    assert result["gap"] == pytest.approx(result["held_out"] - result["trained_on"])
    assert math.isfinite(result["trained_on"]) and math.isfinite(result["held_out"])


def test_an_untrained_model_has_memorised_nothing(model, corpus):
    # Both splits here are the same kind of noise and the model has seen
    # neither, so the gap is sampling variation. Measured across five model
    # seeds before choosing the bound: |gap| stayed under 0.18.
    result = memorisation_gap(model, corpus[:3000], corpus[3000:], seq_len=32)
    assert abs(result["gap"]) < 0.4


# -- induction ------------------------------------------------------------


def test_induction_reports_the_difference_it_computes(model):
    result = induction(model, TINY.vocab_size, length=32, batch_size=4, seed=0)
    assert result["gain"] == pytest.approx(result["first_copy"] - result["second_copy"])


def test_induction_is_deterministic(model):
    first = induction(model, TINY.vocab_size, length=32, batch_size=4, seed=5)
    second = induction(model, TINY.vocab_size, length=32, batch_size=4, seed=5)
    assert first == second


def test_an_untrained_model_shows_no_induction(model):
    # Measured before choosing this bound: across 5 model seeds x 4 probe
    # seeds the gain ranged from -0.1247 to +0.0084, standard deviation
    # 0.0328. A bound of 0.25 sits well outside that spread while staying far
    # below what a model that had actually learned to copy would produce.
    for seed in range(3):
        gain = induction(model, TINY.vocab_size, length=48, batch_size=8, seed=seed)["gain"]
        assert abs(gain) < 0.25


def test_the_induction_probe_lets_the_second_copy_see_the_first(model):
    """The second copy must be scored with the first copy in context.

    Regression guard for LESSONS L9. The probe once passed `inputs[:, length:]`
    to the model, which starts a fresh sequence at the second copy and hides
    the only thing an induction circuit could copy from. Here the probe's own
    number is checked against a forward pass over the whole doubled sequence,
    and against the wrong answer the old code produced — so a reintroduced
    slice fails loudly rather than reporting a plausible zero.
    """
    length, batch = 48, 8
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, max(2, TINY.vocab_size - 1), size=(batch, length))
    doubled = np.concatenate([pattern, pattern], axis=1)

    inputs = mx.array(doubled[:, :-1].astype(np.int32))
    targets = mx.array(doubled[:, 1:].astype(np.int32))

    logits, _ = model(inputs)
    total, count = nll_from_logits(logits[:, length:, :], targets[:, length:])
    in_context = total / count

    sliced_total, sliced_count = token_nll(model, inputs[:, length:], targets[:, length:])
    sliced = sliced_total / sliced_count

    reported = induction(model, TINY.vocab_size, length=length, batch_size=batch, seed=0)
    assert reported["second_copy"] == pytest.approx(in_context, rel=1e-4)
    assert reported["second_copy"] != pytest.approx(sliced, rel=1e-4)


# -- context ablation -----------------------------------------------------


def test_context_ablation_reports_the_difference_it_computes(model, corpus):
    result = context_ablation(model, corpus, seq_len=64, batch_size=8, seed=0)
    assert result["gain"] == pytest.approx(
        result["with_shuffled_context"] - result["with_context"]
    )


def test_context_ablation_is_deterministic(model, corpus):
    first = context_ablation(model, corpus, seq_len=64, batch_size=8, seed=2)
    second = context_ablation(model, corpus, seq_len=64, batch_size=8, seed=2)
    assert first == second


def test_the_context_probe_feeds_the_whole_window_to_the_model(model, corpus):
    """Destroying the prefix has to change the score, or nothing was measured.

    Regression guard for LESSONS L9. When the probe sliced the prefix off the
    model's input, shuffling it could not change anything and the gain came
    back as exactly 0.0 — for every model, trained or not. An untrained model
    uses context only weakly, so the honest expectation is a small non-zero
    number; what is not honest is a perfect zero.
    """
    gain = context_ablation(model, corpus, seq_len=64, batch_size=8, seed=0)["gain"]
    assert gain != 0.0


def test_an_untrained_model_barely_uses_its_context(model, corpus):
    # Measured before choosing this bound: across 5 model seeds x 4 probe
    # seeds the gain ranged from -0.0292 to +0.0525, standard deviation
    # 0.0207. Untrained, the sign is not even consistent — which is the
    # control the trained model's six-of-six positive result is read against.
    for seed in range(3):
        gain = context_ablation(model, corpus, seq_len=64, batch_size=8, seed=seed)["gain"]
        assert abs(gain) < 0.15
