"""Tests for the chapter 02 transformer.

Three of these carry real weight:

    test_written_out_attention_matches_the_fused_kernel
    test_cache_matches_a_full_forward_pass
    test_a_token_cannot_see_the_future

Everything else is shape checking. Those three are the ones that catch the
bugs that would otherwise show up as "the model trains but the samples are
subtly bad", which is the most expensive kind of bug in this repository.

Run with:  python -m pytest 02_model -q
"""

import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from model import (  # noqa: E402
    PRESETS,
    ModelConfig,
    Transformer,
    causal_mask,
    rope,
    sample,
)

TINY = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)


@pytest.fixture
def model():
    mx.random.seed(0)
    return Transformer(TINY)


# -- configuration --------------------------------------------------------


def test_defaults_fill_themselves_in():
    config = ModelConfig(d_model=512, n_heads=8)
    assert config.n_kv_heads == 8  # plain multi-head attention
    assert config.head_dim == 64
    assert config.d_ff % 64 == 0


def test_bad_head_split_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=8)


def test_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError):
        ModelConfig(d_model=512, n_heads=8, n_kv_heads=3)


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_build(name):
    Transformer(PRESETS[name])


# -- rotary positions -----------------------------------------------------


def test_rope_matches_mlx_fast_kernel():
    # The written-out rotation must agree with MLX's kernel, or every other
    # comparison in this file is measuring the wrong thing.
    x = mx.random.normal((2, 4, 9, 16))
    ours = rope(x, offset=0, base=10_000.0)
    theirs = mx.fast.rope(x, dims=16, traditional=False, base=10_000.0, scale=1.0, offset=0)
    assert mx.allclose(ours, theirs, atol=1e-5)


def test_rope_offset_matches_a_later_slice():
    # Rotating positions 4..7 directly must equal rotating 0..7 and slicing.
    x = mx.random.normal((1, 2, 8, 16))
    full = rope(x, offset=0)
    tail = rope(x[:, :, 4:, :], offset=4)
    assert mx.allclose(full[:, :, 4:, :], tail, atol=1e-5)


def test_rope_preserves_length():
    x = mx.random.normal((1, 1, 5, 16))
    before = mx.sqrt(mx.sum(x * x, axis=-1))
    after = mx.sqrt(mx.sum(rope(x) ** 2, axis=-1))
    assert mx.allclose(before, after, atol=1e-4)


def test_rope_dot_product_depends_only_on_distance():
    # The whole point of rotary embeddings: two vectors five apart score the
    # same whether they sit at positions 0 and 5 or at 20 and 25.
    q = mx.random.normal((1, 1, 1, 16))
    k = mx.random.normal((1, 1, 1, 16))

    def score(pos_q, pos_k):
        return mx.sum(rope(q, offset=pos_q) * rope(k, offset=pos_k)).item()

    assert score(0, 5) == pytest.approx(score(20, 25), abs=1e-4)


# -- masking --------------------------------------------------------------


def test_causal_mask_blocks_the_future():
    mask = causal_mask(3, 3)
    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == float("-inf")
    assert mask[2, 0].item() == 0.0


def test_cached_queries_sit_at_the_end_of_the_keys():
    # One new query with four keys behind it may see all four.
    mask = causal_mask(1, 4)
    assert mask.shape == (1, 4)
    assert mx.all(mask == 0.0).item()


# -- forward pass ---------------------------------------------------------


def test_output_shape(model):
    ids = mx.array([[1, 2, 3, 4]])
    logits, cache = model(ids)
    assert logits.shape == (1, 4, TINY.vocab_size)
    assert len(cache) == TINY.n_layers


def test_accepts_an_unbatched_sequence(model):
    logits, _ = model(mx.array([1, 2, 3]))
    assert logits.shape == (1, 3, TINY.vocab_size)


def test_cache_holds_kv_heads_not_query_heads(model):
    _, cache = model(mx.array([[1, 2, 3, 4]]))
    keys, values = cache[0]
    assert keys.shape == (1, TINY.n_kv_heads, 4, TINY.head_dim)
    assert values.shape == keys.shape


def test_a_token_cannot_see_the_future(model):
    # Change the last token. Every position before it must be untouched.
    a = mx.array([[5, 9, 2, 7]])
    b = mx.array([[5, 9, 2, 41]])
    logits_a, _ = model(a)
    logits_b, _ = model(b)
    assert mx.allclose(logits_a[:, :3, :], logits_b[:, :3, :], atol=1e-5)
    assert not mx.allclose(logits_a[:, 3, :], logits_b[:, 3, :], atol=1e-3)


def test_sequences_longer_than_the_context_are_refused(model):
    with pytest.raises(ValueError):
        model(mx.zeros((1, TINY.max_seq_len + 1), dtype=mx.int32))


def test_written_out_attention_matches_the_fused_kernel():
    # The claim the README makes about this repository, as a test.
    mx.random.seed(1)
    fused = Transformer(replace(TINY, fused_attention=True))
    written = Transformer(replace(TINY, fused_attention=False))
    written.update(fused.parameters())

    ids = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    a, _ = fused(ids)
    b, _ = written(ids)
    assert mx.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("fused", [True, False])
def test_cache_matches_a_full_forward_pass(fused):
    # Feeding tokens one at a time through the cache must produce exactly what
    # one pass over the whole sequence produces. When this drifts, generation
    # degrades quietly and training metrics never mention it.
    mx.random.seed(2)
    model = Transformer(replace(TINY, fused_attention=fused))
    ids = mx.array([[7, 3, 11, 2, 5, 8]])

    full, _ = model(ids)

    logits, cache = model(ids[:, :1])
    step_logits = [logits]
    for i in range(1, ids.shape[1]):
        logits, cache = model(ids[:, i : i + 1], cache)
        step_logits.append(logits)
    incremental = mx.concatenate(step_logits, axis=1)

    assert incremental.shape == full.shape
    assert mx.allclose(incremental, full, atol=1e-4)


# -- parameters -----------------------------------------------------------


def test_tying_embeddings_saves_a_vocabulary_sized_matrix():
    tied = Transformer(replace(TINY, tie_embeddings=True))
    untied = Transformer(replace(TINY, tie_embeddings=False))
    saved = untied.num_parameters - tied.num_parameters
    assert saved == TINY.vocab_size * TINY.d_model


def test_parameter_count_is_plausible(model):
    assert 0 < model.num_parameters < 10**6


# -- sampling -------------------------------------------------------------


def test_zero_temperature_is_deterministic():
    logits = mx.array([[1.0, 5.0, 2.0]])
    assert sample(logits, temperature=0.0).item() == 1


def test_top_k_never_leaves_the_top_k():
    logits = mx.array([[0.1, 9.0, 0.2, 8.0]])
    picks = {sample(logits, temperature=1.0, top_k=2).item() for _ in range(50)}
    assert picks <= {1, 3}


def test_generate_yields_the_requested_number_of_tokens(model):
    tokens = list(model.generate(mx.array([1, 2, 3]), max_tokens=5, temperature=0.0))
    assert len(tokens) == 5
    assert all(isinstance(t, int) and 0 <= t < TINY.vocab_size for t in tokens)


def test_greedy_generation_repeats_exactly(model):
    prompt = mx.array([4, 8, 15])
    first = list(model.generate(prompt, max_tokens=6, temperature=0.0))
    second = list(model.generate(prompt, max_tokens=6, temperature=0.0))
    assert first == second
