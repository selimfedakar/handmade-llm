"""Tests for the chapter 04 memory predictor and sweep.

Two of these carry real weight:

    test_parameter_count_matches_the_real_model
    test_the_prediction_lands_inside_its_stated_error_band

The first proves the exact half of the estimate is actually exact. The second
runs real training steps and checks the estimated half against what the machine
did — which is the only thing that stops a predictor from drifting into
fiction while every other test keeps passing.

Run with:  python -m pytest 04_scale -q
"""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from memory import (  # noqa: E402
    GIB,
    PREDICTION_ERROR_BAND,
    count_parameters,
    describe,
    estimate,
    layer_width,
)
from model import PRESETS, ModelConfig, Transformer  # noqa: E402

TINY = ModelConfig(
    vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=128
)


# -- the exact half -------------------------------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_parameter_count_matches_the_real_model(name):
    # The arithmetic in memory.py against the model in 02_model. If these ever
    # disagree, the estimate is describing a network that does not exist.
    config = PRESETS[name]
    assert count_parameters(config) == Transformer(config).num_parameters


@pytest.mark.parametrize("name", list(PRESETS))
def test_parameter_count_matches_with_untied_embeddings(name):
    config = replace(PRESETS[name], tie_embeddings=False)
    assert count_parameters(config) == Transformer(config).num_parameters


def test_tying_embeddings_removes_exactly_one_embedding_table():
    tied = replace(TINY, tie_embeddings=True)
    untied = replace(TINY, tie_embeddings=False)
    saved = count_parameters(untied) - count_parameters(tied)
    assert saved == TINY.vocab_size * TINY.d_model


def test_gradients_and_optimizer_are_three_more_copies_of_the_parameters():
    est = estimate(TINY, batch_size=2, seq_len=32)
    assert est.gradients == est.parameters
    assert est.optimizer == 2 * est.parameters


# -- scaling behaviour ----------------------------------------------------


def test_doubling_the_batch_size_doubles_the_activations():
    single = estimate(TINY, batch_size=4, seq_len=32).activations
    double = estimate(TINY, batch_size=8, seq_len=32).activations
    assert double == pytest.approx(2 * single, rel=1e-6)


def test_doubling_the_batch_size_leaves_the_parameters_alone():
    small = estimate(TINY, batch_size=4, seq_len=32)
    large = estimate(TINY, batch_size=8, seq_len=32)
    assert small.parameters == large.parameters
    assert small.optimizer == large.optimizer


def test_doubling_the_sequence_length_more_than_doubles_the_activations():
    # Linear in the layers and the logits, quadratic in the attention scores.
    # So the growth sits above 2x and below 4x, and knowing which end you are
    # near is the difference between a config that fits and one that does not.
    short = estimate(TINY, batch_size=4, seq_len=32).activations
    long = estimate(TINY, batch_size=4, seq_len=64).activations
    assert 2 * short < long < 4 * short


def test_grouped_query_attention_shrinks_the_kv_cache():
    # Four KV heads instead of eight is half the cache, exactly. This is the
    # entire argument for GQA on a laptop, so it gets an exact assertion.
    full = replace(TINY, n_heads=8, n_kv_heads=8, d_model=64)
    grouped = replace(TINY, n_heads=8, n_kv_heads=4, d_model=64)
    assert estimate(grouped, 4, 32).kv_cache * 2 == estimate(full, 4, 32).kv_cache


def test_grouped_query_attention_also_shrinks_the_layer_activations():
    full = replace(TINY, n_heads=8, n_kv_heads=8, d_model=64)
    grouped = replace(TINY, n_heads=8, n_kv_heads=4, d_model=64)
    assert layer_width(grouped) < layer_width(full)


def test_the_kv_cache_uses_the_full_context_not_the_training_length():
    # The cache is sized for generation, which runs out to max_seq_len whatever
    # you trained at.
    short = estimate(TINY, batch_size=4, seq_len=32)
    long = estimate(TINY, batch_size=4, seq_len=64)
    assert short.kv_cache == long.kv_cache


# -- the fit helper -------------------------------------------------------


def test_fits_in_at_the_boundary():
    est = estimate(TINY, batch_size=4, seq_len=32)
    exactly = est.training_bytes / GIB
    assert est.fits_in(exactly)
    assert est.fits_in(exactly * 1.001)
    assert not est.fits_in(exactly * 0.999)


def test_inference_is_cheaper_than_training():
    est = estimate(PRESETS["small"], batch_size=8, seq_len=256)
    assert est.inference_bytes < est.training_bytes
    assert est.fits_in(1.0, mode="inference")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        estimate(TINY, 4, 32).fits_in(8.0, mode="sideways")


def test_a_sequence_longer_than_the_context_is_refused():
    with pytest.raises(ValueError):
        estimate(TINY, batch_size=4, seq_len=TINY.max_seq_len + 1)


def test_describe_mentions_every_category():
    text = describe(PRESETS["small"], 8, 256)
    for expected in ("parameters", "optimizer", "logits", "attention scores", "KV-cache"):
        assert expected in text


# -- prediction against reality -------------------------------------------


def test_the_prediction_lands_inside_its_stated_error_band():
    """Run real training steps and hold the estimate to the band it claims.

    `memory.py` publishes an error band it was fit to. If a change to the model
    or to MLX moves the real number outside that band, the band is now a lie —
    and this test is what turns that into a failure instead of a slow drift.
    """
    import mlx.core as mx

    from sweep import run_one

    row = run_one("nano", batch_size=4, seq_len=128, steps=3, budget_gib=12.0)
    assert row.status == "ok", row.note
    assert row.measured_gib is not None and row.measured_gib > 0
    assert row.tokens_per_sec is not None and row.tokens_per_sec > 0

    low, high = PREDICTION_ERROR_BAND
    # A little slack on the published band: this is a single small config, and
    # the band was fit across 27 of them.
    assert low * 0.8 <= row.ratio <= high * 1.2, (
        f"predicted {row.predicted_gib:.3f} GiB, measured {row.measured_gib:.3f} GiB, "
        f"ratio {row.ratio:.2f}x is outside the published band {PREDICTION_ERROR_BAND}"
    )
    mx.clear_cache()


def test_a_config_that_cannot_fit_is_skipped_not_run():
    from sweep import run_one

    row = run_one("small", batch_size=8, seq_len=256, steps=1, budget_gib=0.001)
    assert row.status == "skipped"
    assert row.measured_gib is None
    assert "budget" in row.note


def test_the_sweep_table_has_a_row_per_combination():
    from sweep import Row, format_table

    rows = [
        Row("nano", 4, 128, 1_279_104, 0.14, 0.14, 1.0, 46_477.0, "ok"),
        Row("base", 32, 512, 78_662_400, 40.0, None, None, None, "skipped", "too large"),
    ]
    table = format_table(rows)
    lines = table.splitlines()
    assert len(lines) == 4  # header, divider, two rows
    assert "nano" in lines[2] and "46,477" in lines[2]
    assert "skipped" in lines[3] and "—" in lines[3]
