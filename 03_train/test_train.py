"""Tests for the chapter 03 dataloader and training loop.

The one that matters is `test_the_loss_actually_goes_down`. Everything else
here can pass while the model learns nothing at all — a training loop that
runs is not a training loop that trains, and the difference is invisible until
you look at the number.

Run with:  python -m pytest 03_train -q
"""

import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from data import Batcher  # noqa: E402
from model import ModelConfig, Transformer  # noqa: E402
from train import (  # noqa: E402
    TrainConfig,
    learning_rate_at,
    load_checkpoint,
    save_checkpoint,
    train,
)

import mlx.optimizers as optim  # noqa: E402

VOCAB = 32
TINY_MODEL = ModelConfig(
    vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
)


# -- the learning rate schedule -------------------------------------------


def test_warmup_rises_from_near_zero_to_the_peak():
    config = TrainConfig(steps=1000, warmup_steps=100, learning_rate=1e-3)
    assert learning_rate_at(0, config) == pytest.approx(1e-5)
    assert learning_rate_at(49, config) == pytest.approx(5e-4)
    assert learning_rate_at(99, config) == pytest.approx(1e-3)


def test_warmup_is_monotonic():
    config = TrainConfig(steps=1000, warmup_steps=100, learning_rate=1e-3)
    values = [learning_rate_at(s, config) for s in range(100)]
    assert values == sorted(values)


def test_cosine_decays_to_the_floor_and_no_further():
    config = TrainConfig(steps=1000, warmup_steps=100, learning_rate=1e-3, min_lr_fraction=0.1)
    assert learning_rate_at(999, config) == pytest.approx(1e-4, rel=1e-3)
    # Past the end of the schedule it stays at the floor rather than going negative.
    assert learning_rate_at(5000, config) == pytest.approx(1e-4, rel=1e-3)


def test_no_warmup_starts_at_full_rate():
    config = TrainConfig(steps=100, warmup_steps=0, learning_rate=1e-3)
    assert learning_rate_at(0, config) == pytest.approx(1e-3, rel=1e-3)


# -- the dataloader -------------------------------------------------------


@pytest.fixture
def tokens():
    return np.arange(1000, dtype=np.uint16) % VOCAB


def test_targets_are_inputs_shifted_by_one(tokens):
    inputs, targets = Batcher(tokens, seq_len=8).batch(batch_size=4, step=0)
    assert inputs.shape == (4, 8)
    assert targets.shape == (4, 8)
    assert mx.array_equal(inputs[:, 1:], targets[:, :-1])


def test_the_same_step_gives_the_same_batch(tokens):
    batcher = Batcher(tokens, seq_len=8, seed=7)
    first, _ = batcher.batch(4, step=42)
    second, _ = batcher.batch(4, step=42)
    assert mx.array_equal(first, second)


def test_different_steps_give_different_batches(tokens):
    batcher = Batcher(tokens, seq_len=8, seed=7)
    assert not mx.array_equal(batcher.batch(4, step=0)[0], batcher.batch(4, step=1)[0])


def test_different_seeds_give_different_batches(tokens):
    a, _ = Batcher(tokens, seq_len=8, seed=0).batch(4, step=0)
    b, _ = Batcher(tokens, seq_len=8, seed=1).batch(4, step=0)
    assert not mx.array_equal(a, b)


def test_batches_stay_inside_the_corpus(tokens):
    inputs, targets = Batcher(tokens, seq_len=16).batch(batch_size=64, step=3)
    assert mx.max(inputs).item() < VOCAB
    assert mx.max(targets).item() < VOCAB


def test_a_corpus_too_short_for_the_window_is_refused():
    with pytest.raises(ValueError):
        Batcher(np.zeros(5, dtype=np.uint16), seq_len=8)


@pytest.mark.parametrize("batch_size", [1, 3, 7])
def test_sequential_batches_tile_the_split_without_gaps_or_overlap(tokens, batch_size):
    # Laid end to end, the windows must reconstruct the corpus exactly. That
    # catches both an overlap (a window repeated) and a gap (a window skipped),
    # which a check on batch starting points alone would miss.
    batcher = Batcher(tokens, seq_len=8)
    windows = [
        np.asarray(chunk) for chunk, _ in batcher.sequential_batches(batch_size=batch_size)
    ]
    laid_out = np.concatenate(windows).reshape(-1)
    usable = (len(tokens) - 1) // 8
    assert np.array_equal(laid_out, tokens[: usable * 8])


def test_sequential_batches_cover_the_split_once(tokens):
    batcher = Batcher(tokens, seq_len=8)
    total = sum(chunk.shape[0] for chunk, _ in batcher.sequential_batches(batch_size=7))
    assert total == (len(tokens) - 1) // 8


# -- checkpoints ----------------------------------------------------------


def test_checkpoint_round_trip_reproduces_the_model(tmp_path):
    mx.random.seed(3)
    model = Transformer(TINY_MODEL)
    optimizer = optim.AdamW(learning_rate=1e-3)
    mx.eval(model.parameters())

    ids = mx.array([[1, 2, 3, 4, 5]])
    before, _ = model(ids)

    save_checkpoint(tmp_path / "ckpt", model, optimizer, step=17, train_config=TrainConfig())
    restored, _, step, _ = load_checkpoint(tmp_path / "ckpt")

    after, _ = restored(ids)
    assert step == 17
    assert mx.allclose(before, after, atol=1e-6)


def test_checkpoint_keeps_the_model_shape(tmp_path):
    model = Transformer(TINY_MODEL)
    save_checkpoint(
        tmp_path / "ckpt", model, optim.AdamW(learning_rate=1e-3), 0, TrainConfig()
    )
    restored, _, _, _ = load_checkpoint(tmp_path / "ckpt")
    assert restored.config.n_layers == TINY_MODEL.n_layers
    assert restored.num_parameters == model.num_parameters


# -- the thing that actually matters --------------------------------------


def _synthetic_corpus(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    pattern = np.tile(np.arange(8, dtype=np.uint16), 4000)
    np.save(directory / "train.npy", pattern)
    np.save(directory / "val.npy", pattern[:4000])
    return directory


def test_resuming_continues_the_same_run(tmp_path, monkeypatch):
    """Interrupt a run, resume it, and land on the same numbers.

    This is the claim the checkpoint format exists to make, and it is stronger
    than a round-trip of the weights: it also covers the optimizer moments, the
    step counter feeding the learning-rate schedule, and the batch sampler.
    Get any one of those wrong and the resumed curve is *close* — which is
    exactly the failure that goes unnoticed for weeks.
    """
    data_dir = _synthetic_corpus(tmp_path / "tokens")

    import train as train_module

    monkeypatch.setitem(train_module.PRESETS, "test", replace(TINY_MODEL, vocab_size=VOCAB))

    base = TrainConfig(
        preset="test",
        steps=40,
        batch_size=8,
        seq_len=32,
        learning_rate=3e-3,
        warmup_steps=5,
        eval_every=1000,
        checkpoint_every=1000,
        log_every=5,
        seed=0,
    )

    # One run, stopped halfway, then picked up again.
    train(replace(base, stop_after=20), data_dir, tmp_path / "part1")
    resumed = train(base, data_dir, tmp_path / "part2", resume=tmp_path / "part1" / "checkpoint")

    # The same run, never interrupted.
    uninterrupted = train(base, data_dir, tmp_path / "whole")

    tail = {r["step"]: r for r in uninterrupted["history"] if r["step"] > 20}
    assert tail, "the uninterrupted run logged nothing after the halfway point"

    for record in resumed["history"]:
        expected = tail[record["step"]]
        assert record["loss"] == pytest.approx(expected["loss"], rel=1e-4), (
            f"step {record['step']}: resumed {record['loss']:.6f} "
            f"vs uninterrupted {expected['loss']:.6f}"
        )
        assert record["lr"] == pytest.approx(expected["lr"], rel=1e-9)


def test_stopping_early_leaves_the_schedule_alone():
    # `stop_after` must not change the learning rate at any step; it only
    # decides when this run hands over to the next one.
    full = TrainConfig(steps=1000, warmup_steps=100, learning_rate=1e-3)
    partial = replace(full, stop_after=200)
    assert [learning_rate_at(s, full) for s in range(0, 1000, 50)] == [
        learning_rate_at(s, partial) for s in range(0, 1000, 50)
    ]


def test_the_loss_actually_goes_down(tmp_path, monkeypatch):
    """Train a small model on a learnable pattern and watch the loss fall.

    The corpus is a short repeating cycle, so a working loop drops the loss
    quickly and a broken one does not move at all. This test has caught more
    than it looks like it would: a mask bug, a schedule that never left zero,
    and an optimizer that was updating a copy of the model.
    """
    data_dir = tmp_path / "tokens"
    data_dir.mkdir()
    pattern = np.tile(np.arange(8, dtype=np.uint16), 4000)
    np.save(data_dir / "train.npy", pattern)
    np.save(data_dir / "val.npy", pattern[:4000])

    import train as train_module

    monkeypatch.setitem(train_module.PRESETS, "test", replace(TINY_MODEL, vocab_size=VOCAB))

    config = TrainConfig(
        preset="test",
        steps=60,
        batch_size=8,
        seq_len=32,
        learning_rate=3e-3,
        warmup_steps=10,
        eval_every=1000,  # skip mid-run evaluation
        checkpoint_every=1000,
        log_every=10,
        seed=0,
    )
    result = train(config, data_dir, tmp_path / "run")

    losses = [record["loss"] for record in result["history"]]
    assert len(losses) >= 5
    assert losses[-1] < losses[0] * 0.6, f"loss barely moved: {losses}"
    assert result["val_loss"] < losses[0]
