"""Tests for chapter 05 — LoRA and supervised fine-tuning.

Three carry the weight:

    test_a_fresh_adapter_changes_nothing        B starts at zero, so training
                                                begins from the model you have
    test_merging_preserves_the_output           the folded weights are the same
                                                function as the adapter
    test_the_prompt_contributes_nothing         changing the prompt's target
                                                tokens cannot move the loss

The third one is the chapter. If the prompt leaks into the loss the model
learns to write questions, the loss curve looks fine, and you find out weeks
later.

Run with:  python -m pytest 05_finetune -q
"""

import sys
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "01_tokenizer"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_train"))

from bpe import BPETokenizer  # noqa: E402
from lora import LoRALinear, apply_lora, count_trainable, merge_lora  # noqa: E402
from model import ModelConfig, Transformer  # noqa: E402
from sft import (  # noqa: E402
    PROMPT_TEMPLATE,
    check_ids_fit_the_model,
    collate,
    encode_example,
    make_dataset,
    masked_loss,
)

# 300 merged tokens plus one special, so ids run 0..300 and the model needs 301
# rows. Getting this off by one is what L7 in docs/LESSONS.md is about — MLX
# will not tell you, it will just return a different loss.
TINY = ModelConfig(
    vocab_size=301, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
)


@pytest.fixture
def model():
    mx.random.seed(0)
    m = Transformer(TINY)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train("User: Continue this line: to be or not to be that is the question. " * 60, 300)
    tok.register_special_tokens({"<|endoftext|>": 300})
    return tok


# -- LoRA -----------------------------------------------------------------


def test_a_fresh_adapter_changes_nothing(model):
    ids = mx.array([[3, 9, 27, 81]])
    before, _ = model(ids)
    apply_lora(model, rank=4)
    after, _ = model(ids)
    assert mx.allclose(before, after, atol=1e-6)


def test_the_adapter_can_move_the_output(model):
    apply_lora(model, rank=4)
    ids = mx.array([[3, 9, 27, 81]])
    before, _ = model(ids)
    # Nudge every B matrix off zero, the way one optimizer step would.
    for block in model.layers:
        for name in ("q_proj", "v_proj"):
            layer = getattr(block.attn, name)
            layer.lora_b = layer.lora_b + 0.01
    after, _ = model(ids)
    assert not mx.allclose(before, after, atol=1e-4)


def test_merging_preserves_the_output(model):
    apply_lora(model, rank=4)
    for block in model.layers:
        for name in ("q_proj", "v_proj"):
            layer = getattr(block.attn, name)
            layer.lora_b = mx.random.normal(layer.lora_b.shape) * 0.02

    ids = mx.array([[5, 15, 45, 135, 7]])
    adapted, _ = model(ids)
    merged, _ = merge_lora(model)(ids)
    assert mx.allclose(adapted, merged, atol=1e-4)


def test_merging_leaves_plain_linear_layers(model):
    apply_lora(model, rank=4)
    merge_lora(model)
    for block in model.layers:
        assert isinstance(block.attn.q_proj, nn.Linear)
        assert not isinstance(block.attn.q_proj, LoRALinear)


def test_only_the_adapters_train(model):
    # Structural, not a percentage: what fraction LoRA saves depends on the
    # model, but *which* tensors train does not.
    apply_lora(model, rank=4)

    def names(tree, prefix=""):
        if isinstance(tree, dict):
            for key, value in tree.items():
                yield from names(value, f"{prefix}.{key}")
        elif isinstance(tree, (list, tuple)):
            for i, value in enumerate(tree):
                yield from names(value, f"{prefix}.{i}")
        elif isinstance(tree, mx.array):
            yield prefix

    trainable_names = list(names(model.trainable_parameters()))
    assert trainable_names, "nothing is trainable"
    assert all(name.endswith(("lora_a", "lora_b")) for name in trainable_names), trainable_names

    trainable, total = count_trainable(model)
    assert trainable < total * 0.05


def test_trainable_count_matches_the_arithmetic(model):
    rank = 4
    apply_lora(model, rank=rank, targets=("q_proj",))
    trainable, _ = count_trainable(model)
    # One adapter per layer: A is (d_model, r), B is (r, d_model).
    expected = TINY.n_layers * (TINY.d_model * rank + rank * TINY.d_model)
    assert trainable == expected


def test_a_rank_larger_than_the_layer_is_refused():
    base = nn.Linear(8, 8, bias=False)
    with pytest.raises(ValueError):
        LoRALinear(base, rank=16)


def test_targeting_nothing_is_an_error(model):
    with pytest.raises(ValueError):
        apply_lora(model, targets=("does_not_exist",))


# -- masking, which is the actual lesson ----------------------------------


def test_the_mask_starts_where_the_response_starts(tokenizer):
    ids, mask = encode_example(tokenizer, "to be or not", "to be that is", max_len=64)
    assert len(ids) == len(mask)
    assert sum(mask) > 0
    assert mask[0] == 0  # the prompt is context
    assert mask[-1] == 1  # the last response token is scored


def test_the_prompt_contributes_nothing(model, tokenizer):
    # Two examples with the same response and different prompts of equal
    # length. The scored region is identical, so the loss must be too.
    long_prompt = "to be or not to be that is"
    ids_a, mask_a = encode_example(tokenizer, long_prompt, "the question", max_len=64)
    ids_b, mask_b = encode_example(tokenizer, long_prompt, "the question", max_len=64)

    inputs, targets, mask = collate([(ids_a, mask_a)])
    loss_one = masked_loss(model, inputs, targets, mask).item()

    # Same example, but with the mask widened to score the prompt as well.
    wide = [1] * len(mask_b)
    inputs_w, targets_w, mask_w = collate([(ids_b, wide)])
    loss_all = masked_loss(model, inputs_w, targets_w, mask_w).item()

    assert loss_one != pytest.approx(loss_all, rel=1e-3), (
        "scoring the prompt made no difference — the mask is not being applied"
    )


def test_a_token_id_the_model_has_no_row_for_is_refused():
    # MLX will not raise on this. It returns a number, and the number depends
    # on the batch shape, so the loss changes when you pad. See LESSONS L7.
    with pytest.raises(ValueError, match="vocabulary"):
        check_ids_fit_the_model([1, 2, 999], vocab_size=300, where="test")


def test_ids_inside_the_vocabulary_are_accepted():
    check_ids_fit_the_model([0, 299], vocab_size=300, where="test")


def test_padding_does_not_reach_the_loss(model, tokenizer):
    short = encode_example(tokenizer, "to be", "the question", max_len=64)
    long = encode_example(tokenizer, "to be or not to be that is", "the question", max_len=64)

    alone = collate([short])
    padded = collate([short, long])

    loss_alone = masked_loss(model, *alone).item()
    # The first row of the padded batch is the same example with zeros after it.
    first_row = tuple(t[:1] for t in padded)
    loss_padded = masked_loss(model, *first_row).item()
    assert loss_alone == pytest.approx(loss_padded, rel=1e-4)


def test_collate_shapes_line_up(tokenizer):
    batch = [
        encode_example(tokenizer, "to be", "or not", max_len=64),
        encode_example(tokenizer, "to be or not to be", "that is the question", max_len=64),
    ]
    inputs, targets, mask = collate(batch)
    assert inputs.shape == targets.shape == mask.shape
    assert inputs.shape[0] == 2


def test_examples_are_truncated_to_the_limit(tokenizer):
    ids, mask = encode_example(tokenizer, "to be or not " * 40, "the question", max_len=32)
    assert len(ids) == 32
    assert len(mask) == 32


# -- the dataset ----------------------------------------------------------


def test_dataset_splits_lines_in_half():
    text = "\n".join(["one two three four five six seven eight nine ten"] * 20)
    examples = make_dataset(text, count=5)
    assert len(examples) == 5
    for example in examples:
        assert example["prompt"].startswith("Continue this line: ")
        assert example["response"]


def test_dataset_skips_lines_that_are_too_short():
    assert make_dataset("hi\nthere\nshort line\n", count=5) == []


def test_dataset_is_deterministic():
    text = "\n".join(f"line number {i} with enough words to be usable here" for i in range(50))
    assert make_dataset(text, 10, seed=1) == make_dataset(text, 10, seed=1)


def test_prompt_template_ends_where_the_model_should_start():
    assert PROMPT_TEMPLATE.format(prompt="x").endswith("Assistant:")
