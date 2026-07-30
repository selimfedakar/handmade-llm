"""Tests for chapter 07 — group-wise quantization.

Four carry the weight, and three of them are the same argument this repository
has been making since chapter 01:

    test_the_packed_words_match_mlx_exactly
                                the scheme written out here and the fused
                                kernel produce byte-identical uint32 words, at
                                every bit width and group size MLX supports —
                                not "close", identical

    test_the_quantized_matmul_matches_dequantize_then_multiply
                                the readable path (rebuild the float32 matrix,
                                then multiply) and the shipping path (never
                                rebuild it at all) agree to float32 noise

    test_quantize_model_matches_the_generic_quantizer
                                walking the architecture by hand lands on the
                                same model MLX's one-line `nn.quantize` builds

    test_a_tie_breaks_away_from_zero_the_way_the_kernel_does
                                the fourth, and the one that had to be built
                                rather than sampled: a weight sitting exactly
                                half way between two codes, where `mx.round`
                                and MLX's kernel disagree about which way to go

Without the first one, "we implemented 4-bit quantization" is a claim about
code nobody checked. A lossy transform is exactly the kind of thing that can be
subtly wrong forever: every number it produces is *supposed* to be wrong, so
being wrong in one extra way is invisible. Byte equality against an independent
implementation is the only assertion strong enough to notice.

Also worth naming is
`test_aligning_the_zero_point_reconstructs_exact_zeros_exactly`, which pins the
one place this scheme departs from the textbook formula, and the reason it does.

Run with:  python -m pytest 07_quantize -q
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from model import ModelConfig, Transformer  # noqa: E402
from quantize import (  # noqa: E402
    MAX_BITS,
    MIN_BITS,
    QuantizedEmbedding,
    QuantizedLinear,
    array_bytes,
    dequantize_groupwise,
    load_quantized,
    pack_codes,
    quantize_groupwise,
    quantize_model,
    round_trip_error,
    save_quantized,
    unpack_codes,
)

# What MLX's kernel will quantize to. Our packer also handles 7 bits, because
# writing a bit stream does not care whether the width is a nice number, but
# `mx.quantize` refuses it — so 7 is absent from every comparison below.
KERNEL_BITS = (2, 3, 4, 5, 6, 8)
KERNEL_GROUP_SIZES = (32, 64, 128)

TINY = ModelConfig(
    vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=64
)


@pytest.fixture
def weights():
    """A matrix wide enough for every group size under test."""
    mx.random.seed(0)
    return mx.random.normal((13, 256)).astype(mx.float32)


# -- the fused kernel says the same thing ---------------------------------


def test_the_packed_words_match_mlx_exactly(weights):
    """Byte-identical codes against an implementation we did not write.

    This is `test_fast_training_matches_the_slow_obvious_way` from chapter 01,
    six chapters later. Quantization is lossy on purpose, so "the numbers came
    out close" proves nothing — every wrong implementation also produces
    numbers that are close. What proves something is that a scheme derived by
    reading MLX's output and a scheme compiled into a Metal kernel agree on
    every one of the 3,328 codes, at six bit widths and three group sizes,
    *including the bit layout they are packed into*.

    The packing is the part that nearly went wrong. At 4 and 8 bits every code
    sits inside one 32-bit word, so a packer built on "codes per word" works and
    looks right; at 3, 5 and 6 bits codes straddle word boundaries and that
    packer cannot express them at all. Comparing values would have hidden it.
    Comparing words did not.

    Delete this and the module's whole claim reverts to a promise.
    """
    for bits in KERNEL_BITS:
        for group_size in KERNEL_GROUP_SIZES:
            codes, _, _ = quantize_groupwise(weights, group_size, bits)
            ours = pack_codes(codes, bits)
            theirs, _, _ = mx.quantize(weights, group_size=group_size, bits=bits)

            assert ours.shape == theirs.shape, f"bits {bits}, group {group_size}"
            assert mx.array_equal(ours, theirs), (
                f"packed words differ at bits {bits}, group_size {group_size}: "
                f"{int((ours != theirs).sum().item())} of {ours.size} words"
            )


def test_a_tie_breaks_away_from_zero_the_way_the_kernel_does():
    """Regression guard for `docs/LESSONS.md` L12 — the subtlest line in the file.

    `mx.floor(x + 0.5)` in `_round_half_away` reads like a clumsy `mx.round(x)`,
    and sooner or later somebody will tidy it into one. They are different
    functions: `mx.round` sends a tie to the nearest *even* integer, and MLX's
    quantize kernel sends it *away from zero*.

    That difference is invisible on any matrix small enough to inspect by hand —
    it cost two codes out of 688,128 random weights before it showed up at all.
    Which is why this test does not sample; it constructs the tie. The group
    below runs from -8 to 7, so the step is exactly 1.0 and the anchor is
    exactly -8, and the weight at 0.5 sits at exactly code 8.5. `mx.round` calls
    that 8 and disagrees with the kernel. Away from zero calls it 9 and agrees.

    The second half of L12 is worse and has no test here because it cannot be
    built this cleanly: the same tie inside the zero-point alignment changes the
    *scale*, and then forty-six weights in one real group come out a code apart.
    `test_the_packed_words_match_mlx_exactly` is what catches that, on data big
    enough for it to happen.
    """
    group = mx.array([[-8.0, 7.0, 0.5] + [1.0] * 61], dtype=mx.float32)
    codes, scales, biases = quantize_groupwise(group, group_size=64, bits=4)

    assert float(scales[0, 0]) == 1.0 and float(biases[0, 0]) == -8.0
    quotient = (0.5 - float(biases[0, 0])) / float(scales[0, 0])
    assert quotient == 8.5, "the fixture stopped being an exact tie; rebuild it"

    assert int(codes[0, 2]) == 9, "a tie must round away from zero, not to even"
    assert int(mx.round(mx.array(8.5)).item()) == 8, "mx.round rounds a tie to even"

    theirs, _, _ = mx.quantize(group, group_size=64, bits=4)
    assert mx.array_equal(pack_codes(codes, 4), theirs)


def test_the_scales_and_biases_match_mlx_exactly(weights):
    # Not `approx`. The scale is derived by the same arithmetic in the same
    # order, so anything short of exact equality means the derivation drifted —
    # and a scale that is nearly right is a model that is quietly worse.
    for bits in KERNEL_BITS:
        for group_size in KERNEL_GROUP_SIZES:
            _, scales, biases = quantize_groupwise(weights, group_size, bits)
            _, their_scales, their_biases = mx.quantize(
                weights, group_size=group_size, bits=bits
            )
            assert mx.array_equal(scales, their_scales), f"bits {bits}, group {group_size}"
            assert mx.array_equal(biases, their_biases), f"bits {bits}, group {group_size}"


def test_the_reconstruction_matches_mlx_to_float_noise(weights):
    """Same codes, same scales — so the decoded weights have to agree too.

    Measured before choosing the tolerance: across six bit widths x three group
    sizes, the worst disagreement was 2.384e-07. That is one float32 ulp near
    1.0 — the multiply and the add landing in a different order inside the
    kernel, not a different code.
    """
    for bits in KERNEL_BITS:
        for group_size in KERNEL_GROUP_SIZES:
            codes, scales, biases = quantize_groupwise(weights, group_size, bits)
            ours = dequantize_groupwise(codes, scales, biases, group_size)

            packed, their_scales, their_biases = mx.quantize(
                weights, group_size=group_size, bits=bits
            )
            theirs = mx.dequantize(
                packed,
                their_scales,
                their_biases,
                group_size=group_size,
                bits=bits,
                dtype=mx.float32,
            )
            assert mx.abs(ours - theirs).max().item() < 1e-6, f"bits {bits}, group {group_size}"


# -- the readable path and the shipping path ------------------------------


def test_the_quantized_matmul_matches_dequantize_then_multiply():
    """The two ways to use a 4-bit weight have to give the same answer.

    Chapter 02 keeps an explicit softmax next to MLX's fused attention and
    asserts they agree; this is that test for quantization. The readable path
    rebuilds the full float32 matrix and multiplies — obvious, and it throws
    away the entire point by allocating the memory you were trying not to
    allocate. The shipping path decodes each group inside the kernel, so the
    matrix never exists.

    Only the second one is a 4-bit model. This test is what lets the chapter
    explain with the first and ship the second.

    Measured before choosing the tolerance: across five weight seeds on a
    256 -> 96 layer, the worst relative difference was 2.73e-07.
    """
    for seed in range(5):
        mx.random.seed(seed)
        layer = nn.Linear(256, 96, bias=False)
        mx.eval(layer.parameters())

        quantized = QuantizedLinear.from_linear(layer, group_size=64, bits=4)
        x = mx.random.normal((8, 256)).astype(mx.float32)

        shipping = quantized(x)
        readable = x @ quantized.dequantized_weight().T

        relative = (
            mx.abs(shipping - readable).max().item() / mx.abs(readable).max().item()
        )
        assert relative < 1e-6, f"seed {seed}: relative {relative:.2e}"


def test_a_bias_term_survives_quantization():
    # The weight is quantized; the bias is not. It is one number per output and
    # it is added after the matmul in float32. Chapter 02's model has no biases
    # anywhere, so this path is only exercised here.
    mx.random.seed(0)
    layer = nn.Linear(128, 64, bias=True)
    mx.eval(layer.parameters())

    quantized = QuantizedLinear.from_linear(layer, group_size=64, bits=4)
    assert "bias" in quantized
    assert mx.array_equal(quantized.bias, layer.bias)

    x = mx.random.normal((4, 128)).astype(mx.float32)
    readable = x @ quantized.dequantized_weight().T + quantized.bias
    assert mx.abs(quantized(x) - readable).max().item() < 1e-5


# -- the model ------------------------------------------------------------


def _quantized_pair(group_size: int, bits: int):
    """The same model quantized twice: by hand, and by MLX's generic walker."""
    mx.random.seed(0)
    ours = Transformer(TINY)
    mx.eval(ours.parameters())

    mx.random.seed(0)
    theirs = Transformer(TINY)
    mx.eval(theirs.parameters())

    quantize_model(ours, group_size, bits)
    nn.quantize(theirs, group_size=group_size, bits=bits)
    return ours, theirs


def test_quantize_model_matches_the_generic_quantizer():
    """Walking the architecture by hand lands on MLX's model, layer for layer.

    `quantize_model` names every projection it touches, because deciding which
    matrices get four bits is the content of the chapter and a generic walker
    hides that decision. The cost of being explicit is that the list can go
    stale: a projection added to `02_model/model.py` and not added to
    `LINEAR_TARGETS` would quietly stay at float32, and the only symptom would
    be a model slightly larger and slightly *better* than it claims to be —
    which is not a symptom anyone investigates.

    This test is the guard. MLX's walker finds every `Linear` and `Embedding`
    generically, so if the two models still produce the same logits, the
    hand-written list is complete.

    Measured before choosing the tolerance: across group sizes 32/64 and all
    six kernel bit widths, the worst relative difference in the logits was
    6.39e-07.
    """
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    for group_size in (32, 64):
        for bits in KERNEL_BITS:
            ours, theirs = _quantized_pair(group_size, bits)
            our_logits, _ = ours(ids)
            their_logits, _ = theirs(ids)

            relative = (
                mx.abs(our_logits - their_logits).max().item()
                / mx.abs(their_logits).max().item()
            )
            assert relative < 1e-5, (
                f"group_size {group_size}, bits {bits}: relative {relative:.2e}"
            )


def test_quantize_model_replaces_the_matrices_and_leaves_the_norms_alone():
    # The norms are 512 floats each and they multiply the whole residual
    # stream. Quantizing them would save 34 KiB out of 95 MiB and risk the one
    # part of the model with no redundancy to spare.
    ours, _ = _quantized_pair(64, 4)
    assert isinstance(ours.embed, QuantizedEmbedding)
    assert isinstance(ours.layers[0].attn.q_proj, QuantizedLinear)
    assert isinstance(ours.layers[0].attn.o_proj, QuantizedLinear)
    assert isinstance(ours.layers[0].ffn.down_proj, QuantizedLinear)
    assert not isinstance(ours.layers[0].attn_norm, (QuantizedLinear, QuantizedEmbedding))
    assert ours.layers[0].attn_norm.weight.dtype == mx.float32
    assert ours.norm.weight.dtype == mx.float32


def test_leaving_the_embedding_alone_is_a_choice_the_caller_gets():
    # Tied embeddings mean this one array is both the token table and the
    # output head, so it is the most-used matrix in the model. Being able to
    # keep it at float32 while everything else drops to four bits is the first
    # thing to try when a quantized model comes out worse than expected.
    mx.random.seed(0)
    model = Transformer(TINY)
    mx.eval(model.parameters())
    quantize_model(model, 64, 4, quantize_embedding=False)

    assert isinstance(model.embed, nn.Embedding)
    assert not isinstance(model.embed, QuantizedEmbedding)
    assert isinstance(model.layers[0].attn.q_proj, QuantizedLinear)


def test_a_quantized_model_still_generates():
    # The KV-cache path, the tied output head and the single-token forward pass
    # all run through code that changed. Shapes staying valid is not the same
    # as generation working — that is chapter 02's L3 lesson, and it applies
    # here for the same reason.
    ours, _ = _quantized_pair(64, 4)
    tokens = list(ours.generate(mx.array([1, 2, 3]), max_tokens=4, seed=0))
    assert len(tokens) == 4
    assert all(0 <= token < TINY.vocab_size for token in tokens)


def test_quantizing_shrinks_the_model_by_the_arithmetic_it_promises():
    """The memory saving has to be bytes, not a story about bytes.

    At 4 bits with 64-weight groups the scheme costs 4 bits per weight plus two
    float32 numbers per group — 4 + 64/64 = 5 bits, against 32. That predicts
    6.4x, and the norms that stay in float32 pull it down slightly.
    `array_bytes` reads `nbytes`, so a packed uint32 array counts as the words
    it holds and not as the weights it stands for; without that this assertion
    would pass while nothing had shrunk.
    """
    mx.random.seed(0)
    model = Transformer(TINY)
    mx.eval(model.parameters())
    before = array_bytes(model.parameters())

    quantize_model(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    after = array_bytes(model.parameters())

    # Measured on this configuration: 427,264 -> 67,840 bytes, 6.298x.
    assert 6.0 < before / after < 6.5, f"{before} -> {after} is {before / after:.3f}x"


# -- on disk --------------------------------------------------------------


def test_a_saved_quantized_model_reloads_to_the_same_logits(tmp_path):
    """Chapter 08 reads this file from Swift, so the recipe travels with it.

    Nothing here may depend on the caller remembering how the model was made.
    The round trip is exact rather than approximate because no arithmetic
    happens on the way — the codes are integers and the scales are the same
    float32 numbers going out and coming back.
    """
    mx.random.seed(0)
    model = Transformer(TINY)
    mx.eval(model.parameters())
    quantize_model(model, 64, 4)

    ids = mx.array([[1, 2, 3, 4, 5]])
    before, _ = model(ids)
    mx.eval(before)

    save_quantized(tmp_path / "q4", model, group_size=64, bits=4, step=300)
    restored, meta = load_quantized(tmp_path / "q4")
    after, _ = restored(ids)

    assert mx.array_equal(before, after)
    assert meta["step"] == 300
    assert meta["quantization"] == {
        "scheme": "affine",
        "bits": 4,
        "group_size": 64,
        "embedding": True,
    }


def test_a_reloaded_model_is_quantized_rather_than_inflated(tmp_path):
    # The failure this guards against is loading a 4-bit file into a float32
    # model: every number would be right and the memory saving would be gone,
    # which is exactly the thing chapter 08 cannot afford.
    mx.random.seed(0)
    model = Transformer(TINY)
    mx.eval(model.parameters())
    quantize_model(model, 64, 4)
    packed_bytes = array_bytes(model.parameters())

    save_quantized(tmp_path / "q4", model, group_size=64, bits=4)
    restored, _ = load_quantized(tmp_path / "q4")

    assert isinstance(restored.embed, QuantizedEmbedding)
    assert isinstance(restored.layers[0].attn.q_proj, QuantizedLinear)
    assert array_bytes(restored.parameters()) == packed_bytes


# -- packing --------------------------------------------------------------


def test_packing_round_trips_exactly(weights):
    # Lossless by construction, and worth pinning: this is the one place in the
    # chapter where an off-by-one in a shift still produces plausible numbers.
    for bits in KERNEL_BITS:
        codes, _, _ = quantize_groupwise(weights, 64, bits)
        assert mx.array_equal(unpack_codes(pack_codes(codes, bits), bits), codes), f"bits {bits}"


def test_packing_refuses_a_width_it_cannot_store():
    for bits in (MIN_BITS - 1, 0, MAX_BITS + 1, 16):
        with pytest.raises(ValueError):
            pack_codes(mx.zeros((2, 64), dtype=mx.uint8), bits)
        with pytest.raises(ValueError):
            unpack_codes(mx.zeros((2, 8), dtype=mx.uint32), bits)


def test_packing_refuses_a_row_that_does_not_fill_whole_words():
    # Ten 4-bit codes is forty bits: one word and a quarter. Refusing beats
    # writing a partial word and reading garbage back out of it later.
    with pytest.raises(ValueError):
        pack_codes(mx.zeros((2, 10), dtype=mx.uint8), 4)


def test_unpacking_refuses_a_stream_that_does_not_divide_into_codes():
    # Eight words is 256 bits, which is not a whole number of 3-bit codes.
    with pytest.raises(ValueError):
        unpack_codes(mx.zeros((2, 8), dtype=mx.uint32), 3)


# -- the zero point -------------------------------------------------------

# A group with an asymmetric range and three exact zeros in it. Eight numbers,
# small enough to read the codes off the page — which is the point of using it
# rather than random weights.
ZERO_GROUP = mx.array([[-0.9, 0.0, 0.31, 0.0, -0.42, 0.17, 0.0, 0.55]])


def test_aligning_the_zero_point_reconstructs_exact_zeros_exactly():
    """Zero is the one value in a weight matrix that has to survive.

    The textbook affine formula puts the low end of the group at code 0 and
    lays fifteen even steps up to the high end. Zero then lands wherever it
    lands, which is almost never on a code. MLX instead shrinks the step until
    zero sits on an integer code exactly, and this test is why that is worth
    the trouble: padding, masks and pruned weights are all exactly zero, and a
    grid that cannot represent zero turns every one of them into a small
    constant bias that no later measurement attributes to the quantizer.

    Measured on the group below: aligned, all three zeros come back as exactly
    0.0. Textbook, all three come back as -0.03.
    """
    codes, scales, biases = quantize_groupwise(ZERO_GROUP, group_size=8, bits=4)
    restored = dequantize_groupwise(codes, scales, biases, group_size=8)

    for position in (1, 3, 6):
        assert restored[0, position].item() == 0.0, f"position {position}"


def test_the_textbook_grid_puts_zero_between_two_codes():
    codes, scales, biases = quantize_groupwise(
        ZERO_GROUP, group_size=8, bits=4, align_zero=False
    )
    restored = dequantize_groupwise(codes, scales, biases, group_size=8)

    # -0.03 on this group. Across twenty 16 x 128 Gaussian matrices with a fifth
    # of their weights forced to exact zero, quantized at 4 bits in groups of
    # 64, the worst reconstructed zero was 0.1971 — about a fifth of a typical
    # weight, applied to every zero in the model.
    for position in (1, 3, 6):
        assert restored[0, position].item() != 0.0, f"position {position}"
        assert abs(restored[0, position].item()) < 0.05


def test_the_alignment_costs_almost_nothing_on_ordinary_weights():
    """What the zero point buys is not free, and the price is worth printing.

    Anchoring zero means the far end of the range no longer sits exactly on the
    largest weight, so the step is slightly coarser than it strictly had to be.
    Measured across six seeds on Gaussian weights with no exact zeros in them,
    aligned RMS error ran 0.1% to 2.4% above textbook. That is the whole trade:
    a fraction of a percent everywhere, for exactness at the one value that is
    structurally special.
    """
    mx.random.seed(1)
    w = mx.random.normal((32, 256)).astype(mx.float32)

    errors = {}
    for align_zero in (True, False):
        codes, scales, biases = quantize_groupwise(w, 64, 4, align_zero=align_zero)
        errors[align_zero] = round_trip_error(
            w, dequantize_groupwise(codes, scales, biases, 64)
        )["rel_rms"]

    # Measured on this matrix: aligned 0.09176, textbook 0.08961 — a ratio of
    # 1.024, and the alignment is never the cheaper of the two.
    assert 1.0 < errors[True] / errors[False] < 1.05


# -- how wrong it gets ----------------------------------------------------


def test_more_bits_is_monotonically_less_error(weights):
    # Measured at group_size 64: 7.8, 14.3, 20.7, 26.9, 33.1, 45.3 dB for 2, 3,
    # 4, 5, 6 and 8 bits. Close to the textbook 6 dB per bit, which is the real
    # check here — a scheme not gaining about a factor of two in accuracy per
    # bit is wasting bits somewhere.
    signal_to_noise = []
    for bits in KERNEL_BITS:
        codes, scales, biases = quantize_groupwise(weights, 64, bits)
        error = round_trip_error(weights, dequantize_groupwise(codes, scales, biases, 64))
        signal_to_noise.append(error["snr_db"])

    assert signal_to_noise == sorted(signal_to_noise), signal_to_noise
    assert signal_to_noise[2] > 18.0  # four bits, measured at 20.7 dB


def test_smaller_groups_are_more_accurate_than_larger_ones(weights):
    """The whole argument for grouping, as an assertion.

    One scale for a 256-wide row is set by that row's single largest weight,
    and every small weight in the row then rounds toward the same few codes.
    Halving the group halves the range each scale has to cover. Measured at 4
    bits: 21.8, 20.7, 19.8, 19.1 dB for groups of 32, 64, 128 and 256.
    """
    signal_to_noise = []
    for group_size in (32, 64, 128, 256):
        codes, scales, biases = quantize_groupwise(weights, group_size, 4)
        error = round_trip_error(
            weights, dequantize_groupwise(codes, scales, biases, group_size)
        )
        signal_to_noise.append(error["snr_db"])

    assert signal_to_noise == sorted(signal_to_noise, reverse=True), signal_to_noise


def test_no_error_reads_as_no_error(weights):
    error = round_trip_error(weights, weights)
    assert error["max_abs"] == 0.0
    assert error["rms"] == 0.0
    assert error["rel_rms"] == 0.0
    assert error["snr_db"] == float("inf")


def test_the_worst_weight_is_reported_separately_from_the_typical_one(weights):
    # Measured at 4 bits, group 64 on this matrix: max_abs 0.268 against rms
    # 0.091, a factor of 2.95. Quoting either number alone tells a different
    # and incomplete story, which is why `round_trip_error` returns both.
    codes, scales, biases = quantize_groupwise(weights, 64, 4)
    error = round_trip_error(weights, dequantize_groupwise(codes, scales, biases, 64))

    assert error["max_abs"] > 2.5 * error["rms"]
    assert error["rel_rms"] == pytest.approx(
        error["rms"] / float(mx.sqrt(mx.mean(weights * weights)).item()), rel=1e-5
    )


# -- refusals -------------------------------------------------------------


def test_a_group_size_that_does_not_divide_the_row_is_refused():
    # A trailing partial group would get a scale fitted to fewer weights than
    # every other group, and nothing downstream would ever mention it.
    with pytest.raises(ValueError):
        quantize_groupwise(mx.zeros((2, 100)), 64, 4)


def test_a_bit_width_outside_the_supported_range_is_refused():
    for bits in (MIN_BITS - 1, 0, MAX_BITS + 1, 16):
        with pytest.raises(ValueError):
            quantize_groupwise(mx.zeros((2, 64)), 64, bits)


def test_a_vector_is_refused():
    # Quantizing a 1-d array would silently reshape it into one long row and
    # hand back something that looks entirely reasonable.
    with pytest.raises(ValueError):
        quantize_groupwise(mx.zeros((64,)), 64, 4)


def test_dequantizing_with_the_wrong_group_size_is_refused():
    with pytest.raises(ValueError):
        dequantize_groupwise(
            mx.zeros((2, 100), dtype=mx.uint8), mx.zeros((2, 1)), mx.zeros((2, 1)), 64
        )


def test_quantize_model_refuses_a_width_its_groups_do_not_divide():
    model = Transformer(
        ModelConfig(vocab_size=64, d_model=100, n_layers=1, n_heads=4, max_seq_len=32)
    )
    with pytest.raises(ValueError):
        quantize_model(model, group_size=64, bits=4)


# -- edges ----------------------------------------------------------------


def test_a_group_of_identical_weights_survives():
    """A constant group has no range at all, and the arithmetic divides by it.

    Worth pinning, and worth knowing what it pins. Our scale comes out as -0.0
    and MLX's as -1e-07; the codes and the reconstructed values are identical
    either way, because every code is 0 and the bias carries the whole value.
    So the assertion is on the reconstruction and on the packed words, not on
    the scale — asserting equal scales here would fail for a reason that does
    not matter.
    """
    w = mx.full((2, 64), 0.37).astype(mx.float32)
    codes, scales, biases = quantize_groupwise(w, 64, 4)
    restored = dequantize_groupwise(codes, scales, biases, 64)

    assert mx.array_equal(restored, w)
    assert mx.array_equal(pack_codes(codes, 4), mx.quantize(w, group_size=64, bits=4)[0])


def test_a_group_of_zeros_stays_zero():
    w = mx.zeros((2, 64), dtype=mx.float32)
    codes, scales, biases = quantize_groupwise(w, 64, 4)
    assert mx.abs(dequantize_groupwise(codes, scales, biases, 64)).max().item() == 0.0
    assert codes.max().item() == 0


def test_a_single_row_of_a_single_group_quantizes():
    mx.random.seed(2)
    w = mx.random.normal((1, 64)).astype(mx.float32)
    codes, scales, biases = quantize_groupwise(w, 64, 4)

    assert codes.shape == (1, 64)
    assert scales.shape == (1, 1)
    assert mx.array_equal(pack_codes(codes, 4), mx.quantize(w, group_size=64, bits=4)[0])


def test_quantization_works_on_more_than_two_dimensions():
    # Grouping runs along the last axis, so leading axes are along for the
    # ride. Every matrix in this model is 2-d, but nothing in the arithmetic
    # needs that to be true, and a shape bug would hide behind the fact that it
    # never comes up.
    mx.random.seed(3)
    w = mx.random.normal((2, 3, 64)).astype(mx.float32)
    codes, scales, biases = quantize_groupwise(w, 64, 4)

    assert codes.shape == (2, 3, 64)
    assert scales.shape == (2, 3, 1)
    assert mx.array_equal(pack_codes(codes, 4), mx.quantize(w, group_size=64, bits=4)[0])
