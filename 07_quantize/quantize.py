"""Chapter 07, part one — four bits, written out.

A trained model is a pile of float32 numbers. This one is 24.9M of them, which
is 95 MiB, which is more than an iPhone wants to hold in memory for a toy. So
we throw most of each number away and keep the model.

The scheme is group-wise affine quantization, and it is simpler than it sounds:

    take 64 consecutive weights
    find the smallest and largest
    lay 16 evenly spaced levels across that range
    store each weight as which level it is nearest — four bits
    keep the scale and the offset in float32, once per group

So sixty-four float32 weights (256 bytes) become sixty-four 4-bit codes plus
two float32 numbers (32 + 8 = 40 bytes). Call it 5 bits per weight all in,
against 32. The group is what makes it work: one scale for a whole 512-wide row
would be dominated by the row's single largest weight, and every small weight
in it would round to the same code.

Two implementations, as everywhere in this repository. `quantize_groupwise`
below is the readable one — plain arithmetic, one code per byte, nothing packed.
`mx.quantize` is the fused kernel that ships. `test_quantize.py` asserts they
produce **byte-identical codes**, which is a stronger claim than "close enough"
and the only one worth making about a lossy transform.

## The part that surprised me

The textbook affine formula puts the low end of the group at code 0:

    scale = (hi - lo) / 15,   bias = lo,   code = round((w - lo) / scale)

MLX does not do that, and the difference is not an implementation detail. MLX
picks the endpoint with the **larger magnitude**, anchors code 0 there, and then
shrinks the step so that **exactly zero lands on an integer code**. Zero is the
only value in a weight matrix that is special: it is what padding, masks and
pruned weights are, and a scheme that cannot represent it exactly leaks a small
constant error into every one of them.

`align_zero=True` reproduces MLX's choice, down to the last bit. `False` is the
textbook version, kept because chapter 07's notes measure what the alignment
costs and what it buys, and because a claim about two schemes needs both of them
runnable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from model import ModelConfig, Transformer  # noqa: E402

# 64 weights per scale is MLX's default and a reasonable place to stand: 32 is
# measurably more accurate and costs twice the scale storage, 128 saves little
# and starts to hurt. Chapter 07's notes have the measurement.
GROUP_SIZE = 64
BITS = 4

# Widths that make sense to store. Below 2 there is one level and no model;
# above 8 a code is larger than the uint8 this file counts in and you should be
# asking why you are quantizing at all.
MIN_BITS, MAX_BITS = 2, 8


# -- the scheme -----------------------------------------------------------


def _round_half_away(x: mx.array) -> mx.array:
    """Round to the nearest integer, breaking a tie away from zero.

    `mx.round` breaks ties toward the even integer. MLX's quantize *kernel*
    breaks them away from zero, and quantization is full of exact ties — a
    group whose range divides evenly puts weights precisely half way between
    two codes. Get the convention wrong and you agree with the kernel on
    99.9999% of a matrix, which reads as agreement right up until it does not.

    Everything this file rounds is non-negative (see the call sites), so adding
    a half and taking the floor is exactly that rule. `docs/LESSONS.md` L12.
    """
    return mx.floor(x + 0.5)


def quantize_groupwise(
    w: mx.array,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    align_zero: bool = True,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize `w` in groups along its last axis.

    Returns `(codes, scales, biases)`. Codes are one `uint8` per weight — the
    readable form, not the storage form; `pack_codes` turns them into the
    uint32 words a 4-bit model actually holds.

    Reconstruction is `code * scale + bias`, so the two float32 numbers per
    group are the whole decoder.
    """
    if w.ndim < 2:
        raise ValueError(f"quantize a matrix, not a {w.ndim}-d array")
    if not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be between {MIN_BITS} and {MAX_BITS}, got {bits}")
    if w.shape[-1] % group_size:
        raise ValueError(
            f"last axis {w.shape[-1]} is not a multiple of group_size {group_size}; "
            "every group has to be complete or its scale means nothing"
        )

    levels = 2**bits - 1
    groups = w.reshape(-1, group_size).astype(mx.float32)

    lo = groups.min(axis=1, keepdims=True)
    hi = groups.max(axis=1, keepdims=True)
    step = (hi - lo) / levels

    if align_zero:
        # Anchor code 0 at whichever endpoint is further from zero, then move
        # the step size until zero itself lands exactly on some integer code.
        #
        # `step` goes negative when the anchor is the upper endpoint: codes then
        # count *downward* from hi. That is not a trick, it is the only way to
        # keep code 0 at the anchor while the range runs the other way, and it
        # is why the scales MLX hands back are sometimes negative.
        anchor_is_low = mx.abs(lo) > mx.abs(hi)
        bias = mx.where(anchor_is_low, lo, hi)
        step = mx.where(anchor_is_low, step, -step)

        # Which (fractional) code does zero sit on? Round it to an integer and
        # rescale so that it sits there exactly. A group that is entirely zero
        # leaves `step` alone rather than dividing by nothing.
        #
        # `-bias / step` is never negative: `bias` is whichever endpoint is
        # furthest from zero, so the group always lies on one side of it.
        safe = mx.where(step == 0, 1.0, step)
        zero_code = _round_half_away(-bias / safe)
        step = mx.where(zero_code == 0, step, -bias / mx.where(zero_code == 0, 1.0, zero_code))
    else:
        # The textbook version: low end at code 0, 15 even steps up to the high
        # end, zero wherever it happens to fall.
        bias = lo

    # Same tie rule as above, and for the same measured reason: a weight sitting
    # exactly half way between two codes is not a rare accident here, it is what
    # a group with an evenly dividing range produces.
    safe = mx.where(step == 0, 1.0, step)
    codes = mx.clip(_round_half_away((groups - bias) / safe), 0, levels).astype(mx.uint8)

    tail = w.shape[-1] // group_size
    return (
        codes.reshape(w.shape),
        step.reshape(*w.shape[:-1], tail),
        bias.reshape(*w.shape[:-1], tail),
    )


def dequantize_groupwise(
    codes: mx.array, scales: mx.array, biases: mx.array, group_size: int = GROUP_SIZE
) -> mx.array:
    """Rebuild float32 weights from codes and their per-group scale and offset.

    The inverse is one multiply and one add, which is the point: whatever the
    encoder did, decoding has to be cheap enough to happen inside a matmul.
    """
    if codes.shape[-1] % group_size:
        raise ValueError(f"last axis {codes.shape[-1]} is not a multiple of {group_size}")

    groups = codes.reshape(-1, group_size).astype(mx.float32)
    scale = scales.reshape(-1, 1).astype(mx.float32)
    bias = biases.reshape(-1, 1).astype(mx.float32)
    return (groups * scale + bias).reshape(codes.shape)


# -- storage --------------------------------------------------------------


def pack_codes(codes: mx.array, bits: int = BITS) -> mx.array:
    """Pack `bits`-wide codes into uint32 words, low code in the low bits.

    Until this happens the "4-bit" model is a `uint8` array spending eight bits
    on a four-bit code, and the memory saving is a claim rather than a fact.

    The layout is the obvious one and it is worth saying out loud, because the
    first version of this function assumed something else: the codes are one
    long **bit stream** per row, code `j` occupying bits `j*bits` through
    `j*bits + bits`, and the words are windows onto that stream. At 4 or 8 bits
    every code sits inside a single word, which is why it is tempting to write
    the packer that way — but at 3, 5 or 6 bits codes straddle word boundaries,
    and a packer built on "codes per word" simply cannot express them.

    Writing the bits out and folding them back into words handles every width
    with the same four lines, and matches `mx.quantize` exactly at all of them.
    """
    if not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be between {MIN_BITS} and {MAX_BITS}, got {bits}")

    total_bits = codes.shape[-1] * bits
    if total_bits % 32:
        raise ValueError(
            f"{codes.shape[-1]} codes of {bits} bits is {total_bits} bits, which does not "
            "fill a whole number of 32-bit words"
        )

    # (..., n, bits): bit i of code j. Then flatten, and the stream is in the
    # order the format wants it — no index arithmetic anywhere.
    spread = (codes[..., None].astype(mx.uint32) >> mx.arange(bits, dtype=mx.uint32)) & 1
    stream = spread.reshape(*codes.shape[:-1], total_bits)

    words = stream.reshape(*codes.shape[:-1], total_bits // 32, 32)
    return (words << mx.arange(32, dtype=mx.uint32)).sum(axis=-1).astype(mx.uint32)


def unpack_codes(packed: mx.array, bits: int = BITS) -> mx.array:
    """The inverse of `pack_codes`: uint32 words back to one code per byte."""
    if not MIN_BITS <= bits <= MAX_BITS:
        raise ValueError(f"bits must be between {MIN_BITS} and {MAX_BITS}, got {bits}")

    total_bits = packed.shape[-1] * 32
    if total_bits % bits:
        raise ValueError(f"{total_bits} bits does not divide into codes of {bits} bits")

    stream = (packed[..., None] >> mx.arange(32, dtype=mx.uint32)) & 1
    stream = stream.reshape(*packed.shape[:-1], total_bits)

    spread = stream.reshape(*packed.shape[:-1], total_bits // bits, bits)
    return (spread << mx.arange(bits, dtype=mx.uint32)).sum(axis=-1).astype(mx.uint8)


# -- what it costs you ----------------------------------------------------


def round_trip_error(original: mx.array, restored: mx.array) -> dict:
    """How far the weights moved, in four ways that answer different questions.

    `max_abs` is the worst single weight, `rms` the typical one, `rel_rms` that
    same error as a fraction of the weights' own size, and `snr_db` the same
    ratio on a log scale because that is how everyone else quotes it.

    Report more than one. A max on its own is a single unlucky weight in a
    matrix of millions; an RMS on its own hides the outlier that a max would
    have shown you.
    """
    original = original.astype(mx.float32)
    restored = restored.astype(mx.float32)
    error = original - restored

    signal = float(mx.sqrt(mx.mean(original * original)).item())
    noise = float(mx.sqrt(mx.mean(error * error)).item())
    return {
        "max_abs": float(mx.abs(error).max().item()),
        "rms": noise,
        "rel_rms": noise / signal if signal else 0.0,
        "snr_db": 20.0 * mx.log10(mx.array(signal / noise)).item() if noise else float("inf"),
    }


def array_bytes(tree) -> int:
    """Total bytes held by every array in an MLX parameter tree.

    Counted from `nbytes`, so a packed uint32 array counts as the four bytes per
    word it really is and not as the sixteen weights it stands for. That is the
    only way the before-and-after numbers in chapter 07 mean anything.
    """
    if isinstance(tree, dict):
        return sum(array_bytes(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return sum(array_bytes(v) for v in tree)
    if isinstance(tree, mx.array):
        return tree.nbytes
    return 0


# -- the layers -----------------------------------------------------------


class QuantizedLinear(nn.Module):
    """A `nn.Linear` whose weight is packed 4-bit codes plus per-group scales.

    Two ways to use the weight, and the difference is the whole chapter:

        dequantize, then matmul     rebuilds the full float32 matrix first, so
                                    you are back to the memory you were trying
                                    to avoid — but every step is readable
        quantized_matmul            unpacks each group inside the kernel, one
                                    tile at a time, so the float32 matrix never
                                    exists anywhere

    Only the second one is a 4-bit model. The first is a 4-bit *file* being
    inflated back into a float32 model, which is a different and much less
    useful thing. `test_quantize.py` asserts the two agree.
    """

    def __init__(
        self,
        packed: mx.array,
        scales: mx.array,
        biases: mx.array,
        group_size: int = GROUP_SIZE,
        bits: int = BITS,
        bias: mx.array | None = None,
    ) -> None:
        super().__init__()
        self.weight = packed
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        if bias is not None:
            self.bias = bias
        self.freeze()

    @classmethod
    def from_linear(
        cls, layer: nn.Linear, group_size: int = GROUP_SIZE, bits: int = BITS
    ) -> "QuantizedLinear":
        codes, scales, biases = quantize_groupwise(layer.weight, group_size, bits)
        return cls(
            pack_codes(codes, bits),
            scales,
            biases,
            group_size,
            bits,
            layer.bias if "bias" in layer else None,
        )

    def dequantized_weight(self) -> mx.array:
        """The float32 matrix this layer stands for. For reading, not for serving."""
        return dequantize_groupwise(
            unpack_codes(self.weight, self.bits), self.scales, self.biases, self.group_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        return out + self.bias if "bias" in self else out


class QuantizedEmbedding(nn.Module):
    """The token table, quantized — and it earns its place twice over.

    On this model the embedding is 4,097 x 512, about 8% of the parameters. It
    is also the output head, because chapter 02 ties them: the same matrix maps
    ids to vectors going in and vectors to scores coming out. So quantizing it
    shrinks one array and speeds up two operations.

    Lookup dequantizes only the rows the batch asked for. Sixteen rows out of
    four thousand, decoded on the way past.
    """

    def __init__(
        self,
        packed: mx.array,
        scales: mx.array,
        biases: mx.array,
        group_size: int = GROUP_SIZE,
        bits: int = BITS,
    ) -> None:
        super().__init__()
        self.weight = packed
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.freeze()

    @classmethod
    def from_embedding(
        cls, layer: nn.Embedding, group_size: int = GROUP_SIZE, bits: int = BITS
    ) -> "QuantizedEmbedding":
        codes, scales, biases = quantize_groupwise(layer.weight, group_size, bits)
        return cls(pack_codes(codes, bits), scales, biases, group_size, bits)

    def __call__(self, ids: mx.array) -> mx.array:
        return dequantize_groupwise(
            unpack_codes(self.weight[ids], self.bits),
            self.scales[ids],
            self.biases[ids],
            self.group_size,
        )

    def as_linear(self, x: mx.array) -> mx.array:
        """The tied output head. Same weights, read as a projection to the vocabulary."""
        return mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )


# -- the model ------------------------------------------------------------

# The norms keep their float32 weights. Together they are 8 x 2 x 512 + 512
# floats — 34 KiB out of 95 MiB, and they multiply the entire residual stream,
# so there is nothing to win and something real to lose. Quantize the big
# matrices; leave the small sensitive ones alone. That rule is not specific to
# this model.
LINEAR_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quantize_model(
    model: Transformer,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    quantize_embedding: bool = True,
) -> Transformer:
    """Swap every projection in the model for its quantized counterpart.

    Walking the architecture by hand, the way chapter 05 hangs LoRA adapters:
    we wrote this model five chapters ago, and being explicit about which
    matrices get four bits is the entire content of the decision. MLX ships
    `nn.quantize(model)`, which does this generically in one line —
    `test_quantize.py` runs both and asserts the logits match, so the readable
    version has a receipt rather than a promise.

    The model is modified in place and returned, and it is inference-only
    afterwards. There is no gradient through a lookup table of codes.
    """
    if model.config.d_model % group_size:
        raise ValueError(
            f"d_model {model.config.d_model} is not a multiple of group_size {group_size}"
        )

    quantized = 0
    for block in model.layers:
        for holder in (block.attn, block.ffn):
            for name in LINEAR_TARGETS:
                layer = getattr(holder, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(holder, name, QuantizedLinear.from_linear(layer, group_size, bits))
                    quantized += 1

    if quantize_embedding:
        model.embed = QuantizedEmbedding.from_embedding(model.embed, group_size, bits)
        quantized += 1
    if model.lm_head is not None:
        model.lm_head = QuantizedLinear.from_linear(model.lm_head, group_size, bits)
        quantized += 1

    if quantized == 0:
        raise ValueError("nothing was quantized; check the model's layer names")

    model.eval()
    return model


# -- on disk --------------------------------------------------------------
#
# Chapter 08 loads what this writes, from Swift, so the format has to carry
# everything needed to rebuild the model without guessing: the architecture, the
# group size, the bit width, and whether the embedding was quantized too. A
# checkpoint that needs a human to remember how it was made is not a checkpoint.


def save_quantized(
    directory: Path,
    model: Transformer,
    group_size: int = GROUP_SIZE,
    bits: int = BITS,
    quantized_embedding: bool = True,
    step: int | None = None,
) -> Path:
    """Write a quantized model and the recipe for reconstructing it."""
    directory.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(directory / "weights.safetensors"), weights)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "step": step,
                "model_config": asdict(model.config),
                "quantization": {
                    "scheme": "affine",
                    "bits": bits,
                    "group_size": group_size,
                    "embedding": quantized_embedding,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory


def load_quantized(directory: Path) -> tuple[Transformer, dict]:
    """Rebuild a quantized model from `save_quantized`'s output.

    The shell is built first and the weights are poured into it: quantizing a
    freshly initialised model produces arrays of exactly the right shapes and
    dtypes, and `update` then replaces the values. Ten lines shorter than
    reconstructing each layer from the file, and it cannot drift out of step
    with `quantize_model` because it calls it.
    """
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    spec = meta["quantization"]

    model = Transformer(ModelConfig(**meta["model_config"]))
    quantize_model(model, spec["group_size"], spec["bits"], spec["embedding"])
    model.update(
        tree_unflatten(list(mx.load(str(directory / "weights.safetensors")).items()))
    )
    mx.eval(model.parameters())
    model.eval()
    return model, meta
