"""Chapter 04, part one — predicting what a configuration costs.

Before you run a config you want to know whether it will fit, and "try it and
see" is a slow way to find out on a machine whose failure mode is the whole
system swapping. So this file does the arithmetic.

Four things live in memory while you train:

    parameters      what you are learning
    gradients       one per parameter, so the same size again
    optimizer       AdamW keeps two running estimates, so twice more again
    activations     every intermediate tensor the backward pass needs to see
                    a second time

The first three are exact. You can count them off the model definition and be
right to the byte — `test_scale.py` checks that against the real model, and it
matches for every preset.

The fourth is not exact and cannot be. It depends on what MLX's autodiff
decides to keep alive and when it frees things, which is a property of the
framework, not of your model. So the activation term here is a **measured**
model, not a derived one: the shape of the formula comes from reading the
code, and the three coefficients were fit against 27 real training
configurations on an M1 Pro. The method and the leftover error are written
down below, because a calibration you cannot see is worse than no calibration.

A note on dtype: this repository trains in float32, four bytes per element
throughout. That gives 16 bytes per parameter for the first three categories,
which is the same figure chapter 00 quotes for mixed-precision Adam. Same
number, different reason — do not read either as confirming the other.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_model"))

from model import ModelConfig  # noqa: E402

GIB = 1024**3


# --- the activation model ------------------------------------------------
#
# The structure below comes from reading `02_model/model.py` and counting the
# tensors each layer saves for the backward pass.
#
# Per layer, in units of (batch * seq_len * d_model) — the residual stream:
#
#   1  attn_norm output          5  o_proj output
#   2  q_proj output             6  residual after attention
#   3  rope(q)                   7  ffn_norm output
#   4  attention output          8  residual after the feed-forward block
FORWARD_RESIDUAL_TENSORS = 8

# Per layer, in units of (batch * seq_len * d_kv), d_kv = n_kv_heads * head_dim.
# k_proj output, v_proj output, rope(k). This is the term grouped-query
# attention shrinks.
FORWARD_KV_TENSORS = 3

# Per layer, in units of (batch * seq_len * d_ff):
# gate_proj output, up_proj output, and their product after the SiLU.
FORWARD_FFN_TENSORS = 3

# --- and the three coefficients that were fit to measurement -------------
#
# Method: 27 configurations (presets nano/micro/small x batch 4/8/16 x
# seq_len 128/256/512), each run for real training steps with peak memory read
# from `mx.get_peak_memory()`. The exact parameter/gradient/optimizer bytes
# were subtracted, and what remained was fit by least squares against three
# terms that vary independently of each other.
#
# An earlier attempt fit one coefficient per tensor category and produced a
# beautiful fit with *negative* tensor counts — the layer terms are nearly
# proportional to one another across these presets, so the fit could trade
# them off freely. A model that fits with negative counts is a curve, not a
# model. Collapsing the layer terms into one multiplier fixed it.

# The backward pass allocates a cotangent for roughly every saved forward
# tensor, so the layer term slightly more than doubles. Fit: 2.18.
LAYER_BACKWARD_MULTIPLIER = 2.2

# In units of (batch * seq_len * vocab_size). Cross entropy is expensive at the
# head: the logits, the log-sum-exp reduction, the gather, and gradients for
# each. On a small model with a large vocabulary this single term can outweigh
# everything in the layers, which catches people out. Fit: 10.08.
LOGIT_TENSORS = 10

# In units of (batch * n_heads * seq_len * seq_len), with no factor of
# n_layers: the backward pass walks layer by layer and frees each layer's
# scores before reaching the next. Fit: 12.79.
#
# Measured, and worth knowing: switching `fused_attention` off barely moves the
# peak (1.236 GiB against 1.257 GiB on micro at batch 8 x 256). MLX's fused
# kernel avoids materialising the score matrix in the forward pass, but the
# backward pass needs it either way. So this term does not depend on which
# attention implementation you chose — the fused kernel buys speed here, not
# memory.
SCORE_TENSORS = 12

# What is left over after all that. The coefficients were fit on 27
# configurations; the honest number comes from the wider sweep that followed,
# 44 configurations across all four presets:
#
#     measured / predicted   0.52x to 1.24x
#
# Read that as: the estimate under-predicts by at most about 20%, and
# over-predicts by up to a factor of two. The over-prediction concentrates at
# large batch and long context, where the attention-score term runs ahead of
# what MLX actually keeps alive.
#
# Both directions were left as they are on purpose. Over-predicting is the safe
# error for a "will this fit" tool — it declines a config that would have run
# rather than sending your machine into swap. Do not tune it tighter without
# re-running `sweep.py --full` and updating this number with what you saw.
PREDICTION_ERROR_BAND = (0.52, 1.24)  # measured / predicted


@dataclass
class MemoryEstimate:
    """What a configuration is predicted to cost, broken down."""

    parameters: int
    gradients: int
    optimizer: int
    activations: int
    kv_cache: int
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def training_bytes(self) -> int:
        """Training does not use the KV-cache: every position is computed at once."""
        return self.parameters + self.gradients + self.optimizer + self.activations

    @property
    def inference_bytes(self) -> int:
        """Generation needs no gradients or optimizer state, but does need the cache."""
        return self.parameters + self.kv_cache

    def bytes_for(self, mode: str = "training") -> int:
        if mode == "training":
            return self.training_bytes
        if mode == "inference":
            return self.inference_bytes
        raise ValueError(f"mode must be 'training' or 'inference', got {mode!r}")

    def fits_in(self, gib: float, mode: str = "training") -> bool:
        """Would this fit in `gib` gibibytes of usable memory?

        Against the raw prediction, deliberately. The caller decides how much
        margin to leave; see `PREDICTION_ERROR_BAND` for how much the
        prediction can be off by.
        """
        return self.bytes_for(mode) <= gib * GIB

    def as_gib(self, mode: str = "training") -> float:
        return self.bytes_for(mode) / GIB


def count_parameters(config: ModelConfig) -> int:
    """Count parameters exactly, straight off the model definition.

    Every linear layer here is `bias=False`, so it is one matrix and nothing
    else. Read this next to `02_model/model.py` and each line has an obvious
    counterpart.
    """
    d_model = config.d_model
    d_kv = config.n_kv_heads * config.head_dim

    embedding = config.vocab_size * d_model

    # q_proj and o_proj are square, because n_heads * head_dim == d_model.
    attention = 2 * d_model * d_model + 2 * d_kv * d_model
    feed_forward = 3 * d_model * config.d_ff
    norms = 2 * d_model  # attn_norm and ffn_norm, one gain per feature
    per_layer = attention + feed_forward + norms

    final_norm = d_model
    # Tied embeddings reuse the table above rather than allocating a second.
    output_head = 0 if config.tie_embeddings else config.vocab_size * d_model

    return embedding + config.n_layers * per_layer + final_norm + output_head


def layer_width(config: ModelConfig) -> int:
    """Forward-saved elements per layer, per token — the structural part."""
    d_kv = config.n_kv_heads * config.head_dim
    return (
        FORWARD_RESIDUAL_TENSORS * config.d_model
        + FORWARD_KV_TENSORS * d_kv
        + FORWARD_FFN_TENSORS * config.d_ff
    )


def estimate(
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 4,
) -> MemoryEstimate:
    """Predict the memory a training step and a full-context cache will need."""
    if seq_len > config.max_seq_len:
        raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {config.max_seq_len}")

    d_kv = config.n_kv_heads * config.head_dim
    tokens = batch_size * seq_len

    n_params = count_parameters(config)
    parameters = n_params * dtype_bytes
    gradients = parameters  # one gradient per parameter
    optimizer = 2 * parameters  # AdamW keeps a first and a second moment

    layers = LAYER_BACKWARD_MULTIPLIER * config.n_layers * tokens * layer_width(config)
    logits = LOGIT_TENSORS * tokens * config.vocab_size
    scores = SCORE_TENSORS * batch_size * config.n_heads * seq_len * seq_len

    activations = int((layers + logits + scores) * dtype_bytes)

    # Two tensors per layer, K and V, held for every position in the context.
    # Halving the KV heads halves this, which is the entire argument for GQA.
    kv_cache = 2 * config.n_layers * batch_size * config.max_seq_len * d_kv * dtype_bytes

    return MemoryEstimate(
        parameters=parameters,
        gradients=gradients,
        optimizer=optimizer,
        activations=activations,
        kv_cache=kv_cache,
        breakdown={
            "n_parameters": n_params,
            "layers": int(layers * dtype_bytes),
            "logits": int(logits * dtype_bytes),
            "scores": int(scores * dtype_bytes),
        },
    )


def describe(config: ModelConfig, batch_size: int, seq_len: int) -> str:
    """A human-readable breakdown, for when a number looks wrong."""
    est = estimate(config, batch_size, seq_len)
    low, high = PREDICTION_ERROR_BAND
    lines = [
        f"{config.n_layers} layers x {config.d_model}, "
        f"{config.n_heads} query heads / {config.n_kv_heads} KV heads, "
        f"vocab {config.vocab_size:,}",
        f"batch {batch_size} x {seq_len} tokens = {batch_size * seq_len:,} tokens per step",
        "",
        "  exact:",
        f"    parameters      {est.breakdown['n_parameters'] / 1e6:>8.1f}M "
        f"->{est.parameters / 1024**2:>8.0f} MiB",
        f"    gradients                  ->{est.gradients / 1024**2:>8.0f} MiB",
        f"    optimizer (AdamW)          ->{est.optimizer / 1024**2:>8.0f} MiB",
        "",
        "  estimated:",
        f"    layer activations          ->{est.breakdown['layers'] / 1024**2:>8.0f} MiB",
        f"    logits and cross entropy   ->{est.breakdown['logits'] / 1024**2:>8.0f} MiB",
        f"    attention scores           ->{est.breakdown['scores'] / 1024**2:>8.0f} MiB",
        "",
        f"  training total               ->{est.as_gib('training'):>8.2f} GiB",
        f"    measured, on past evidence,   {est.as_gib('training') * low:>6.2f}"
        f" to {est.as_gib('training') * high:>5.2f} GiB",
        "",
        f"  KV-cache at full context     ->{est.kv_cache / 1024**2:>8.0f} MiB",
        f"  inference total              ->{est.as_gib('inference'):>8.2f} GiB",
    ]
    return "\n".join(lines)


def main() -> int:
    import argparse

    from model import PRESETS

    parser = argparse.ArgumentParser(description="Predict what a configuration costs.")
    parser.add_argument("--preset", choices=list(PRESETS), default="small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    print(describe(PRESETS[args.preset], args.batch_size, args.seq_len))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
