"""Chapter 02 — a transformer, in one file.

Nothing is imported that does the thinking for you. `nn.Linear` and
`nn.Embedding` are here because a matrix multiply is a matrix multiply, but
normalisation, positional encoding, attention and the feed-forward block are
written out. That is the part worth reading.

The architecture is the one everything modern converged on, and each piece is
here because it earned its place:

    RMSNorm      cheaper than LayerNorm, and the mean-centring turned out not
                 to matter
    RoPE         position lives in the rotation of q and k, so relative
                 distance survives to the attention scores
    GQA          several query heads share one key/value head; the KV-cache
                 gets smaller, which is the whole game on a 16 GB laptop
    SwiGLU       a gated feed-forward block that beats plain ReLU at equal
                 parameter count
    pre-norm     normalise going into each sublayer, so the residual stream
                 stays a clean path from input to output

Attention is implemented twice: once written out with an explicit softmax so
you can read it, once through MLX's fused kernel so it is fast. A test in
`test_model.py` proves the two agree. That pattern repeats through this
repository — the readable version explains, the fast version ships, and the
test is what keeps it honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ModelConfig:
    """Everything that decides the shape of the network."""

    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int | None = None  # None means one KV head per query head
    d_ff: int | None = None  # None means the usual 8/3 * d_model, rounded
    max_seq_len: int = 512
    rope_base: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True  # reuse the embedding matrix as the output head
    fused_attention: bool = True  # False selects the readable implementation

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            # Round to a multiple of 64: matmuls like it and the parameter
            # count barely moves.
            self.d_ff = 64 * round(self.d_model * 8 / 3 / 64)
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} must divide into {self.n_heads} heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads {self.n_heads} must be a multiple of n_kv_heads {self.n_kv_heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# Configurations that fit on an Apple Silicon laptop. Chapter 04 measures what
# each one actually costs; these are the starting points.
PRESETS = {
    "nano": ModelConfig(d_model=128, n_layers=4, n_heads=4, max_seq_len=256),
    "micro": ModelConfig(d_model=256, n_layers=6, n_heads=8, n_kv_heads=4),
    "small": ModelConfig(d_model=512, n_layers=8, n_heads=8, n_kv_heads=4),
    "base": ModelConfig(d_model=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=1024),
}


# -- pieces ---------------------------------------------------------------


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then apply a learned gain.

    LayerNorm subtracts the mean first. RMSNorm skips that, and models train
    just as well without it — which tells you the centring was never doing the
    work people assumed it was.
    """

    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # In float32 regardless of input dtype: the sum of squares is exactly
        # the operation that overflows float16 on long vectors.
        f = x.astype(mx.float32)
        scale = mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return (f * scale).astype(x.dtype) * self.weight


def rope(x: mx.array, offset: int = 0, base: float = 10_000.0) -> mx.array:
    """Rotary position embedding, written out.

    Take each vector, read it as pairs of coordinates, and rotate every pair by
    an angle proportional to its position in the sequence. Low-index pairs turn
    slowly, high-index pairs turn quickly.

    The reason this works is one line of trigonometry: the dot product of two
    rotated vectors depends on the *difference* of their angles, not on either
    angle alone. So after rotation, `q · k` carries how far apart the two
    tokens are. Position information reaches attention without ever being added
    to the residual stream.

    `x` is (batch, heads, seq, head_dim). `offset` is where this chunk starts
    in the full sequence, which is what makes generation with a KV-cache line
    up with a full forward pass.
    """
    *_, seq_len, head_dim = x.shape
    half = head_dim // 2

    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    # freq_i = base ** (-i / half): a geometric spread of rotation speeds.
    idx = mx.arange(half, dtype=mx.float32)
    inv_freq = mx.exp(-math.log(base) * idx / half)

    angles = positions[:, None] * inv_freq[None, :]  # (seq, half)
    cos, sin = mx.cos(angles), mx.sin(angles)

    # The two halves of the vector are the two coordinates of each pair.
    x1, x2 = x[..., :half], x[..., half:]
    rotated = mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return rotated.astype(x.dtype)


def causal_mask(n_queries: int, n_keys: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    """An additive mask: 0 where a query may look, -inf where it may not.

    During generation `n_queries` is 1 while `n_keys` keeps growing, so the
    queries sit at the *end* of the key range. Getting that alignment wrong is
    the classic KV-cache bug, and it is invisible until generation quality
    quietly degrades.
    """
    q_pos = mx.arange(n_keys - n_queries, n_keys)[:, None]
    k_pos = mx.arange(n_keys)[None, :]
    return mx.where(k_pos <= q_pos, 0.0, -mx.inf).astype(dtype)


class Attention(nn.Module):
    """Grouped-query attention with rotary positions and an optional cache."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        batch, seq_len, _ = x.shape

        # (batch, seq, heads * head_dim) -> (batch, heads, seq, head_dim)
        def split_heads(t: mx.array, heads: int) -> mx.array:
            return t.reshape(batch, seq_len, heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = split_heads(self.q_proj(x), self.n_heads)
        keys = split_heads(self.k_proj(x), self.n_kv_heads)
        values = split_heads(self.v_proj(x), self.n_kv_heads)

        # Rotate by absolute position. With a cache, this chunk starts after
        # everything already stored.
        offset = cache[0].shape[2] if cache is not None else 0
        queries = rope(queries, offset, self.config.rope_base)
        keys = rope(keys, offset, self.config.rope_base)

        if cache is not None:
            keys = mx.concatenate([cache[0], keys], axis=2)
            values = mx.concatenate([cache[1], values], axis=2)
        new_cache = (keys, values)

        if self.config.fused_attention:
            out = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        else:
            out = self._attention_written_out(queries, keys, values, mask)

        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out), new_cache

    def _attention_written_out(
        self, queries: mx.array, keys: mx.array, values: mx.array, mask: mx.array | None
    ) -> mx.array:
        """The same operation, spelled out. Slower, and the one to read."""
        # Grouped-query attention: each KV head serves several query heads, so
        # repeat the KV heads to line up. The fused kernel does this without
        # materialising anything, which is part of why it is faster.
        #
        # Faster, and only barely cheaper. Chapter 04 measured the peak with
        # the fused path at 1.236 GiB and this one at 1.257 GiB — under two
        # percent. MLX skips the score matrix on the way forward, but the
        # backward pass wants it either way. Reach for the fused kernel for
        # speed, not to fit a model that would not otherwise fit.
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            keys = mx.repeat(keys, repeats, axis=1)
            values = mx.repeat(values, repeats, axis=1)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        # Softmax in float32: exponentials of attention logits leave float16's
        # range far more easily than people expect.
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(values.dtype)
        return weights @ values


class SwiGLU(nn.Module):
    """A gated feed-forward block.

    Two projections go up, one of them through a SiLU and acting as a gate on
    the other, then one projection comes back down. The gate lets the block
    decide per-dimension how much signal to pass, and at equal parameter count
    that beats a plain ReLU sandwich.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """One transformer layer: attention, then feed-forward, both pre-normed.

    The residual stream `x` is never normalised in place — each sublayer reads
    a normalised copy and adds its result back. That is what keeps a clean
    gradient path from the loss all the way to the embeddings, and it is the
    reason deep transformers train at all.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        attn_out, new_cache = self.attn(self.attn_norm(x), mask, cache)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


# -- the model ------------------------------------------------------------


class Transformer(nn.Module):
    """Token ids in, logits over the vocabulary out."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = [Block(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        # Tied embeddings: the same matrix maps ids to vectors on the way in
        # and vectors to scores on the way out. On a small model the embedding
        # table is a large share of the parameters, so this is not a detail.
        self.lm_head = None if config.tie_embeddings else nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

    def __call__(
        self,
        ids: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        if ids.ndim == 1:
            ids = ids[None]
        batch, seq_len = ids.shape

        past = cache[0][0].shape[2] if cache is not None else 0
        if past + seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence of {past + seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        x = self.embed(ids)

        # A single query attends to everything before it, so no mask is needed
        # during token-by-token generation.
        mask = causal_mask(seq_len, past + seq_len, x.dtype) if seq_len > 1 else None

        new_cache = []
        for i, layer in enumerate(self.layers):
            x, layer_cache = layer(x, mask, cache[i] if cache is not None else None)
            new_cache.append(layer_cache)

        x = self.norm(x)
        logits = self.embed.as_linear(x) if self.lm_head is None else self.lm_head(x)
        return logits, new_cache

    @property
    def num_parameters(self) -> int:
        return sum(p.size for _, p in tree_flatten(self.parameters()))

    def generate(
        self,
        prompt: mx.array,
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int | None = None,
        seed: int | None = None,
    ):
        """Yield tokens one at a time, reusing the KV-cache.

        The prompt goes through in a single pass; after that each step feeds in
        exactly one token and the cache supplies all the history. Without the
        cache, generating n tokens costs a full forward pass over the whole
        sequence n times — which is the difference between usable and not.
        """
        if seed is not None:
            mx.random.seed(seed)
        if prompt.ndim == 1:
            prompt = prompt[None]

        logits, cache = self(prompt)
        for _ in range(max_tokens):
            token = sample(logits[:, -1, :], temperature, top_k)
            mx.eval(token)  # MLX is lazy; force this step before yielding
            yield token.item() if token.size == 1 else token
            logits, cache = self(token, cache)


def sample(logits: mx.array, temperature: float = 0.8, top_k: int | None = None) -> mx.array:
    """Pick the next token. Temperature 0 means always take the most likely one."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = logits.astype(mx.float32)
    if top_k is not None and 0 < top_k < logits.shape[-1]:
        # Keep the k best scores, push the rest out of reach of the softmax.
        kth = mx.sort(logits, axis=-1)[..., -top_k][..., None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    return mx.random.categorical(logits / temperature)[..., None]


def tree_flatten(tree, prefix: str = ""):
    """Walk MLX's nested parameter dictionaries into (path, array) pairs."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from tree_flatten(value, f"{prefix}.{key}")
    elif isinstance(tree, (list, tuple)):
        for i, value in enumerate(tree):
            yield from tree_flatten(value, f"{prefix}.{i}")
    elif isinstance(tree, mx.array):
        yield prefix, tree
