"""Chapter 05, part one — LoRA, written out.

Full fine-tuning of a 24.9M model means holding 24.9M gradients and twice that
again in optimizer moments. That fits on this laptop. A 7B model does not, and
the technique that closes the gap is worth understanding on a model small
enough to check.

The idea is one line of linear algebra. A weight update `ΔW` learned during
fine-tuning turns out to be low-rank in practice — it does not need all
`in × out` degrees of freedom. So do not store it that way. Store two thin
matrices instead:

    W' = W + (A @ B) * scale        A is (in, r),  B is (r, out),  r << in, out

`W` never moves. Only `A` and `B` train. At rank 8 on this model that is 0.3%
of the parameters, and you can keep one frozen base model in memory with
several adapters swapping in and out on top of it.

The detail that makes it work at all: **B starts at zero.** So `A @ B` is zero,
`W'` equals `W` exactly, and fine-tuning begins from the model you already have
rather than from a random perturbation of it. There is a test for that, because
it is the kind of thing that is easy to get subtly wrong and hard to notice —
the loss still falls either way, it just falls from the wrong place.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

# Which projections get an adapter. Attention's query and value projections are
# the usual choice: the original paper found them enough, and they are a small
# share of the parameters. `apply_lora` takes this as an argument, so try
# others — chapter 05's notes have what changed when I did.
DEFAULT_TARGETS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` with a trainable low-rank update beside it."""

    def __init__(self, base: nn.Linear, rank: int = 8, scale: float = 20.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        out_dim, in_dim = base.weight.shape
        if rank > min(in_dim, out_dim):
            raise ValueError(
                f"rank {rank} is larger than the {in_dim}x{out_dim} layer it adapts; "
                "at that point train the layer directly"
            )

        self.base = base
        self.base.freeze()

        self.rank = rank
        self.scale = scale
        # A is initialised the way a Linear would be, B at zero. Both matter:
        # zero A would kill the gradient to B, and non-zero B would move the
        # model before training starts.
        bound = 1 / math.sqrt(in_dim)
        self.lora_a = mx.random.uniform(-bound, bound, (in_dim, rank))
        self.lora_b = mx.zeros((rank, out_dim))

    def __call__(self, x: mx.array) -> mx.array:
        # x @ A is (…, rank) — the whole point is that the wide multiply never
        # happens. Cost is O(in*r + r*out) instead of O(in*out).
        update = (x @ self.lora_a) @ self.lora_b
        return self.base(x) + self.scale * update

    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain Linear.

        For serving, or for chapter 07. Once merged there is no adapter and no
        extra multiply — the model is exactly the shape it started as, with
        different numbers in it.
        """
        out_dim, in_dim = self.base.weight.shape
        merged = nn.Linear(in_dim, out_dim, bias="bias" in self.base)
        # base.weight is (out, in); the adapter builds (in, out), so transpose.
        merged.weight = self.base.weight + self.scale * (self.lora_a @ self.lora_b).T
        if "bias" in self.base:
            merged.bias = self.base.bias
        return merged


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    scale: float = 20.0,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> nn.Module:
    """Freeze the model and hang adapters off the named projections.

    Walking the blocks by hand rather than reaching for a generic module
    rewriter: we wrote this architecture two chapters ago, and being explicit
    about which layers get adapted is the point of the exercise.
    """
    model.freeze()
    adapted = 0

    for block in model.layers:
        for holder in (block.attn, block.ffn):
            for name in targets:
                layer = getattr(holder, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(holder, name, LoRALinear(layer, rank, scale))
                    adapted += 1

    if adapted == 0:
        raise ValueError(f"no layers matched {targets}; nothing would train")
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter in the model back into plain Linear layers."""
    for block in model.layers:
        for holder in (block.attn, block.ffn):
            # MLX's Module is a dict subclass and submodules live in the dict,
            # not in __dict__ — `vars(holder)` returns the plain attributes and
            # none of the layers. Iterate the module itself.
            for name in list(holder):
                layer = holder[name]
                if isinstance(layer, LoRALinear):
                    setattr(holder, name, layer.merged())
    return model


def count_trainable(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts — the number LoRA exists to change."""

    def total(tree) -> int:
        if isinstance(tree, dict):
            return sum(total(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(total(v) for v in tree)
        if isinstance(tree, mx.array):
            return tree.size
        return 0

    return total(model.trainable_parameters()), total(model.parameters())
