"""Chapter 05, part two — supervised fine-tuning.

Chapter 03 trained the model to predict the next token of everything. That is
how it learned English, and it is also why it will not stop talking: nothing in
that objective ever rewarded finishing.

Supervised fine-tuning changes the objective, not the model. You show it
conversations and you score it **only on the parts it was supposed to write**.
The prompt is context, not homework. Every token of it is masked out of the
loss.

That masking is the entire chapter. Get it wrong — score the prompt too — and
the model learns to generate plausible *questions*, which is a failure that
looks like success right up until you try to use it.

The dataset here is built from the corpus you already have: take a line, cut it
in half, ask for the second half. Small, honest, and it makes the point. A 25M
model trained on a megabyte of Shakespeare is not going to learn facts from
forty examples. What it can learn — and what you can watch it learn — is the
*shape* of a response: answer, then stop.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(REPO_ROOT / "01_tokenizer"))
sys.path.insert(0, str(REPO_ROOT / "02_model"))
sys.path.insert(0, str(REPO_ROOT / "03_train"))

from bpe import BPETokenizer  # noqa: E402
from lora import apply_lora, count_trainable  # noqa: E402
from model import Transformer  # noqa: E402
from train import learning_rate_at, load_checkpoint  # noqa: E402
from train import TrainConfig  # noqa: E402

# Plain-text role markers rather than new special tokens. Adding real special
# tokens means widening the embedding table of a model that is already trained,
# and that is a different chapter's worth of care. These encode fine with the
# tokenizer from chapter 01, and the model learns them as ordinary text — which
# is exactly what smaller instruction-tuned models did for years.
PROMPT_TEMPLATE = "User: {prompt}\nAssistant:"
END_OF_TEXT = "<|endoftext|>"


def format_example(prompt: str, response: str) -> tuple[str, str]:
    """Split a conversation into (context, what the model must produce)."""
    return PROMPT_TEMPLATE.format(prompt=prompt), f" {response}{END_OF_TEXT}"


def encode_example(
    tokenizer: BPETokenizer, prompt: str, response: str, max_len: int
) -> tuple[list[int], list[int]]:
    """Return token ids and a loss mask that is 1 only on the response."""
    context, answer = format_example(prompt, response)
    context_ids = tokenizer.encode(context)
    answer_ids = tokenizer.encode(answer, allowed_special="all")

    ids = (context_ids + answer_ids)[:max_len]
    # The mask lines up with the *targets*, which are the inputs shifted left by
    # one. Position i of the target is token i+1, so the first response token
    # to score sits at index len(context_ids) - 1.
    mask = [0] * len(ids)
    for i in range(max(0, len(context_ids) - 1), len(ids)):
        mask[i] = 1
    return ids, mask


def make_dataset(text: str, count: int, seed: int = 0) -> list[dict]:
    """Build a small instruction set out of the corpus itself.

    Take a line with enough words in it, cut it in half, and ask for the rest.
    No annotation, no external download, and the task is genuinely learnable by
    a model this size — which matters, because a chapter you cannot watch work
    teaches nothing.
    """
    rng = random.Random(seed)
    lines = [line.strip() for line in text.splitlines() if len(line.split()) >= 8]
    rng.shuffle(lines)

    examples = []
    for line in lines[:count]:
        words = line.split()
        cut = len(words) // 2
        examples.append(
            {
                "prompt": "Continue this line: " + " ".join(words[:cut]),
                "response": " ".join(words[cut:]),
            }
        )
    return examples


def collate(
    batch: list[tuple[list[int], list[int]]], pad_id: int = 0
) -> tuple[mx.array, mx.array, mx.array]:
    """Pad a batch to a common length and return (inputs, targets, mask).

    Right-padding is safe here without an attention mask: attention is causal,
    so a padded position can only influence positions after it, and every one
    of those is zero in the loss mask. Left-padding would need real care.
    """
    width = max(len(ids) for ids, _ in batch)
    inputs, targets, masks = [], [], []

    for ids, mask in batch:
        padded = ids + [pad_id] * (width - len(ids))
        padded_mask = mask + [0] * (width - len(mask))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
        masks.append(padded_mask[1:])

    return (
        mx.array(np.array(inputs, dtype=np.int32)),
        mx.array(np.array(targets, dtype=np.int32)),
        mx.array(np.array(masks, dtype=np.float32)),
    )


def check_ids_fit_the_model(ids: list[int] | np.ndarray, vocab_size: int, where: str) -> None:
    """Refuse token ids the model has no row for.

    MLX does not raise on an out-of-range index here. It returns a number —
    a different number depending on the batch shape — and training proceeds
    with a loss that is quietly wrong. Cheap to check once, expensive to find
    later. `docs/LESSONS.md` L7 is what this cost.
    """
    largest = int(max(ids)) if len(ids) else -1
    if largest >= vocab_size:
        raise ValueError(
            f"{where}: token id {largest} but the model's vocabulary is {vocab_size} "
            f"(valid ids are 0..{vocab_size - 1}). The tokenizer and the model disagree — "
            "check that the model was built with tokenizer.vocab_size, special tokens included."
        )


def masked_loss(
    model: Transformer, inputs: mx.array, targets: mx.array, mask: mx.array
) -> mx.array:
    """Cross entropy over the response tokens only."""
    logits, _ = model(inputs)
    per_token = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).astype(mx.float32),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    # Divide by the number of scored tokens, not by the batch size — otherwise
    # a long prompt quietly shrinks the gradient of a short answer.
    return (per_token * mask).sum() / mx.maximum(mask.sum(), 1.0)


def evaluate_sft(
    model: Transformer, examples: list[tuple[list[int], list[int]]], batch_size: int
) -> float:
    """Mean masked loss over a fixed held-out set.

    The per-batch training loss here is unreadable as a curve, and that is not
    a bug in the training — it is arithmetic. Four examples with a handful of
    scored tokens each means the denominator is tiny and the variance between
    batches swamps the signal. Chapter 03 hit the same wall and solved it the
    same way: evaluate on fixed material, in a fixed order, so two numbers from
    two different steps are actually comparable.
    """
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        if not chunk:
            continue
        total += masked_loss(model, *collate(chunk)).item()
        count += 1
    model.train()
    return total / max(1, count)


def finetune(
    checkpoint: Path,
    tokenizer_path: Path,
    corpus: Path,
    out_dir: Path,
    examples: int = 400,
    steps: int = 200,
    batch_size: int = 4,
    max_len: int = 192,
    rank: int = 8,
    scale: float = 20.0,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> dict:
    mx.random.seed(seed)

    model, _, base_step, _ = load_checkpoint(checkpoint)
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Base model from step {base_step}, {model.num_parameters / 1e6:.1f}M parameters")

    apply_lora(model, rank=rank, scale=scale)
    trainable, total = count_trainable(model)
    print(
        f"LoRA rank {rank}: {trainable:,} trainable of {total:,} "
        f"({100 * trainable / total:.2f}%)\n"
    )

    data = make_dataset(corpus.read_text(encoding="utf-8"), examples, seed)
    encoded = [encode_example(tokenizer, d["prompt"], d["response"], max_len) for d in data]
    for ids, _ in encoded:
        check_ids_fit_the_model(ids, model.config.vocab_size, "fine-tuning data")
    scored = sum(sum(mask) for _, mask in encoded)
    context = sum(len(ids) for ids, _ in encoded) - scored
    print(f"{len(encoded)} examples: {scored:,} tokens scored, {context:,} tokens as context only")

    # A held-out tenth, never trained on, evaluated in a fixed order.
    split_at = max(1, int(len(encoded) * 0.9))
    held_out = encoded[split_at:]
    encoded = encoded[:split_at]
    print(f"{len(encoded)} for training, {len(held_out)} held out\n")

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, masked_loss)
    schedule = TrainConfig(steps=steps, warmup_steps=max(1, steps // 20), learning_rate=learning_rate)

    rng = random.Random(seed)
    history = []
    start = time.perf_counter()

    before = evaluate_sft(model, held_out, batch_size)
    print(f"held-out loss before fine-tuning: {before:.4f}")

    for step in range(steps):
        optimizer.learning_rate = learning_rate_at(step, schedule)
        batch = collate([encoded[rng.randrange(len(encoded))] for _ in range(batch_size)])
        loss, grads = loss_and_grad(model, *batch)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state, loss)

        if (step + 1) % max(1, steps // 8) == 0:
            held = evaluate_sft(model, held_out, batch_size)
            record = {"step": step + 1, "batch_loss": loss.item(), "held_out": held}
            history.append(record)
            print(
                f"step {step + 1:>5}/{steps}  "
                f"batch {record['batch_loss']:.4f}  held-out {held:.4f}"
            )

    elapsed = time.perf_counter() - start
    after = evaluate_sft(model, held_out, batch_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora.json").write_text(
        json.dumps(
            {
                "rank": rank,
                "scale": scale,
                "steps": steps,
                "held_out_before": before,
                "held_out_after": after,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{steps} steps in {elapsed:.1f}s")
    print(f"held-out loss {before:.4f} -> {after:.4f}")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "held_out_before": before,
        "held_out_after": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "runs" / "latest" / "checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data" / "tokenizer.json")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data" / "tinyshakespeare.txt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "sft")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"No checkpoint at {args.checkpoint}", file=sys.stderr)
        print("Run chapter 03 first: python 03_train/train.py --steps 300", file=sys.stderr)
        return 1

    result = finetune(
        args.checkpoint,
        args.tokenizer,
        args.corpus,
        args.out_dir,
        steps=args.steps,
        rank=args.rank,
        batch_size=args.batch_size,
    )

    model, tokenizer = result["model"], result["tokenizer"]
    prompt = "Continue this line: To be, or not to be, that is"
    ids = mx.array(tokenizer.encode(PROMPT_TEMPLATE.format(prompt=prompt)))
    end_id = tokenizer.special_tokens[END_OF_TEXT]

    print(f"\n{prompt}\n")
    produced = []
    for token in model.generate(ids, max_tokens=60, temperature=0.7, top_k=40):
        if token == end_id:
            print("  [stopped on <|endoftext|>]")
            break
        produced.append(token)
    else:
        print("  [ran to the token limit without stopping]")
    print(f"  Assistant:{tokenizer.decode(produced)}")

    print(
        "\nWhat to look for is whether it stops, not whether it is right. "
        "Stopping is the behaviour this chapter installs."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
