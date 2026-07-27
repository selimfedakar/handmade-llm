# Chapter 05 — Fine-tuning

**What it does.** `05_finetune/lora.py` is LoRA written out: a frozen `Linear`
with a trainable low-rank update beside it, plus merging back down to a plain
layer. `05_finetune/sft.py` is supervised fine-tuning: chat formatting, loss
masking, a held-out evaluation, and a training loop that moves 0.46% of the
parameters.

**Why it comes after chapter 03 and not instead of it.** Chapter 03 taught the
model to predict the next token of everything, which is how it learned English
and also why it will not stop talking — nothing in that objective ever rewarded
finishing. Fine-tuning does not make the model bigger or smarter. It changes
what gets scored.

**Technologies.** MLX's `freeze()` / `trainable_parameters()` for the frozen
base, `nn.losses.cross_entropy` with `reduction="none"` so the mask can be
applied per token, AdamW on the adapters only.

**Decisions.**

- *LoRA, not full fine-tuning.* At 24.9M parameters this laptop could do either.
  At 7B it could not, and the technique is worth understanding on a model small
  enough to check every claim about it. Rank 8 on the query and value
  projections: **114,688 trainable of 25,027,072 — 0.46%.**
- *B starts at zero.* So the adapter is a no-op at initialisation and
  fine-tuning starts from the model you already have.
  `test_a_fresh_adapter_changes_nothing` asserts it, and
  `test_merging_preserves_the_output` asserts the folded weights are the same
  function. The repository's usual move, applied twice.
- *Plain-text role markers, not new special tokens.* `User:` / `Assistant:`
  encode with the tokenizer you already trained. Real special tokens mean
  widening the embedding table of a trained model, which is a different
  chapter's worth of care — and, as it turned out, exactly the seam where this
  chapter's worst bug lived.
- *Right padding, no attention mask.* Attention is causal, so a padded position
  can only influence positions after it, and every one of those is zero in the
  loss mask. Left padding would need real care.
- *The dataset is built from the corpus.* Take a line, cut it in half, ask for
  the second half. No annotation, no download, and it is genuinely learnable by
  a model this size — which matters, because a chapter you cannot watch work
  teaches nothing.

**Masking is the whole chapter.** The prompt is context, not homework. Every
token of it is zero in the loss. Score the prompt too and the model learns to
generate plausible *questions* — a failure that looks like success until you try
to use it. `test_the_prompt_contributes_nothing` is the guard.

**Two things the run taught that reading did not.**

*The training loss was unreadable, and it was not the training's fault.* Four
examples per batch with a handful of scored tokens each means the variance
between batches swamps the signal. Two hundred steps went 4.73 → 5.09 — up —
while the model was visibly learning. The fix is chapter 03's fix: a held-out
tenth, evaluated in a fixed order. `docs/LESSONS.md` L8.

*An out-of-range token id does not raise.* It returns a number that depends on
the batch shape. A test fixture was off by one, the guard that came out of it
immediately caught the same bug in real code — chapter 03 was building models
with a preset's constant vocabulary instead of the tokenizer's. L7 has the whole
elimination trail, and it is the entry to read if you only read one.

**Measured on this machine.**

```
Base model from step 300, 24.9M parameters
LoRA rank 8: 114,688 trainable of 25,027,072 (0.46%)
400 examples: 3,463 tokens scored, 9,224 tokens as context only
360 for training, 40 held out

held-out loss before fine-tuning   6.5296
step  50                           5.1474
step 125                           5.0779
step 200                           5.0644
batch loss over the same run       4.59 - 5.36, no trend

200 steps in 4.5s
19 tests pass in 0.52s
```

**The result, honestly.** Same prompt, 60 tokens, temperature 0.7, before and
after:

```
base model      60 tokens produced, never stopped
fine-tuned      stopped on <|endoftext|>
```

The text it produces is nonsense — *"the and and; my."* — and it was always
going to be. A 25M model trained on a megabyte of Shakespeare and fine-tuned on
360 examples has nothing to say. What it now has is a *shape*: answer, then
stop. That behaviour was installed by 0.46% of the parameters in four and a half
seconds, and the control run proves it was not there before.

That is the honest claim for this chapter, and it is worth more than a
cherry-picked sample would be.
