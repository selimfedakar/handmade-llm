# Chapter 06 — Did it actually learn anything?

**What it does.** `06_eval/metrics.py` measures perplexity, bits per byte and
top-k accuracy. `06_eval/probes.py` asks three specific questions the metrics
cannot answer. `06_eval/evaluate.py` runs all of it against a checkpoint, and
`--untrained` runs the same thing against a freshly initialised model, which is
the only reason any of the numbers are readable.

**Why the chapter exists.** Chapter 03 ends with a falling loss. That is not the
same as the model having learned anything you care about, and everything in this
chapter lives in the gap between those two sentences.

**Technologies.** MLX, and nothing else. The probes are numpy and arithmetic.

**Decisions.**

- *Bits per byte is the headline number.* Perplexity is not comparable across
  tokenizers — a bigger vocabulary means fewer tokens for the same text, each
  carrying more information, so perplexity rises without the model getting
  worse. Dividing by the bytes of the original text instead of by the token
  count removes the tokenizer from the denominator. Almost everyone quotes
  perplexity as if it were portable. There is a test asserting it is not.
- *Sum, then divide once.* Averages of per-batch averages over batches of
  different sizes are quietly wrong. Chapter 05 already paid for that lesson.
- *Accuracy alongside perplexity, always.* Perplexity is dominated by the
  positions the model finds hardest; a model can carry a bad perplexity while
  getting most tokens right.
- *Every probe is a controlled comparison with a predictable null.* Same model,
  same amount of text, one thing changed, and a value you can state in advance
  for a model that learned nothing. Otherwise it is a demonstration, not a
  measurement.
- *One rule for accuracy at every k.* A token counts as top-k when fewer than k
  tokens score strictly above it — top-1 included, so "top-1" is exactly
  "top-k at k=1". The obvious alternative, `argmax == target`, is quietly a
  different question: argmax breaks ties by lowest index, so of two tokens
  sharing the top score one would be wrong at k=1 and both right at k=2. Exact
  float ties over four thousand tokens do not happen, and the reported numbers
  are identical either way — but a metric that answers one question two ways is
  the sort of thing that surfaces at the worst possible moment.

**Measured on this machine.** Checkpoint at step 300, 24.9M parameters, against
a freshly initialised control:

```
                         trained (step 300)    untrained control
loss                     4.6032                8.8388
perplexity               99.8                  6,896.9
bits per byte            2.134                 4.098
top-1 accuracy           24.0%                  0.0%
top-5 accuracy           38.8%                  0.1%
memorisation gap        +0.3207                +0.0492
```

A uniform guess over this vocabulary would score 8.32. The untrained model
scores 8.8388 — slightly *worse* than guessing, because a random initialisation
is confidently wrong rather than uniformly unsure.

**The probes, and what they actually support.**

Both probes were run across six seeds, because a single number from a probe is
not a result — `docs/REPRODUCIBILITY.md` already established that comparisons
in this repository need a margin.

```
                    untrained                        trained
context gain        mean -0.0135  [-0.1454, +0.1281]  mean +0.0312  [+0.0179, +0.0477]
induction gain      mean +0.0055  [-0.0271, +0.0355]  mean -0.0222  [-0.0711, +0.0357]
```

*Context: yes, but read it carefully.* The trained model is positive on **six
seeds out of six**, in a tight band, while the control swings across zero by
more than the whole effect. The defensible claim is the consistency of the
sign, not the size of the number — +0.031 is smaller than the control's own
spread, so anyone quoting the magnitude alone would be quoting noise. Shuffling
the first half of a window makes the second half measurably harder to predict.
The model is using its context. Barely, and it is early.

*Induction: no.* The trained model is at −0.022 with a range straddling zero —
seeing a random sequence a second time does not help it. There is no induction
circuit at 300 steps.

That is a negative result and it stays in. A repository that only reports the
probes that came out well is not measuring anything; it is decorating. The
honest summary of this checkpoint is that after 1.2M tokens it has learned local
statistics — which characters follow which, roughly what Shakespeare looks like
— and has not yet learned to look back and copy. The way to find out when it
does is to train longer and run this chapter again, which is exactly what the
chapter is for.

**What went wrong first.** The context probe reported a gain of exactly
`+0.0000`. Not small — exact. Both probes were slicing the sequence before
feeding the model, so the model never saw the prefix it was supposed to be
using, or the first copy it was supposed to be copying from. The fix is one
line in each: run the model on the whole window and restrict only the grading.
`docs/LESSONS.md` L9 has the trail, and the tell was the suspicious precision
of that zero.

```
25 tests pass in 0.62s
```

The one that carries the chapter's thesis is
`test_bits_per_byte_survives_a_change_of_tokenizer`: same text, same total
negative log likelihood, 250 tokens against 500. Perplexity moves by more than
1.5×. Bits per byte does not move at all, because the token count never enters
the formula. That is the claim, and that is the receipt.
