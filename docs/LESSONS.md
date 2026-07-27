# What this cost me to learn

Every entry below is something I got wrong first, on this machine, while
building this repository. They are here because the fix is in the code and the
*reason* is not — and the reason is the part worth having.

Some of these nearly shipped. I have said so where that is true. A page like
this is only useful if it includes the ones that make me look bad.

Newest first.

---

## L8 — A loss curve you cannot read is not a loss curve

**What I expected.** Fine-tuning starts, the printed loss falls, chapter done.

**What happened.** Two hundred steps: 4.73 at step 10, 5.09 at step 200. It went
*up*. And yet the model had clearly learned the thing I was fine-tuning it to
do — it now stops on `<|endoftext|>`, which it never did before.

**Why.** Arithmetic, not training. A batch is four examples with a handful of
scored tokens each, so the denominator is tiny and the variance between batches
swamps whatever the model learned in one step. The number was never a curve. It
was four random examples wearing a curve's clothes.

**What changed.** A held-out tenth, evaluated in a fixed order, exactly the way
chapter 03 evaluates on fixed windows. Same run, now legible:

```
held-out loss before fine-tuning   6.5296
step  50                           5.1474
step 125                           5.0779
step 200                           5.0644
batch loss over the same run       4.59 – 5.36, no trend
```

**What it cost.** Almost a wrong conclusion in the other direction — I was one
step away from tuning the learning rate to fix a problem that was not there.
The measurement was broken, not the model. Chapter 03 had already taught me
this and I still had to learn it twice.

## L7 — The framework will not tell you the ids are out of range

**What I expected.** A padded batch and an unpadded one score the same example
identically. Attention is causal, the padding sits at the end, the loss mask is
zero there. It cannot matter.

**What happened.** It mattered. 6.41 unpadded, 6.57 padded, on the same example.

**How I found it.** This is the part worth keeping, more than the answer:

1. *Hypothesis: the model is not causal — logits depend on sequence length.*
   Testable directly. Ran a prefix against the full sequence, both attention
   paths: `max |logits(prefix) − logits(full)[:n]| = 0.000e+00`. Exactly zero.
   **Dead.** And worth noticing that chapter 02's causality test would never
   have caught this — it changes the *last* token of a fixed-length sequence,
   it does not *extend* the sequence. Two different properties, and I had only
   been testing one.

2. *Hypothesis: collate misaligns something — inputs, targets, or the mask.*
   Printed all three side by side. Inputs equal on the overlap, targets equal
   on the overlap, mask equal on the overlap, mask sums both 4.0, scored
   indices both `[12, 13, 14, 15]`. **Dead.**

3. Now a contradiction: identical inputs, identical targets, identical mask,
   identical logits — and different per-token losses. One of those four
   statements had to be false, so I stopped reasoning and printed the
   per-token losses. Not equal on the overlap. So the inputs to cross entropy
   were the same and its output was not.

4. Which leaves cross entropy itself, and the one thing about it I had not
   checked: the target *values*. `max target id: 300`, `vocab_size: 300`.
   Valid ids are 0–299. **The tokenizer had 300 merged tokens and I registered
   `<|endoftext|>` as id 300 on top of them, then built the model with 300
   rows.** Off by one.

**Why it was invisible.** MLX does not raise on an out-of-range index here. It
returns a number — and the number it returns depends on the batch shape, which
is why the bug only appeared when padding changed the shape. Silent, shape
dependent, and it would have shown up as "fine-tuning does not work very well".

**What changed.** A guard that refuses ids the embedding has no row for, at
dataset-build time in chapter 05 and once per run in chapter 03, with a message
that names the actual numbers. And then the guard immediately caught a *second*
instance of the same bug, this one in real code rather than in a test fixture:
chapter 03 was building the model from `PRESETS[preset].vocab_size`, a constant
4096, while the tokenizer's vocabulary was 4097 with the special token included.
Chapter 03 survived on luck — its corpus never contains `<|endoftext|>`, so no
id ever exceeded 4095. The moment chapter 05 used the token, it broke.

The real fix is upstream of both: `prepare()` now writes the vocabulary size
next to the tokens, and the training loop reads it. **The tokenizer decides the
vocabulary. A preset default has no way of knowing what you trained on.**

**What it cost.** An hour, and it was the best hour in the chapter. One test
fixture's off-by-one led to a guard, and the guard found a real defect one
chapter over that nothing else was going to find.

## L6 — I almost published a number that was fifty times wrong

**What I expected.** Chapter 04's sweep measures throughput across
configurations. Run five steps, average them, print the table.

**What happened.** The table said `base` at batch 8 × 512 runs at **40
tokens/sec**, sitting between two neighbours running at four thousand. It looked
like a real result — large model, long context, of course it is slower.

I ran that one configuration three times:

```
trial 1: 6.37 GiB,    85 tok/s
trial 2: 6.37 GiB, 4,233 tok/s
trial 3: 6.37 GiB, 4,120 tok/s
```

Byte-identical memory every time. Throughput off by fifty on the first pass.

**Why.** The allocator pays a one-time cost growing into a large working set.
One warm-up step does not absorb it, and a mean over five steps lets that single
outlier drag the whole answer.

**What changed.** Two warm-up steps, and the **median** of the timed steps
rather than the mean. Everywhere in this repository that reports a rate.

**What it cost.** Nearly the credibility of the entire chapter. A published
benchmark table with a fifty-times error in it is worse than no table, and I
would have had no reason to look again. The thing that saved it was that the
number was *implausible next to its neighbours* — which is an argument for
printing neighbours, and for reading your own output instead of pasting it.

## L5 — Two identical runs are not identical

**What I expected.** Same seed, same config, same data, same numbers. That is
what setting a seed is for.

**What happened.**

```
run 1:  step 10  7.0819    step 30  6.0782    step 60  5.4491
run 2:  step 10  7.0819    step 30  6.0782    step 60  5.4507
```

Identical, identical, apart.

**Why.** Floating-point addition is not associative, and a GPU schedules the
partial sums of a reduction in whatever order suits it. Each step, that
difference goes through the optimizer and comes back multiplied.

**What changed.** `docs/REPRODUCIBILITY.md` exists, the resume test asserts
agreement to a tolerance instead of to the bit, and the tolerance now has a
receipt behind it rather than being a number I picked to make a test pass.

**What it cost.** Half an hour of hunting a checkpoint bug that was not there.
Worth every minute — I went looking for a defect and came back with the noise
floor of the whole project. Every loss comparison in this repository is now read
against it.

Pair this with L6: **peak memory is repeatable to the byte, throughput is not.**
Memory comparisons can be read flat. Speed and loss comparisons need a margin.
Two facts about the same machine that point in opposite directions, and knowing
which is which decides whether a result means anything.

## L4 — "Fused" does not mean "cheaper"

**What I expected.** MLX's fused attention kernel avoids materialising the score
matrix, so it should use noticeably less memory. I would have written that down
as fact.

**What happened.** Measured: 1.236 GiB fused, 1.257 GiB written out. Under two
percent.

**Why.** The forward pass skips the score matrix. The backward pass wants it
regardless.

**What changed.** The comment in `02_model/model.py` now carries the measurement
instead of the intuition, and says what the fused path is actually for: speed,
not fitting a model that would not otherwise fit.

**What it cost.** Nothing, this time, because chapter 04 measured it before
chapter 02 could claim it. That is the only reason — the intuition was sitting
right there in a comment, sounding correct.

## L3 — The bug that does not crash

**What I expected.** A KV-cache is bookkeeping. Store the keys, store the
values, concatenate.

**What happened.** During generation there is one query and a growing pile of
keys, so the query sits at the *end* of the key range. Get that alignment wrong
and nothing raises. Shapes stay valid. Training metrics stay clean. Generation
quality just quietly degrades.

**What changed.** `test_cache_matches_a_full_forward_pass` demands that feeding
tokens one at a time produces exactly what one pass over the whole sequence
produces — on both the fused and the written-out attention paths.

**What it cost.** Nothing yet, because the test came first. This is the one
entry on the page written from fear rather than from damage: I had read enough
about this class of bug to build the trap before walking into it.

## L2 — The obvious algorithm does not finish

**What I expected.** BPE is a simple loop. Count pairs, merge the most frequent
one, repeat. Four thousand merges on a megabyte of text.

**What happened.** Ten minutes in, still running. Several billion pair visits,
because the loop re-counts the entire corpus before every single merge.

**Why.** The textbook description of an algorithm is a description, not an
implementation plan.

**What changed.** Deduplicate chunks — ` the` appears thousands of times and
merges identically every time — and keep a running pair counter with an index
from each pair to the chunks containing it. **Ten minutes to 6.2 seconds.**

And because a speedup that quietly changes the output is worse than no speedup,
`test_fast_training_matches_the_slow_obvious_way` keeps the textbook version
alive and asserts the merges are byte-identical.

**What it cost.** An afternoon, and it gave the repository its shape. The
readable version, the fast version, and a test tying them together — that
pattern came from here and now runs through every chapter.

## L1 — Lazy means your timer measures nothing

**What I expected.** Write the operation, time it, read the number.

**What happened.** MLX builds a graph and runs nothing until you ask for a
value. Timing code without `mx.eval()` measures graph construction and reports
something wonderful and completely false.

**What changed.** `mx.eval()` in chapter 00's benchmark, at the bottom of
chapter 03's training loop, and inside the `bench/` harness. Every timing site
in this repository has a comment saying why it is there, because it is the kind
of line a future reader deletes as redundant.

**What it cost.** Ten confused minutes at the very beginning — and it is the
single most important thing to know about this framework, which is why it is
entry one and why it comes back in L6.

---

## The pattern underneath all of them

Almost every one of these was caught by *measuring something I already
believed*. Not by careful reasoning, not by reading the documentation more
attentively — by running it and looking at the number.

That is what all the equivalence tests in this repository are: a habit of not
letting a belief through without a measurement behind it. It started as a way to
make a fast path trustworthy and turned into how the whole thing gets built.

L7 has the method written out, and it is the one to copy. When four things you
believe cannot all be true at once, stop reasoning about which one is wrong and
print all four. The contradiction is doing the work for you — it has already
narrowed the answer to a handful of candidates, and measuring them is faster
than thinking about them. Every time I have tried to reason my way out of one
of these instead, I have picked the wrong candidate first.
