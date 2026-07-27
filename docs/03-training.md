# Chapter 03 — Training

**What it does.** `03_train/data.py` turns a text file into token arrays and
deterministic batches. `03_train/train.py` runs the loop: AdamW, warmup and
cosine decay, gradient clipping, periodic validation, checkpoints that resume.

**Why this is the chapter that matters.** Everything before it can be verified
by reading. This one produces a number that falls, and that number is the only
thing in the repository you cannot get any other way.

**Technologies.** MLX for the model and `mlx.optimizers` for AdamW and gradient
clipping. NumPy for the corpus on disk, memory-mapped so a large one never has
to fit in RAM. Safetensors for checkpoints. Logs go to JSONL, one record per
entry, so plotting later needs no parser.

**Decisions.**

- *Tokenize once, to disk.* Encoding is slow and the answer never changes.
  uint16 while the vocabulary fits in it — half the memory of int32, and the
  corpus is the largest thing on disk.
- *Validation is the tail of the document, not a shuffle.* Shuffle first and
  sentences from one paragraph land on both sides; validation loss then
  measures memorisation of context the model has already seen.
- *Batches are a function of seed and step.* Not a running generator — an
  actual function. `rng = default_rng((seed, step))`. That is what makes
  resuming at step 5,000 identical to never having stopped, instead of merely
  similar.
- *Evaluation walks fixed, non-overlapping windows.* Validation loss has to be
  comparable across runs, so it cannot be sampled randomly.
- *Warmup, then cosine to a floor.* The first gradients are the largest and the
  least informative; opening at full learning rate is the classic way to lose a
  run in the first fifty steps. The floor keeps late training moving instead of
  freezing at zero.
- *Gradient clipping at norm 1.0.* One bad batch produces one enormous
  gradient, and a single unclipped step can undo an hour.
- *Checkpoints carry the optimizer moments.* Resume without them and Adam
  restarts its estimates from zero, which shows up as a visible bump in the
  curve — a bump people usually blame on the data.
- *`--stop-after` stops the run without touching the schedule.* `--steps` still
  describes the whole run, so the cosine decays toward the same place whether
  you do it in one sitting or five. This is how you train on a laptop you also
  need for other things: an hour tonight, an hour tomorrow, one curve.

**The MLX trap, again.** `mx.eval(model.parameters(), optimizer.state, loss)`
sits at the bottom of the loop. Without it the graph keeps growing, the timer
measures nothing, and memory goes somewhere unpleasant. Same lesson as chapter
00, and it costs more here.

**The test that earns its place.** `test_the_loss_actually_goes_down` trains a
tiny model on a short repeating pattern for 60 steps and demands the loss drop
below 60% of where it started. A training loop that *runs* is not a training
loop that *trains*, and every other test in the file can pass while the model
learns nothing.

**Measured on this machine.**

```
preset "small", 24.9M parameters, batch 16 x 256 tokens, 300 steps
step  10: loss 7.0818
step 100: loss 4.9548
step 300: loss 4.1205
final validation loss 4.6077  (perplexity 100.3)
11,700-12,200 tokens/sec sustained
17 tests pass in 0.68s
```

300 steps is 1.2M tokens, which is about four passes over TinyShakespeare — far
too few to expect English. What it does show is the loop working: warmup
climbing, cosine decaying, gradient norms settling from 2.93 to around 0.65,
and the loss falling the whole way. Chapter 04 takes it further and measures
what each configuration actually costs.

**Resume, verified for real.** The same 300-step run, interrupted at step 150
and picked up again:

```
                straight through      interrupted at 150, resumed
step 160        4.6638                4.6629
step 300        4.1205                4.1273
final val       4.6077                4.6067
```

Close, and deliberately not called identical. Chasing that last decimal turned
up something worth its own page: two *uninterrupted* runs with the same seed do
not agree with each other either. `docs/REPRODUCIBILITY.md` has the experiment,
the numbers, and what a checkpoint can honestly promise once you know that.
