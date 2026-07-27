# Chapter 04 — Scale

**What it does.** `04_scale/memory.py` predicts what a configuration will cost
before you run it. `04_scale/sweep.py` runs real training steps across presets,
batch sizes and sequence lengths, and prints what it measured next to what was
predicted.

**Why both halves.** A predictor nobody checks is a story you tell yourself
about your own machine, and the first time it matters is the first time it is
wrong. A measurement with no model behind it tells you about the twelve configs
you tried and nothing about the thirteenth. Together they are useful: the model
generalises, the measurement keeps it honest.

**Technologies.** MLX, and `mx.get_peak_memory()` / `mx.reset_peak_memory()` for
the real numbers. The sweep imports `loss_fn` straight from chapter 03 — a
sweep that measured a different forward pass than the one you train with would
be measuring the wrong thing. Batches are random token ids on purpose: this
measures the machine, not the data, and no loss value from it is reported.

**Decisions.**

- *Exact where it can be exact.* Parameters, gradients and optimizer state are
  counted off the model definition and match `Transformer.num_parameters` to
  the byte, on every preset, tied and untied. There is a test.
- *Honest where it cannot.* Activation memory depends on what MLX's autodiff
  keeps alive and when it frees things. That is a property of the framework,
  so the activation term is **fit to measurement**, and the method, the
  coefficients and the leftover error are all written down in the file.
- *Skip rather than swap.* If the prediction says a config will not fit, the
  sweep says so in the table and moves on. Proving it by sending a 16 GB
  machine into swap costs ten minutes and teaches nothing.
- *An allocation failure fails one row, not the sweep.* Forty-eight
  combinations is too many to restart because the last one died.
- *Median, not mean, for throughput.* See below — this one was forced on me.

**How the activation model got its numbers.** First attempt: count the tensors
each layer saves, one coefficient per category, fit by least squares against 27
measured configurations. The fit was beautiful — within 5% everywhere — and the
coefficients came out **negative**. Across these presets the residual, KV and
feed-forward terms are nearly proportional to each other, so the fit could
trade them off freely and did. A model that fits with negative tensor counts is
a curve, not a model.

Collapsing the three layer terms into one multiplier left three terms that vary
independently, and the fit came back physically sensible:

```
layer multiplier (on 8*d_model + 3*d_kv + 3*d_ff)    2.18
logit tensors (batch * seq * vocab)                 10.08
attention scores (batch * heads * seq * seq)        12.79
```

That 2.18 is the backward pass allocating a cotangent for roughly every saved
forward tensor — which is what you would predict from reading the code, and it
is reassuring to see the fit agree.

**Two things the measurements said that reading did not.**

*The fused kernel buys speed, not memory.* Turning `fused_attention` off barely
moves the peak: 1.236 GiB against 1.257 GiB on micro at batch 8 x 256. MLX's
fused attention avoids materialising the score matrix in the forward pass, but
the backward pass needs it either way. I had assumed the fused path would be
visibly cheaper in memory. It is not.

*The first step at a new working set can be a hundred times slower.* The first
full sweep reported 40 tokens/sec for base at batch 8 x 512, sitting between
neighbours running at four thousand. Running that config three times in a row:

```
trial 1: 6.37 GiB,    85 tok/s
trial 2: 6.37 GiB, 4,233 tok/s
trial 3: 6.37 GiB, 4,120 tok/s
```

Byte-identical memory, throughput off by fifty times on the first pass. It is a
one-time cost paid when the allocator grows into a large new working set, and
one warm-up step does not absorb it. The sweep now takes two warm-up steps and
reports the **median** of the timed steps rather than the mean, because a mean
over five steps lets one outlier like that drag the answer a long way. Before
the fix, this chapter would have published 40 tok/s as a fact.

Worth putting next to `REPRODUCIBILITY.md`: peak memory here is perfectly
repeatable, down to the byte, run after run. Throughput is not. When you compare
two configurations, the memory numbers can be trusted directly and the
throughput numbers need a margin.

**Measured on this machine.** M1 Pro, 16 GiB unified, budget 12.0 GiB, 5 timed
steps per combination, 44 of 48 combinations run:

```
| preset | params | batch x seq | predicted | measured | ratio | tokens/sec |
|--------|--------|-------------|-----------|----------|-------|------------|
| nano   |   1.3M |    16 x 256 |  1.15 GiB | 0.99 GiB | 0.86x |     99,958 |
| nano   |   1.3M |    32 x 512 |  5.29 GiB | 2.76 GiB | 0.52x |     93,873 |
| micro  |   5.5M |    16 x 256 |  2.00 GiB | 2.19 GiB | 1.10x |     35,167 |
| micro  |   5.5M |    32 x 512 |  9.24 GiB | 5.51 GiB | 0.60x |     30,947 |
| small  |  24.9M |    16 x 256 |  3.76 GiB | 3.29 GiB | 0.87x |     12,335 |
| small  |  24.9M |    32 x 256 |  7.15 GiB | 4.93 GiB | 0.69x |     13,183 |
| small  |  24.9M |    32 x 512 | 15.43 GiB |        — |     — | skipped    |
| base   |  78.7M |     8 x 512 |  8.18 GiB | 6.37 GiB | 0.78x |      4,341 |
| base   |  78.7M |    16 x 256 |  7.62 GiB | 5.90 GiB | 0.77x |      4,732 |
| base   |  78.7M |    32 x 256 | 14.07 GiB |        — |     — | skipped    |

measured/predicted across 44 combinations: 0.52x to 1.24x
24 tests pass in 0.55s
```

The full table lands in `runs/sweep.json`. Read the ratio column as: the
estimate under-predicts by at most about 20%, and over-predicts by up to a
factor of two at large batch and long context, where the attention-score term
runs ahead of what MLX keeps alive. Over-predicting is the safe error for a
"will this fit" tool — it declines a config that would have run, instead of
sending your machine into swap — so it was left biased that way.

The practical summary for a 16 GB machine: **base at 78.7M parameters trains
comfortably at batch 8 x 512 or batch 16 x 256, and stops there.** Chapter 03's
default of small at batch 16 x 256 uses 3.29 GiB and leaves the laptop usable
for everything else, which is why it is the default.
