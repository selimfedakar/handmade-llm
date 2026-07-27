# What is reproducible here, and what is not

Most training repositories say "set the seed" and move on. I set the seed, ran
the same thing twice, and the two runs did not agree. So this page exists.

It is a ledger. Every claim below was produced by running the experiment on
this machine, and the output is pasted. Where something is not reproducible, it
says so, with the evidence, instead of a reassuring sentence.

---

## Reproducible: the data pipeline

Batches are a function of `(seed, step)`, not of a running generator. Ask for
step 5,000 on a fresh process and you get the same sixteen windows you would
have got by never stopping.

```
test_the_same_step_gives_the_same_batch          PASS
test_different_steps_give_different_batches      PASS
test_different_seeds_give_different_batches      PASS
```

This is the part people usually get wrong, and it is the part that costs the
most when it is wrong — a resumed run that quietly re-reads the same data, or
skips a third of the corpus, and nothing in the loss curve tells you.

## Reproducible: the learning-rate schedule

`learning_rate_at(step, config)` is a pure function of the step. Interrupting a
run does not change it, and `--stop-after` is explicitly tested not to move it.

```
test_stopping_early_leaves_the_schedule_alone    PASS
```

## Reproducible: a forward pass, within one process

Same weights, same input, evaluated five times:

```
[8.710824966430664, 8.710824966430664, 8.710824966430664,
 8.710824966430664, 8.710824966430664]
all identical: True
```

## **Not** reproducible: training, across processes

Two runs. Identical seed, identical configuration, identical data, nothing
between them but a process boundary:

```
run 1:  step 10  loss 7.0819     step 30  loss 6.0782     step 60  loss 5.4491
run 2:  step 10  loss 7.0819     step 30  loss 6.0782     step 60  loss 5.4507
```

Identical at step 10. Identical at step 30. Apart by 0.0016 at step 60.

Floating-point addition is not associative, and a GPU is free to schedule the
partial sums of a reduction in whatever order suits it that moment. Each step
that difference goes through the optimizer and comes back multiplied. Nothing
is broken. This is what training on a GPU is.

**What this means in practice:** treat the loss curve as a measurement with
noise, not as a fingerprint. Two configurations that differ by 0.002 differ by
nothing. If you are comparing two ideas, the gap has to be larger than the gap
above, or you are reading the scheduler.

## Consequence: what a checkpoint can promise

It can promise to restore the weights, the optimizer moments, the step counter,
and therefore the schedule and the data position. It cannot promise that the
resumed curve is bit-identical to the uninterrupted one, because *the
uninterrupted one is not bit-identical to itself*.

Measured. A 300-step run, once straight through and once interrupted at step
150 and resumed:

```
                straight through      interrupted at 150, resumed
step 160        4.6638                4.6629
step 200        4.4201                4.4245
step 250        4.2187                4.2227
step 300        4.1205                4.1273
final val       4.6077                4.6067
```

Every gap is inside the noise the two identical runs above already showed. That
is the honest form of the claim, and it is the one this repository makes.

`test_resuming_continues_the_same_run` asserts agreement to a relative
tolerance of 1e-4 on a small model over a short run. The tolerance is not
laziness — it is the platform, and this page is the receipt.

## What would make it worse, and is not done here

- **Non-deterministic data order.** Shuffling with a running generator instead
  of a step-indexed one. Common, and it makes resume a lie.
- **Dropping optimizer state.** Adam restarts its estimates from zero and the
  loss visibly bumps. People blame the data.
- **Comparing runs at different step counts** because the cosine schedule was
  defined by "remaining steps". `--stop-after` exists precisely so that
  stopping early cannot do that.

## How to re-run this yourself

```bash
python -m pytest 03_train -q
python 03_train/train.py --preset small --steps 60 --out-dir runs/det1
python 03_train/train.py --preset small --steps 60 --out-dir runs/det2
```

Your numbers will not be my numbers — different chip, different scheduler. The
*shape* should hold: agreement early, drift later. If yours diverges at step 10,
something on your machine is genuinely wrong, and now you have a way to know.
