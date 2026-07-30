# What this cost me to learn

Every entry below is something I got wrong first, on this machine, while
building this repository. They are here because the fix is in the code and the
*reason* is not — and the reason is the part worth having.

Some of these nearly shipped. I have said so where that is true. A page like
this is only useful if it includes the ones that make me look bad.

Newest first.

---

## L16 — The build succeeded, and the compiler was missing

**What I expected.** `swift build` in chapter 08 either works or tells me what is
wrong. That is what a build system is for.

**What happened.** It worked. Every Swift file compiled, the package linked, the
test bundle started, and then the first MLX operation died inside C++:

```
MLX error: Failed to load the default metallib.
library not found library not found library not found library not found
  at .../mlx-c/mlx/c/stream.cpp:106
```

**Why.** MLX's GPU kernels are 49 `.metal` files inside mlx-swift, and since
Xcode 26 the Metal compiler is not part of Xcode. It is a separate download:
`xcodebuild -downloadComponent MetalToolchain`, 688 MB. Without it the shaders
are silently not built, and a missing *compiler* arrives as a missing *file*, at
run time, four layers below anything I wrote.

**The wrong turn, and it is the interesting part.** My first move was to fall
back to MLX's CPU backend for the tests, which sounded obviously right — the
correctness claims here are about a port, not about speed. So I set the default
device to CPU and ran it, and it failed in the same place, with a line number
that says `mlx_default_cpu_stream_new`. Which reads exactly like "the CPU path is
broken too", and is really the scheduler building its per-device streams the
first time it is touched, GPU included. **There is no CPU fallback in MLX**, and
the error message points at the door I was trying to use rather than at the one
that is locked.

Then, with the toolchain installed, `swift build` still produced no metallib. The
answer was in mlx-swift's own README, in a note it is easy to read past:

> SwiftPM (command line) cannot build the Metal shaders so the ultimate build has
> to be done via Xcode.

**What changed.** The chapter's test command is `xcodebuild test`, not
`swift test`, and it is written that way everywhere. `TestDevice.swift` looks for
the compiled library before MLX is touched and fails with the exact command to
run. It does not *skip* — I wrote it as a skip first, watched twelve tests
report "skipped" in a green run, and changed my mind inside a minute. A chapter
whose central test quietly reports "skipped" on a machine that cannot run it is
the kind of green checkmark this repository exists to argue against.

**What it cost.** An evening, split roughly evenly between chasing the CPU
fallback that does not exist and not reading a README note that was sitting
there. Two transferable pieces. First: **a fallback that fails in the same place
as the thing it was meant to avoid is telling you the failure is upstream of
both.** I had that evidence immediately and spent an hour treating it as a second
bug. Second: when a tool's own documentation has a one-line caveat about the
exact thing you are doing, you will find it after the experiment, not before —
which is an argument for grepping vendor READMEs for your verb before you start.

## L15 — The simulator is not a small phone

**What I expected.** Build chapter 08's app for the iOS simulator, watch the
model generate, and save the physical device for the numbers at the end.

**What happened.** The app built, installed, launched, and died before the first
frame:

```
.../c++/v1/string:1062: libc++ Hardening assertion __s != nullptr failed:
basic_string(const char*) detected nullptr
```

**Why.** One line, in `mlx/backend/metal/device.cpp`:

```cpp
arch_ = std::string(device_->architecture()->name()->utf8String());
```

`MTLDevice.architecture` is not implemented by the simulator's Metal. It returns
nil, `name()` on nil returns nullptr, and `std::string(nullptr)` aborts. This
happens while MLX is constructing its device, before any code in this repository
runs, so there is nothing an app can do about it.

**What changed.** Chapter 08 is planned around a physical device: correctness on
macOS, where the same Metal backend works, and the phone for the measurements.
`docs/08-ship_ios.md` says so at the top rather than leaving the next person to
find it the same way.

**What it cost.** Twenty minutes, and it is here for the shape rather than the
size. I checked this **first**, before writing the app, because "does the
simulator work" was the one question whose answer changes the plan for the whole
chapter rather than one function in it. It turned out to be the answer that
costs a day if you find it last — the version of this session that wrote the app,
the tests and the docs first and then discovered the simulator was never an
option would have had to re-plan everything about how the chapter gets verified.

Cheap checks of load-bearing assumptions go at the start. That is not caution, it
is ordering.

## L14 — Small test data understates your noise floor too

**What I expected.** The Swift port of the model should reproduce Python's
logits. The tiny model committed to `Golden/` did, exactly — worst difference
`0.000e+00` across 1,280 logits, in both float32 and 4-bit form. So the real
24.9M checkpoint should as well.

**What happened.** It came out `1.431e-06` apart across 4,097 logits. Nothing
downstream noticed: the argmax matched, sixteen greedy tokens matched. Exactly
the size of difference that is easiest to shrug at, and shrugging at it is how a
real port bug hides.

**How I found what it was.** Not by reasoning, because I tried that first and
killed three hypotheses that were all wrong:

1. *The rope frequencies lose precision at head_dim 64.* Python computes
   `-math.log(base)` as a double and Swift as a `Float`. Testable directly:
   computed the frequency table both ways, at `half = 8` and `half = 32`.
   `max |Δ| = 0.000e+00`, both. **Dead.**
2. *MLX is not reproducible across processes — `docs/REPRODUCIBILITY.md` says so
   for training.* Ran the Python reference twice in two processes: `0 of 4097`
   values differ. A forward pass is deterministic across processes even though
   training is not. **Dead**, and worth knowing on its own.
3. *The embedding decode.* This one was **alive**, and it was a real finding, but
   it was not the whole answer — see below.

Then I stopped reasoning and printed, which is what L7 already told me to do.
`export_golden.py` now exports two points from inside the forward pass, and the
answer arrives as three numbers instead of one:

```
python vs swift, three points in one forward pass
  after the embedding    worst |Δ| = 0.000e+00
  after eight layers     worst |Δ| = 9.537e-07
  after the output head  worst |Δ| = 1.431e-06
```

So the divergence enters in the layers. And then the number that makes all of
this readable, which took one more experiment — swapping MLX's fused attention
for the written-out softmax **inside Python alone**, changing nothing about what
the model computes:

```
python: fused vs written-out attention, real model    max |Δ| = 1.907e-06
                                        tiny model    max |Δ| = 0.000e+00
```

**Why.** Two correct implementations of this model, in one language, differ by
*more* than the two languages do. 1.4e-06 across eight layers of 512 float32
numbers is the floor, not a defect. And the tiny model reproduces Python exactly
only because two layers of 64 numbers never reach that floor.

**What changed.** Two things, and the second matters more than the first.

The embedding decode was genuinely wrong to leave as it was: Swift called MLX's
`dequantized` kernel while `07_quantize/quantize.py` does that arithmetic itself,
and the two agree to 3e-08 rather than exactly. `Unpack.swift` now performs the
same operations in the same order, so the embedding is bit-identical, and the
repository gets its signature move in Swift as well — a readable decoder next to
the kernel, with a test between them.

And the tiny-model logit assertion is now at a tolerance of **exactly zero**,
deliberately brittle. Under the 1e-4 I first wrote, the 2e-06 caused by the
embedding would have passed silently and I would never have looked at it.

**What it cost.** An afternoon, and the finding is the one sentence I did not
have before: **a fixture small enough to commit is small enough to hide your
noise floor.** L12 said the same thing about rounding ties — small data does not
manufacture the conditions where two correct implementations are allowed to
disagree — and this is the same lesson approached from the opposite side. There,
small data made a real difference invisible. Here, it made a normal one look
impossible, so the first appearance of a normal difference read as a bug.

Both times the fix was the same: run it at the size that ships.

## L13 — A median protects you from a noisy sample, not from a moving baseline

**What I expected.** Chapter 04 already taught me how to time things here: two
warm-up steps, then a median, never a mean. Apply that to both models and the
comparison is sound.

**What happened.** I wrote "there is no measurable speed difference between the
float32 model and the 4-bit one" into the chapter, with the overlapping ranges
underneath it, and it was **the opposite of the truth**.

Seven runs. Same two models, same machine, same script, nothing changed between
them:

```
float32    379.0  379.5  351.4  249.8  244.6  309.1  255.8   tokens/sec
quantized  380.4  376.0  367.7  338.3  358.8  278.3  260.0
```

The ordering flipped four times. The spread inside one model — 244 to 379 — was
larger than any gap between the two. Meanwhile, in those same seven runs:

```
peak memory   97 MiB   /   18 MiB      seven times out of seven, to the byte
```

L5 arriving exactly on schedule, one chapter later. Memory repeats; throughput
does not.

**Why.** Two warm-ups and a median control the noise *inside* one measurement.
They do nothing whatever about drift *between* two of them. My script measured
the float32 model, then did other work, then measured the quantized one — so
whichever model went second was measured on a machine in a different thermal and
allocator state than the first. The order was a confound and I had not noticed
it was there, because the rule I was following ("warm up, take the median")
sounded like it covered timing in general.

**What changed.** `interleaved_speed`: both models resident at once, timed in
one loop, one round each, alternating, medians taken from the interleaved
samples. The drift then lands on both equally. Three runs of it:

```
341.6 / 364.1      336.4 / 361.0      345.5 / 367.1
```

And because a new instrument makes a new claim, I tried to break it — reversed
which model goes first inside each round, twice each way:

```
float32 first     quantized/float32 = 1.102,  1.021
quantized first   quantized/float32 = 1.046,  1.051
```

Seven paired measurements, quantized ahead in every one. So the real finding is
that **4-bit generation is faster, by something between 2% and 10%**, and the
honest form of that claim is the consistency of the sign rather than the
magnitude — which is the rule chapter 06 already wrote down for probes, arriving
here from a completely different direction.

**What it cost.** The chapter's speed section was written, reasoned about, and
wrong, and the reasoning underneath it was *good* — I had overlapping ranges and
I read them correctly. The measurement was broken, not the interpretation. That
is the third time in this repository (L6, L8, and now this) that a careful
conclusion was drawn from an instrument nobody had checked, and it is starting
to look less like bad luck than like the default failure mode.

If you are comparing two things on a machine that drifts, measuring them one
after the other is not measuring them under the same conditions. Put them in the
same loop, and then flip the order to prove the loop did not decide the answer.

## L12 — Agreement on a matrix you can read is not agreement

**What I expected.** Chapter 07 writes group-wise 4-bit quantization from
scratch and claims the codes it produces are byte-identical to MLX's fused
kernel. I checked that across every bit width from 2 to 8 and every group size,
on matrices of a few thousand elements. All identical, every time. Claim earned.

**What happened.** I ran it on a 512 x 1344 matrix — 688,128 weights — and two
codes disagreed. Then on the real checkpoint, and forty-six codes in a single
layer disagreed, and there it was not even the code that differed first. It was
the *scale*.

**How I found it.** Print the disagreements with everything around them, the way
L7 says to. First the two random-matrix cases:

```
w[344,1037]=0.458563596   scale=0.0173042845   bias=-2.42259979
(w - bias) / scale = 166.500001130227      mine 166,  mlx 167
w[465,289] =0.428149939   scale=0.0190288946   bias=-2.70210314
(w - bias) / scale = 164.499995497286      mine 164,  mlx 165
```

Exactly a half, both times. Not a bug in either implementation — a disagreement
about which way a tie goes. `mx.round` breaks a tie toward the even integer.
MLX's quantize kernel breaks it away from zero. Two defensible conventions, and
the reconstruction error is identical to sixteen digits whichever you pick.

The second cause was the one worth the afternoon. In
`layers.7.attn.v_proj` a group ran from `-0.05107788` to `0.0390595607`, and
the step that spreads fifteen levels across it divides that range so evenly
that the zero-alignment step landed on `8.5` exactly:

```
mine   -bias/step = 8.5 -> 8  ->  scale 0.006384735
mlx    -bias/step = 8.5 -> 9  ->  scale 0.00567532005
```

One tie, in one scalar, and the whole group's scale changes — so forty-six of
its sixty-four weights come out one code different. The element-level tie is a
rounding curiosity. This one propagates.

**Why ties are common here, when they are rare everywhere else.** Because
quantization *manufactures* them. A group whose range divides evenly by the
number of levels puts weights precisely half way between two codes, and that is
not an unlucky float, it is what the arithmetic is built to do.

**What changed.** One helper, `_round_half_away`, used at both rounding sites,
with the reason written above it — because `mx.floor(x + 0.5)` is exactly the
kind of line a reader tidies back into `mx.round(x)`. And the evidence behind
the equivalence claim is no longer the test file. It is a sweep over all
24,904,192 weights of the real checkpoint, at every legal bit width and group
size: **0 differing words**.

**What it cost.** An afternoon, and nothing shipped — but the claim was one
commit away from shipping on evidence that could not possibly have caught it.
That is the transferable part. **An equivalence test on data small enough to
read tests your algorithm. It does not test your conventions.** Conventions only
surface at a scale where ties occur, and ties are precisely where two correct
implementations are permitted to disagree. Small test data is not a weak version
of large test data; for this class of bug it is the wrong instrument entirely.

## L11 — A limitation is a claim, and it needs evidence too

**What I expected.** At 3, 5 and 6 bits a code straddles the boundary between
two 32-bit words, and packing that needs index arithmetic nobody wants to read.
So I restricted the packer to 2, 4 and 8 bits and wrote a comment explaining
that this was deliberate, that the readable alternative was worth more than the
missing widths, and that "the answer does not change".

**What happened.** The first sweep table came back with twelve of its eighteen
rows empty.

**Why.** The comment was wrong, and it was wrong in the way L10's comment was
wrong — confidently, about something adjacent, in a sentence that reads like a
decision. The missing widths were not an accessory to the chapter, they were the
part where the answer starts to move: 4 bits costs nothing measurable, 2 bits
costs something visible, and everything interesting is in between.

The restriction also rested on a bad mental model. I had been thinking in
"codes per word", which cannot express a code that spans two of them. Codes are
a **bit stream**; words are windows onto it. Written that way — spread each code
into its bits, flatten, refold into groups of 32 — every width from 2 to 8 comes
out in four lines, with no index arithmetic anywhere. The general version is
shorter and easier to read than the restricted one I was defending.

**What changed.** `pack_codes` and `unpack_codes` rewritten as a bit stream, and
the sweep filled in. The row that had been cut turned out to be the most useful
one in the table: 3 bits at group 32 is exactly the same 14.9 MiB as 4 bits at
group 64, and measurably worse at that size. Spend your bits on codes, not on
scales. I could not have said that from the table I nearly published.

**What it cost.** An hour of rewriting, and it is here for the shape of the
mistake rather than the size of it. I documented a limitation with a reassuring
sentence instead of a measurement, and a reassuring sentence in a comment is
still a claim about results — in this case, results I had not produced yet.

## L10 — A metric that answers one question two ways

**What I expected.** `top-1 accuracy` and `top-5 accuracy` are the same
measurement with a different threshold. Nothing to think about.

**What happened.** They were computed by two different rules. Top-1 came from
`argmax(logits) == target`. Top-k came from counting how many logits score
strictly above the true one. Those agree everywhere except on an exact tie, and
on a tie they contradict each other at k=1:

```
logits [3.0, 3.0, 0.0], target 1, k=1   ->   top1 = 0.0, topk = 1.0
```

Argmax breaks ties by taking the lowest index. Counting does not break them at
all. So of two tokens sharing the top score, one was right and the other wrong
by a measure that is supposed to be the same measure.

**Why it survived.** Because it is harmless. Exact float ties over a
four-thousand-token vocabulary do not occur, and the numbers this repository
reports are byte-identical before and after the fix — 24.0% and 38.8% either
way. Nothing was ever wrong on screen.

**What changed.** Both numbers now come from the counting rule, k=1 included,
so top-1 *is* top-k at k=1 by construction rather than by coincidence. The tie
convention is stated in the code instead of being an accident of which function
each line happened to call.

I had also written a comment claiming the counting path "matches what argmax
does". It does not. The comment was confidently wrong about code sitting three
lines below it, which is the most ordinary way for a comment to be wrong.

**What it cost.** Nothing yet, and that is the entry. This is a bug with no
symptom — it produces correct output on every input anyone will ever feed it.
The reason to fix it anyway is that "correct because the failing case never
comes up" is a property of today's inputs, not of the code, and the day it stops
being true is not a day you get a warning about.

## L9 — A zero that is too clean is not a zero

**What I expected.** The context probe shuffles the first half of a window and
rescores the second half. A model that uses its context should get worse. A
model that does not should stay the same.

**What happened.** It stayed exactly the same. Not approximately —
`4.6998` against `4.6998`, a gain of `+0.0000`.

**What I did next.** Not "the model must not use context". Two numbers from two
separate forward passes over a 24.9M-parameter network agreeing to four decimal
places is not a finding, it is a smell. Real quantities are not that polite.

First I checked the shuffle itself, because the cheapest hypothesis is that the
thing I thought I changed did not change:

```
shuffle actually mutated the prefix:  True
first half differs:                   True
second half preserved:                True
```

So the input was fine, which left the only other place a difference could get
lost: the model call.

**Why.** There it was, one line:

```python
total, _ = token_nll(model, inputs[:, half:], targets[:, half:])
```

I sliced the sequence *before* handing it to the model. So the model never saw
the prefix at all — it was scoring the second half in isolation, and shuffling
something the model never reads cannot change anything. Scoring a slice of a
sequence is not the same as running the model on that slice, and I had written
the probe as if it were.

The induction probe had the same bug in the same shape: it scored the second
copy from its own forward pass, hiding the first copy — the very thing the model
was supposed to be copying from — so it was measuring whether a model can copy
from text it cannot see.

**What changed.** `nll_from_logits` in `metrics.py`, so a probe can run the
model on the whole window and restrict only the *grading*. One line in each
probe. And with it fixed, the probes finally say something: the trained model
is positive on context across six of six seeds, and has no induction circuit at
all at 300 steps.

**What it cost.** Nothing, because the exactness gave it away — and that is the
transferable part. **An implausibly clean result is evidence about your
instrument, not about the world.** The version of me that shipped this would
have written "the model does not use its context" in the chapter notes, with a
number to back it up, and been wrong in print.

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
