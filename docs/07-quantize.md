# Chapter 07 — Four bits, and what they cost

**What it does.** `07_quantize/quantize.py` implements group-wise affine
quantization from scratch — the encoder, the decoder, the bit packing, the
layers that use it, and the file format chapter 08 reads. `07_quantize/compare.py`
runs it against the real checkpoint and prints what changed: bytes, tokens per
second, and chapter 06's metrics *and probes* before and after, against the
untrained control. `--sweep` does it at every bit width and group size.

**Why the chapter exists.** Chapter 06 established that the model learned
something. This chapter asks what happens to that when you throw away 87% of
every number, and it exists because chapter 08 cannot happen without it — a
95 MiB float32 model is not what you put in an iOS app.

**Technologies.** MLX. `mx.quantized_matmul` is the only kernel used and the
only one that has to be, because it is what makes the shipping path a 4-bit
model rather than a 4-bit file inflated back to float32.

## The scheme, in four lines

Take 64 consecutive weights. Find the smallest and the largest. Lay 16 evenly
spaced levels across that range. Store each weight as the level it is nearest —
four bits — plus one float32 scale and one float32 offset for the whole group.

Sixty-four float32 weights (256 bytes) become sixty-four 4-bit codes plus two
float32 numbers: 40 bytes, or five bits per weight against thirty-two. The
grouping is what makes it work at all. One scale for a whole 512-wide row would
be set by that row's single largest weight, and every small weight in the row
would round toward the same few codes.

**Decisions.**

- *Written out, then checked against the kernel.* `quantize_groupwise` is plain
  arithmetic with one code per byte. `mx.quantize` is a Metal kernel. The test
  asserts they produce **byte-identical uint32 words** — the same claim chapter
  01 makes about its two BPE trainers. For a lossy transform this is the only
  assertion worth making: every wrong implementation also produces numbers that
  are "close".
- *The zero point is anchored, and that is not the textbook formula.* The
  textbook version puts the group's low end at code 0 and steps evenly to the
  high end. MLX instead anchors code 0 at whichever endpoint is further from
  zero, then shrinks the step until **exactly zero lands on an integer code**.
  Measured cost and benefit are below; the reason to match MLX rather than argue
  with it is that chapter 08 loads these weights through the same kernel.
- *Codes are a bit stream, not codes-per-word.* At 4 and 8 bits a code fits
  inside one 32-bit word. At 3, 5 and 6 it straddles two. Spreading each code
  into its bits, flattening, and refolding into groups of 32 handles every width
  in four lines and needs no index arithmetic. `docs/LESSONS.md` L11 is the
  version of this file that could not do 3, 5 or 6 bits at all, and what that
  cost the table below.
- *The norms stay float32.* Every RMSNorm gain in the model together is 34 KiB
  out of 95 MiB, and each one multiplies the entire residual stream. Nothing to
  win, something real to lose.
- *The embedding is quantized, and it is the interesting one.* Chapter 02 ties
  it to the output head, so one array is both the token table and the projection
  to the vocabulary. It is also the worst-behaved matrix under quantization —
  9.08% relative error against 6.8–8.1% for everything else. `--keep-embedding-float32`
  exists so that is a decision rather than a default.
- *Graded with chapter 06's code, not a new metric.* `compare.py` imports
  `measure` **and all three probes** from `06_eval`. A chapter that invented its
  own quality metric to evaluate quantization would be marking its own homework,
  and chapter 06 has already argued for bits per byte and already has the tests.
  The probes matter as much as the average: a number that did not move is
  consistent with a circuit that is gone and a hole the average closed over.
- *Both models timed in one alternating loop.* Not one after the other. That
  distinction is the difference between this chapter concluding "4-bit is no
  faster" and concluding "4-bit is about 6% faster", and it took seven runs to
  notice. `docs/LESSONS.md` L13.

## Measured on this machine

M1 Pro, 16 GB, MLX 0.31.2. The step-300 checkpoint from chapter 03, 24.9M
parameters, quantized to 4 bits in groups of 64.

```
Size                       float32   quantized
  embedding                8.00 MiB    1.25 MiB
  layers                  87.00 MiB   13.59 MiB
  norms                    0.03 MiB    0.03 MiB
  total                   95.04 MiB   14.88 MiB
  compression                            6.39x

Weight error, before any of it reaches a forward pass
  mean relative RMS       7.34%
  worst matrix            embed (9.08%, 20.8 dB)
  best matrix             layers.1.attn.v_proj (6.82%, 23.3 dB)
  largest single error    0.0206

Quality on the held-out split
                           float32   quantized   untrained
  loss                      4.6032      4.6070      8.8388
  perplexity                  99.8       100.2      6896.9
  bits per byte              2.134       2.136       4.098
  top-1 accuracy             24.0%       24.0%        0.0%
  top-5 accuracy             38.8%       38.8%        0.1%

Generation, 64 tokens from a one-token prompt
  tokens/sec                 346.4       368.7      (paired, 7 alternating rounds)
  peak memory               97 MiB      18 MiB
  slowest–fastest        284–361     323–386

Probes, six seeds each — chapter 06's, unchanged
                    mean  range                  seeds positive
  context gain
    float32      +0.0312  [+0.0179, +0.0477]               6/6
    quantized    +0.0316  [+0.0178, +0.0493]               6/6
    untrained    -0.0135  [-0.1454, +0.1281]               2/6
  induction gain
    float32      -0.0222  [-0.0711, +0.0357]               2/6
    quantized    -0.0184  [-0.0817, +0.0490]               3/6
    untrained    +0.0055  [-0.0271, +0.0355]               4/6
  memorisation gap
    float32      +0.3207
    quantized    +0.3183
    untrained    +0.0492
```

The float32 column reproduces chapter 06 exactly, which is the cross-check that
makes the rest of the table readable.

**What you gain.** 6.39x on disk — 95.04 MiB down to 14.88 MiB, and 14.9 MiB is
what actually lands in `runs/latest/quantized/weights.safetensors`. Peak memory
during generation falls from 97 MiB to 18 MiB, a factor of 5.4, and that is the
number chapter 08 cares about.

**What you give up.** 0.002 bits per byte. Training moved this model 1.964 bits
per byte away from an untrained control; quantizing hands back 0.002 of that, so
**99.9% of the learning survives**. Top-1 and top-5 accuracy do not move at the
resolution the metric is reported at.

**And the model is still doing the same thing.** Bits per byte says the model is
as surprised as it was; it does not say a circuit survived, because an average
can close over a hole. So chapter 06's probes run too, six seeds each. The
context gain is positive on **six seeds out of six before quantizing and six out
of six after** — same tight band, same sign, while the untrained control swings
across zero. Induction is absent both before and after, as it was in chapter 06.
The memorisation gap moves from +0.3207 to +0.3183. Nothing the model had is
gone; nothing it lacked appeared.

**What you gain in speed — after I measured it away once.** 346.4 tokens/sec
against 368.7, so the 4-bit model is about **6% faster**. That number arrived
late and by a different instrument than the one I started with, and the story is
worth more than the number.

The first version of this section said there was *no measurable difference*, with
overlapping ranges printed underneath to prove it. It was wrong. Measuring one
model, then the other, on a machine that drifts means the second one is measured
somewhere else — across seven such runs the ordering flipped four times and the
spread inside one model (244 to 379 tokens/sec) dwarfed any gap between the two.
Peak memory in those same seven runs was byte-identical every time, which is L5
saying the same thing it said in chapter 03.

The fix is to time both models in one alternating loop so the drift lands on
both. Then it holds still:

```
paired, three runs        341.6 / 364.1     336.4 / 361.0     345.5 / 367.1
order reversed inside the round      quantized/float32 = 1.021 to 1.102
```

Seven paired measurements, quantized ahead in all seven, ratios from 1.02 to
1.10. The defensible claim is the **consistency of the sign, not the magnitude**
— the same rule chapter 06 applies to its probes, reached from a completely
different direction. `docs/LESSONS.md` L13.

So: 6.4x smaller, 5.4x less memory at generation, and modestly faster. Whether
the speed gap widens on a phone, where bandwidth is scarcer, was a question for
chapter 08 — and chapter 08 has now measured it on an iPhone 17 Pro Max.

**It does not widen.** Fourteen paired rounds on the phone, both orders, 4-bit
ahead in all fourteen, at ratios of 1.066 and 1.088 — inside the 1.02–1.10 band
this laptop produced. Peak memory travels too: 98 MiB against 17 on the phone,
where this machine measured 97 against 18. What does not travel is the absolute
speed, which is roughly 40% of the laptop's. The ratio is portable; the
throughput is not. [`docs/08-ship_ios.md`](08-ship_ios.md) has the numbers, and
the caveat that one phone — and a fast one — is not a claim about phones.

## Every width, every group size

```
| bits | group | MiB | compression | weight rel RMS | bits/byte | top-1 | top-5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 29.7 | 3.20x | 0.40% | 2.134 | 24.0% | 38.8% |
| 8 | 64 | 26.8 | 3.55x | 0.43% | 2.134 | 24.0% | 38.8% |
| 6 | 32 | 23.8 | 4.00x | 1.64% | 2.134 | 24.0% | 38.8% |
| 6 | 64 | 20.8 | 4.57x | 1.76% | 2.134 | 24.1% | 38.8% |
| 5 | 32 | 20.8 | 4.57x | 3.34% | 2.135 | 24.1% | 38.8% |
| 5 | 64 | 17.8 | 5.33x | 3.57% | 2.135 | 24.1% | 38.9% |
| 4 | 32 | 17.8 | 5.33x | 6.89% | 2.136 | 24.0% | 38.8% |
| 4 | 64 | 14.9 | 6.39x | 7.34% | 2.136 | 24.0% | 38.8% |
| 3 | 32 | 14.9 | 6.39x | 14.50% | 2.142 | 23.8% | 38.6% |
| 3 | 64 | 11.9 | 7.98x | 15.24% | 2.147 | 23.7% | 38.4% |
| 2 | 32 | 11.9 | 7.98x | 31.79% | 2.186 | 23.0% | 37.6% |
| 2 | 64 |  8.9 | 10.63x | 33.00% | 2.200 | 22.8% | 37.0% |
```

Three things fall out of that table, and only one of them was obvious.

**Group size 128 has no rows at all.** The feed-forward block is 1,344 wide and
1,344 is not a multiple of 128, so a partial group would have to exist and its
scale would be fitted to however many weights happened to be left over. The
sweep refuses rather than inventing one. A group size is not a free parameter —
it has to divide every matrix in the model, and `d_ff` is where that goes wrong
first.

**At equal size, spend the bits on codes rather than on scales.** Two pairs of
rows in that table are the same file size, and the comparison goes the same way
both times:

```
14.9 MiB    4 bits / group 64    7.34% weight error   2.136 bits/byte   24.0%
14.9 MiB    3 bits / group 32   14.50% weight error   2.142 bits/byte   23.8%

17.8 MiB    5 bits / group 64    3.57% weight error   2.135 bits/byte   24.1%
17.8 MiB    4 bits / group 32    6.89% weight error   2.136 bits/byte   24.0%
```

Halving the group doubles the scale storage — at group 32 that is two float32
numbers for every 32 weights, a full extra bit per weight — and buys about 5%
in accuracy. Spending that same bit on the code itself buys a factor of two.
This is the most useful row in the chapter and it was in the part I nearly cut;
`docs/LESSONS.md` L11 is why.

**The model is far more robust than the weight error suggests.** At 2 bits the
weights carry 33% relative RMS error — a third of every number is gone — and
bits per byte moves from 2.134 to 2.200, about 3%. Top-1 falls 1.2 points. That
gap between how wrong the weights are and how wrong the model is deserves a
caveat rather than a celebration: **this is a 300-step model at 2.134 bits per
byte, and it is not a strong one.** A model that had learned sharper structure
has more to lose, and there is nothing in this chapter that measures how much.
The way to find out is to train longer and run this chapter again, which is
exactly what the chapter is for.

## What anchoring zero actually costs, on real weights

MLX's scheme is not the textbook affine formula, and the difference is worth a
measurement rather than an opinion. On all 57 matrices of the real checkpoint,
at 4 bits in groups of 64:

```
                       mean relative RMS   worst single weight
  zero anchored              7.337%              0.02060
  textbook (low end = 0)     7.248%              0.01291

  exact zeros in the checkpoint's 24,904,192 weights:   0
```

So the alignment is **worse on both counts** for this model — 1.2% more RMS
error, and a worst-case error 60% larger — and it buys a property this
checkpoint has no use for. Densely trained weights contain no exact zeros at
all.

What it buys is this:

```
  an exact 0.0 reconstructs as   +0.00000000   anchored
                                 +0.00037591   textbook
```

That matters for padding, masks, and pruned or sparse weights, none of which
this model has and all of which a real deployment eventually does. It is also
what the kernel does, and interoperating with the kernel — and with chapter 08's
Swift side, which will call the same one — is worth more than 1.2%. Both paths
are in the code (`align_zero=False`) so the trade stays measurable rather than
becoming folklore.

## The equivalence, and the scale it needed

The chapter's central claim is that the implementation here is not merely
similar to MLX's kernel but identical to it. The evidence:

```
57 matrices, 24,904,192 real trained weights
bits 2, 3, 4, 5, 6, 8  x  group sizes 32, 64

  differing packed words:  0
  differing scales:        0
  differing biases:        0
```

That claim did not survive its first honest test. On small matrices it held
perfectly; on 688,128 random weights two codes disagreed, and on the real
checkpoint forty-six codes in one layer disagreed because the *scale* did.
`mx.round` breaks a tie toward the even integer and MLX's kernel breaks it away
from zero, and quantization manufactures exact ties rather than encountering
them by accident. `docs/LESSONS.md` L12 has the trail.

The transferable part: an equivalence test on data small enough to read tests
your algorithm, not your conventions.

## What went wrong first

Two things, and both are in `docs/LESSONS.md` because the fixes are in the code
and the reasons are not.

**L12 — the tie.** Above. Nearly shipped a "byte-identical" claim on evidence
that could not have detected the counter-example.

**L11 — the limitation I documented instead of measuring.** The first packer
handled only 2, 4 and 8 bits, with a comment explaining that this was a
deliberate simplification and that "the answer does not change". The first sweep
table came back with twelve of eighteen rows empty, and the missing rows were
where the answer moves.

**L13 — the speed conclusion that was backwards.** Two warm-ups and a median is
chapter 04's rule and it is a rule about noise inside one measurement, not about
drift between two. Measured sequentially, this chapter concluded there was no
speed difference. Measured in one alternating loop, the 4-bit model is
consistently ahead. The interpretation was never wrong; the instrument was.

## Running it

```bash
python 07_quantize/compare.py                 # float32 against 4-bit
python 07_quantize/compare.py --sweep         # every width and group size
python 07_quantize/compare.py --bits 8        # a different width
python 07_quantize/compare.py --keep-embedding-float32
python 07_quantize/compare.py --rounds 15     # a tighter paired timing
python -m pytest 07_quantize -q
```

`compare.py` writes `runs/latest/quantized/` — a `weights.safetensors` and a
`meta.json` carrying the architecture, the bit width, the group size and whether
the embedding was quantized. Chapter 08 reads exactly that, from Swift, so the
file has to describe itself; a checkpoint that needs a human to remember how it
was made is not a checkpoint. `load_quantized` round-trips it to **bit-identical
logits** — nothing about saving is allowed to be lossy, because the lossy step
already happened.

```
33 tests pass in 0.42s
```

The one carrying the chapter is `test_the_packed_words_match_mlx_exactly`: six
bit widths, three group sizes, and equality asserted on the packed uint32 words
rather than on the reconstructed floats. Comparing values would have hidden the
packing bug in L11 completely, because a packer that cannot express 3-bit codes
still produces perfectly good 4-bit ones.
