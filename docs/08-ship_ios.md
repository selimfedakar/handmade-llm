# Chapter 08 — The phone

**What it does.** `08_ship_ios/` is the model on an iPhone. A Swift package
holds the tokenizer, the transformer and the loader for chapter 07's file; an
Xcode project wraps them in one SwiftUI screen. Nothing downloads, nothing
phones home, and the app asks for no permissions — the 14.9 MiB checkpoint is
inside the bundle and everything after that happens on the device.

**Why the chapter exists.** Every other chapter is a step toward a claim that
nobody in this genre makes: not *train a model*, but *train a model and then
carry it*. That last step is where the whole promise either lands or does not,
and it is also where the numbers get honest — a laptop plugged into the wall
forgives a lot that a phone does not.

**Technologies.** [mlx-swift](https://github.com/ml-explore/mlx-swift), pinned
to **0.31.6** with `exact:`, against mlx 0.31.2 in Python. Both pin the same
underlying MLX C++ commit, which is the point: chapter 07 asserts byte-identical
agreement with a specific quantize kernel, and this chapter asserts Swift reads
those bytes back to identical logits. Two claims about one build of one library,
so both sides name it. `upToNextMinor` would let the kernel move underneath a
claim that is about the kernel.

## The shape of it

```
08_ship_ios/
  export_golden.py          what Swift has to agree with, written by Python
  bundle_model.py           copies the model into the app bundle
  HandmadeLLM/              a SwiftPM package — tokenizer, model, tests
    Sources/HandmadeLLM/
      TextSplitter.swift    chapter 01's regular expression, by hand
      BPETokenizer.swift    chapter 01's encoder and decoder
      Unpack.swift          chapter 07's decoder, written out
      Projection.swift      one matrix, stored two ways
      Transformer.swift     chapter 02's network
      Checkpoint.swift      reads runs/latest/quantized/
      Generator.swift       sampling, streaming, and holding back half a character
      Benchmark.swift       two models, one alternating loop
    Tests/HandmadeLLMTests/
      Golden/               fixtures Python wrote — 788 KiB, committed
  HandmadeLLMApp/           the SwiftUI screen
```

The split is not tidiness. Everything that can be checked without a phone lives
in the package, so the chapter's central claim runs from a terminal on a Mac:

```bash
cd 08_ship_ios/HandmadeLLM
xcodebuild test -scheme HandmadeLLM -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation
```

A claim that can only be verified by tapping a button on somebody's phone is not
a claim a reader can check.

## The equivalence, and what it took

The repository's signature move is a readable implementation, a fast one, and a
test proving they agree. Chapter 01 does it for BPE, chapter 02 for attention,
chapter 07 for quantization against MLX's kernel. Here the two implementations
are in **different languages**, which turns out to be a harder version of the
same exercise, because a language brings its own conventions along.

`export_golden.py` writes what Python does; the Swift tests assert Swift does the
same:

```
splits.json          49 texts -> chunks, straight from re.findall
tokens.json          the same texts -> token ids, and back
tiny-float32/        a 106,496-weight model, float32, with its logits
tiny-quantized/      the same model at 4 bits, with its logits
real-quantized.json  the 24.9M checkpoint: logits, greedy tokens, intermediates
```

The tiny model is 490 KiB of weights and is committed. It has grouped-query
attention, a tied quantized embedding, a KV-cache and both attention paths — it
is small in size and not in kind, so `xcodebuild test` on a fresh clone with no
corpus, no training run and no phone still exercises all of it.

### The regular expression that could not come along

Chapter 01 splits text before any merge happens, with one pattern:

```python
r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"
```

The obvious move was to hand that string to `NSRegularExpression`. It would have
worked, for a while.

**`\w` is not the same set in the two engines.** Python's `re` builds it from
CPython's own Unicode tables — alphanumeric or underscore. ICU, which is what
`NSRegularExpression` runs on, defines it as
`[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}‌‍]`. Those agree on every character
in TinyShakespeare and part company on combining marks, on connector
punctuation, and on a handful of scripts. Feed the model a different token
sequence than it was trained on and nothing crashes; it just gets quietly worse
on the inputs where the two disagree, forever.

So `TextSplitter.swift` writes the character classes out as named predicates,
each citing the CPython rule it reproduces, and the fixture contains the inputs
chosen to break them. Three of the results are worth stating on their own:

```
"snake_case_name"   ->  "snake"  "case"  "name"
```

The underscore is **dropped**. It is a word character, so `[^\W\d_]` and
`[^\s\w]` both exclude it, and no other branch wants it — and `re.findall`
returns matches, not a partition, so it is simply skipped. Chapter 01's
tokenizer cannot see an underscore. Surprising, true, and reproduced on purpose.

```
"́ alone"      ->  "́"  " alone"
"a‿b"          ->  "a"  "‿"  "b"
```

A lone combining acute and a connector punctuation character are *punctuation* to
Python and *word characters* to ICU. A regex port would have swallowed both.

That is `docs/LESSONS.md` L12 in different clothes — an equivalence test on data
small enough to read tests your algorithm, not your conventions. ASCII English is
data small enough to read.

### Greedy, not sampled

Both languages draw from the same distribution, but MLX's Python bindings carry a
global random state and mlx-swift threads an explicit key, so identical logits
still produce different samples. Comparing sampled text would compare two random
number generators. Every cross-language assertion here is on greedy decoding,
and the app samples because a user is not a test.

## Measured on this machine

M1 Pro, 16 GB, macOS 26.5.1, Xcode 26.6, Swift 6.3.3, mlx-swift 0.31.6 against
mlx 0.31.2. The step-300 checkpoint from chapter 03, quantized by chapter 07.

```
xcodebuild test -scheme HandmadeLLM -destination 'platform=OS X,arch=arm64'

  Executed 32 tests, with 0 failures (0 unexpected) in 0.837 seconds

Python against Swift — the tiny model, both forms
  float32   logits: 1280 values, worst |Δ| = 0.000e+00
  4-bit     logits: 1280 values, worst |Δ| = 0.000e+00
  greedy generation, 24 tokens                       identical, both forms
  weights: 417.2 KiB float32, 66.2 KiB quantized              6.30x

Python against Swift — the 24.9M checkpoint that ships
  after the embedding      worst |Δ| = 0.000e+00
  after eight layers       worst |Δ| = 9.537e-07
  after the output head    worst |Δ| = 1.431e-06   (4,097 logits)
  argmax                                            identical
  greedy generation, 16 tokens                      identical
  weights held                                      14.88 MiB

Two correct implementations of the same model, in one language
  fused vs written-out attention, real model   worst |Δ| = 1.907e-06
  fused vs written-out attention, tiny model   worst |Δ| = 0.000e+00

Inside Swift
  written-out decode vs MLX's kernel           worst |Δ| = 2.980e-08
  KV-cache step-by-step vs one full pass       worst |Δ| = 1.4–1.7e-06

App payload
  tokenizer.json                  172.5 KiB
  model-quantized.safetensors      14.89 MiB
  model-float32.safetensors        95.04 MiB   (only for the comparison)
                                  109.9 MiB total, or 15.1 with --quantized-only
```

**Read the third block before the second.** The 1.4e-06 between Python and Swift
on the real model is smaller than the 1.9e-06 between MLX's fused attention and
the written-out softmax **inside Python alone** — two implementations of one
operation, neither of them wrong. So that gap is not a port defect with a cause
worth finding. It is the float32 floor at eight layers of 512 numbers, and the
tiny model reproduces Python exactly only because two layers of 64 never reach
it.

Getting there took an experiment rather than an argument. The first version of
this port decoded the quantized embedding through MLX's `dequantized` kernel
while chapter 07 does that arithmetic itself; the two agree to 3e-08, and by the
output head that had become 2e-06. `Unpack.swift` now performs the same
arithmetic in the same order, the embedding comes out **bit-identical**, and the
logit comparison is asserted at a tolerance of exactly zero for the tiny models.
A tolerance of 1e-4 would have passed the whole way and nobody would have looked.

## Measured on the phone

iPhone 17 Pro Max (`iPhone18,2`), iOS 26.5.2, built with Xcode 26.6 and installed
over the cable. Same bundle as everything above: the step-300 checkpoint from
chapter 03, quantized by chapter 07.

**The paired comparison**, which is what chapter 07 deferred to this chapter.
Seven recorded rounds after two warm-up rounds, 64 greedy tokens per model per
round, both models resident, alternating inside one loop — and then the whole
thing again with the order inside the round reversed. Transcribed off the device
exactly as it read, including the part of it that turned out to be wrong:

```
float32 139.1 vs 4-bit 151.3 tok/s (1.088x) · 4-bit ahead in all 7 · peak 111 vs 111 MiB · float32 first

float32 135.1 vs 4-bit 144.0 tok/s (1.066x) · 4-bit ahead in all 7 · peak 111 vs 111 MiB · 4-bit first
```

**Fourteen paired measurements, 4-bit ahead in fourteen.** The claim is that sign,
not the ratio — the same rule chapter 06 uses for its probes and chapter 07 for
this measurement on the laptop.

`peak 111 vs 111 MiB` is the wrong part, and `Benchmark.swift` no longer prints
it that way — it now reports one process-wide figure, `111 MiB peak, both
resident`, for reasons worked through at the end of this page. The output above
is left as it was produced rather than re-rendered in the new format, because
the run that produced these timings produced that line too.

**Single-model generation**, which is what the app does when you press Run:
sampled at temperature 0.8 with top-k 40, 96 tokens, streaming into the view.

```
4-bit     103.8 tok/s    peak  17 MiB    weights 14.9 MiB   fresh launch, first generation
float32   138.8 tok/s    peak 112 MiB    weights 95.0 MiB   after the paired benchmark
float32   125.2 tok/s    peak  98 MiB    weights 95.0 MiB   fresh launch, first generation
```

### Does the gap widen on a phone?

Chapter 07 ended by asking, and refusing to answer: 4-bit was about 6% faster on
the laptop, and *whether that widens on a phone, where bandwidth is scarcer,* was
left here.

**It does not widen.** On the laptop, seven paired measurements gave ratios from
1.02 to 1.10. On this phone, 1.066 and 1.088 — inside the same band. The 4-bit
model is consistently faster in both places, and by about the same amount.

Two honest limits on that. **One phone is one phone**, and this one is an iPhone
17 Pro Max, which is not obviously the bandwidth-starved device the question had
in mind; the premise deserves a cheaper phone before anyone generalises from it.
And the phone was doing 139–151 tok/s against the laptop's 346–369, so it is
roughly 40% of the laptop's throughput in absolute terms — the *ratio* travels,
the speed does not.

### The single-model numbers say the opposite, and that is the point

Read the table above as a comparison and it says float32 is faster — 125.2
against 103.8 on the fairest pairing available there, both first-generations
after a fresh launch. That is the **reverse** of what fourteen paired rounds
found on the same device in the same half hour.

I cannot fully account for the gap, and saying so is more useful than a story.
Those readings differ from the paired loop in four ways at once — they sample
with top-k rather than decoding greedily, they stream every token into a SwiftUI
view, they are single unrepeated measurements with no warm-up, and they were
taken minutes apart on a device whose thermal state was moving. Any of those
could dominate. I did not isolate which, because isolating it would answer a
question nobody has: **the readout is a readout, not an instrument.**

What it demonstrates is L13, on the hardware L13 predicted would be worse.
Chapter 07 learned on a laptop that two models timed one after another are timed
on two different machines. A phone has less thermal mass and throttles harder,
and here the difference between the two arrangements is not a few percent of
noise — it is the **sign of the answer**. If the only number in this chapter had
been the one at the bottom of the screen, chapter 08 would have reported that
quantization makes the model slower on a phone, and it would have had a
measurement to show you.

That is why there is a Compare button. The app cannot answer its own question
from its own readout.

### Peak memory, and the number that is not a comparison

`111 MiB peak, both models resident` is not a finding about either model. It is
one process-wide counter read while 95 MiB and 15 MiB of weights are both loaded,
which is exactly the arrangement that makes the speed measurement trustworthy.
The instrument that fixes the timing makes the memory column meaningless, and the
summary line says so rather than printing two identical numbers side by side.
`docs/LESSONS.md` L18.

The per-model figure has to come from a single-model run, each on its own launch,
and once taken that way the phone reproduces the laptop almost exactly:

```
                    phone      laptop (chapter 07)
4-bit               17 MiB     18 MiB
float32             98 MiB     97 MiB
                   5.8x        5.4x
```

Getting there needed one more experiment, and it is worth writing down because
the first attempt was wrong. Measured in one session with the picker switched
from 4-bit to float32, the float32 reading was **112 MiB**, not 98 — because MLX
allocates a model's buffers when it is first *used* rather than when it is
loaded, so that reading carried the 4-bit model's 15 MiB along with it. On this
screen, **measurement order contaminates peak memory**.

I noticed because 112 − 15 lands on the laptop's 97, wrote that subtraction into
this file, and then flagged it as arithmetic rather than a measurement. Which was
the right label and the wrong place to stop: the actual experiment — quit the app,
relaunch, select float32 before anything else is touched — takes thirty seconds
and returns **98 MiB**. The arithmetic was right. It was still a guess, and this
repository does not publish guesses that a thirty-second run can replace.

### What still is not measured

Model load time from a cold app start was not timed. One device, one iOS version,
one thermal state; the `bench/` table exists for Macs and has no phone column.

## The simulator does not work, and here is the line

Worth checking early, so: it was, and it does not.

```
$ xcrun simctl launch --console <sim> com.selimfedakar.handmade-llm
.../c++/v1/string:1062: libc++ Hardening assertion __s != nullptr failed:
basic_string(const char*) detected nullptr
```

The app builds for the simulator, installs, launches, and dies before the first
frame. The cause is one line, in `mlx/backend/metal/device.cpp`:

```cpp
arch_ = std::string(device_->architecture()->name()->utf8String());
```

`MTLDevice.architecture` is not implemented by the simulator's Metal, so it
returns nil, `name()` on nil returns nullptr, and `std::string(nullptr)` aborts.
Nothing on the app side can avoid it — the crash happens while MLX is
constructing its device, before any of this repository's code runs.

**So chapter 08 needs a physical device.** Not "prefers", not "is faster on".
Plan the chapter around that: the tests are on macOS, where the same Metal
backend works, and the phone is for the numbers.

## Two build problems that cost an evening between them

**The Metal compiler is not in Xcode any more.** Since Xcode 26 it is a separate
download, and without it `swift build` succeeds and says nothing, while the first
MLX call dies four layers down:

```
MLX error: Failed to load the default metallib.
library not found library not found library not found library not found
  at .../mlx-c/mlx/c/stream.cpp:106
```

```bash
xcodebuild -downloadComponent MetalToolchain      # 688 MB, once
```

There is no CPU fallback, which was the first thing tried. MLX's scheduler builds
its per-device streams when it is first touched, GPU included, so
`Device.setDefault(device: .cpu)` fails inside the very call it was meant to
avoid — and the line number in the error says `mlx_default_cpu_stream_new`, which
reads like the CPU path is broken and is really the scheduler initialising both.

**And `swift test` cannot build them either**, even with the toolchain
installed — mlx-swift's own README says so, in a note that is easy to read past:

> SwiftPM (command line) cannot build the Metal shaders so the ultimate build has
> to be done via Xcode.

`xcodebuild test` does build them. So the command in this chapter is `xcodebuild
test`, and `TestDevice.swift` checks for the compiled library before MLX is
touched and fails with the fix in the message rather than letting the process
abort. It does not *skip*: a chapter whose central test quietly reports "skipped"
on a machine that cannot run it is the kind of green checkmark this repository is
written against.

## Running it

```bash
# once
xcodebuild -downloadComponent MetalToolchain

# from the repository root, after chapter 07 has produced runs/latest/quantized/
python 08_ship_ios/export_golden.py            # fixtures for the Swift tests
python 08_ship_ios/bundle_model.py             # the model into the app bundle
python 08_ship_ios/bundle_model.py --quantized-only   # what a release ships

cd 08_ship_ios/HandmadeLLM
xcodebuild test -scheme HandmadeLLM -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation

cd ../HandmadeLLMApp
open HandmadeLLMApp.xcodeproj                  # then run on a connected iPhone
```

`export_golden.py` runs against whatever is on this machine and skips what is
not there, so the tokenizer fixtures need `data/tokenizer.json` and the
`real-quantized.json` fixture needs `runs/latest/quantized/`. Both are
gitignored; the tests that use them skip with the command to run.

The app needs a signing team. `HandmadeLLMApp.xcodeproj` leaves
`DEVELOPMENT_TEAM` unset so that a clone does not inherit somebody else's — set
it in Xcode's Signing & Capabilities, or on the command line with
`DEVELOPMENT_TEAM=XXXXXXXXXX`.
