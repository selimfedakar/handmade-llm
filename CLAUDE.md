# CLAUDE.md — handmade-llm

Train a language model from scratch on Apple Silicon, in eight chapters, ending
with the model running inside an iOS app.

## Thinking protocols (mandatory)

Read `~/claude_efficiency/THINKING_PROTOCOLS.md` at session start
(public mirror: https://github.com/selimfedakar/my_claude_efficiency).
Non-negotiable triggers:
any bug → T2 (3 hypotheses + evidence before fix) and search `docs/PATTERNS.md` first;
any edit to existing code → T1 (read + call sites);
any non-trivial feature → T4 altitude check + T7 edge grid;
any fix → T3 root-cause gate; any design → T5 least-mechanism order;
multi-step task → T6 assumption ledger; before any commit plan/report → T8 rule refresh.
Before presenting substantial work, self-grade with `~/claude_efficiency/FABLE_BAR_RUBRIC.md`.
Every completion claim carries a level from `~/claude_efficiency/VERIFICATION_PROTOCOL.md`
(L0 compiled · L1 tests · L2 exercised+observed · L3 +adversarial+sweep).
**Chapter code is product code: L2 minimum, and for this repository L2 means
pasted terminal output from an actual run — not a passing test alone.**

## Commands

```bash
pip install -r requirements.txt
python data/download.py                                   # fetch the corpus
python 00_setup/check_mac.py                              # measure this machine
python 01_tokenizer/train_tokenizer.py --vocab-size 4096  # ~6s on an M1 Pro
python -m pytest -q                                       # whole suite
python -m pytest 01_tokenizer -q                          # one chapter

# chapter 08 is Swift, and its tests do not run under `swift test` — see below
python 08_ship_ios/export_golden.py                       # fixtures for the Swift side
python 08_ship_ios/bundle_model.py                        # the model into the app
cd 08_ship_ios/HandmadeLLM && xcodebuild test -scheme HandmadeLLM \
  -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation
```

## Map

| Path | Holds |
|---|---|
| `00_setup/` | machine measurement, memory budget, throughput baseline |
| `01_tokenizer/` | byte-level BPE, stdlib only — `bpe.py` is the teaching artifact |
| `02_model/` | transformer in MLX, one file |
| `03_train/` | training loop, checkpoints, 16 GB survival |
| `04_scale/` | measured config-vs-chip tables |
| `05_finetune/` | SFT and LoRA |
| `06_eval/` | did it actually learn |
| `07_quantize/` | 4-bit and the quality it costs |
| `08_ship_ios/` | mlx-swift + SwiftUI — the finish line. `HandmadeLLM/` is the package (tokenizer, model, 32 tests), `HandmadeLLMApp/` the screen, `export_golden.py` the fixtures Python writes for the Swift side to match |
| `bench/` | community table: chip × memory → tokens/sec |
| `data/` | gitignored; downloaded corpora and trained tokenizers |

## Rules

- **Never run `git commit` or `git push`.** Selim runs them. Append the
  commands to `COMMITS.md` instead, one file per commit, and stop there.
- **Atomic commits: one file per commit.** No exceptions.
- **No AI attribution anywhere** — not in commits, code comments, or docs.
- **English only** in every source file and document in this repository.
- **The reader is the user.** Every chapter is read before it is run. Prefer the
  clear version, and where the clear version is too slow to be usable, keep
  both and add a test proving they agree (see below).
- **Never claim a chapter works without pasted output from running it.**
- **Anything that cost more than one attempt goes in `docs/LESSONS.md`**, in
  first person, newest first, with the five parts: what I expected · what
  happened · why · what changed in the repo · what it cost. Near-misses count —
  especially near-misses. If a wrong number almost shipped, say so and say what
  caught it. This page is not a changelog; it is the reasoning that the fixed
  code no longer shows.
- **Every session report to Selim carries the same two sections**: what was
  built, and what was learned the hard way. The second one is not decoration —
  it is half of what the repository is demonstrating.

## The repository's signature move

Chapter 01 ships a naive BPE trainer and a fast one, plus
`test_fast_training_matches_the_slow_obvious_way`, which proves the fast path
learns byte-identical merges. Chapter 02 does the same for attention: an
explicit softmax implementation next to MLX's fused kernel, with an equivalence
test between them.

Chapter 08 does it across a language boundary: `export_golden.py` writes what
Python produces and the Swift tests assert Swift produces the same — token ids,
logits, and greedy generation. It also keeps the pattern *inside* Swift, with
`Unpack.swift` decoding quantized weights by hand next to MLX's kernel.

Keep doing this. The teaching version explains, the fast version ships, and the
test is what makes it honest instead of a claim. It is the most distinctive
thing about this repository — do not quietly drop it to save effort.

## Landmines

- **MLX is lazy.** Nothing executes until `mx.eval()` or a value is read. Timing
  code without `mx.eval()` measures graph construction and reports fantasy
  numbers. Chapter 00 gets this right; copy the pattern.
- **This machine is an M1 Pro with 16 GB.** Roughly 12 GB is usable, about
  805M parameters of ceiling under mixed-precision Adam with activations. If a
  config does not fit here, it does not go in a chapter — the whole promise is
  that the smallest Apple Silicon machine is enough.
- **The tokenizer trainer used to take over ten minutes.** Chunk deduplication
  plus an incremental pair counter brought it to 6.2s. Do not "simplify" that
  back into the textbook loop.
- **The first step at a new working set can be 50x slow.** Chapter 04's sweep
  once reported 40 tok/s for a config that actually runs at 4,200 — the
  allocator pays a one-time cost growing into a large working set, and one
  warm-up step does not absorb it. Any timing code in this repository takes at
  least two warm-up steps and reports a **median**, never a mean. It nearly
  shipped as a published number.
- **Peak memory is repeatable to the byte; throughput is not.** Memory
  comparisons can be read flat. Throughput comparisons need a margin, and so do
  loss comparisons (see the reproducibility section above).
- **Two models compared for speed must be timed in the same alternating loop.**
  Warm-ups and a median handle noise inside one measurement; they do nothing
  about drift between two. Measured sequentially, chapter 07 concluded 4-bit was
  no faster; measured paired, it is ahead every time. Flip the order inside the
  loop to prove the loop did not decide it. `docs/LESSONS.md` L13.
- **Rounding ties are not an edge case in chapter 07, they are the common case.**
  `mx.round` breaks a tie toward the even integer; MLX's quantize kernel breaks
  it away from zero. `_round_half_away` in `07_quantize/quantize.py` exists for
  exactly that and must not be "simplified" back. It cost two codes in 688,128
  to notice. `docs/LESSONS.md` L12.
- **A quantization group size has to divide every matrix in the model**, and
  `d_ff` is where it fails first — 1,344 is not a multiple of 128, so group 128
  is impossible for this architecture at any bit width.
- **`data/` is gitignored.** Never commit a corpus or a trained tokenizer.
- **Python here is Anaconda 3.10** at `/Users/selimfedakar/anaconda3/bin/python3`.
- **Chapter 08 needs the Metal Toolchain, which Xcode 26 does not ship.**
  `xcodebuild -downloadComponent MetalToolchain`, 688 MB, once per machine.
  Without it `swift build` succeeds silently and the first MLX call dies in C++
  with `Failed to load the default metallib`. There is **no CPU fallback** —
  MLX's scheduler builds its GPU stream whichever device you ask for, so
  `Device.setDefault(device: .cpu)` fails inside the call it was meant to avoid.
  `docs/LESSONS.md` L16.
- **`swift test` cannot build MLX's Metal shaders**, even with the toolchain
  installed; mlx-swift's README says so. Chapter 08's command is
  **`xcodebuild test`**. Do not "simplify" it back.
- **MLX does not run in the iOS simulator.** The app builds, installs, launches
  and aborts before the first frame: `mlx/backend/metal/device.cpp:328` calls
  `device_->architecture()->name()->utf8String()`, and `MTLDevice.architecture`
  is nil in the simulator. Nothing on the app side can avoid it. Chapter 08
  needs a physical device. `docs/LESSONS.md` L15.
- **A fixture small enough to commit is small enough to hide your noise floor.**
  The 106,496-weight golden model reproduces Python's logits exactly; the 24.9M
  one is 1.4e-06 away, which is *less* than the gap between MLX's fused
  attention and the written-out softmax inside Python alone. Do not read a first
  appearance of 1e-06 at the real size as a port bug. `docs/LESSONS.md` L14.
- **Swift and Python only agree on greedy decoding.** MLX's Python bindings
  carry a global random state; mlx-swift threads an explicit key. Identical
  logits, different samples. Every cross-language assertion is on `argmax`.

## State as of 2026-07-27

- Chapter 00 `check_mac.py`: **L2** — ran on this machine: M1 Pro, 16 GiB,
  3.89 TFLOP/s fp16, ~805M parameter ceiling.
- Chapter 01 BPE + trainer: **L2** — 4096-vocab run on TinyShakespeare, 6.2s,
  3.24 bytes/token, exact round-trip observed.
- Chapter 01 test suite: **L1** — 33 passed in 0.07s.
- Chapter 02 model: **L2** — `demo.py --preset small` ran: 24.9M parameters,
  95 MiB at float32, 40 tokens generated in 0.33s (122.8 tokens/sec), untrained
  output as expected.
- Chapter 02 test suite: **L1** — 27 passed in 1.62s, including
  `test_written_out_attention_matches_the_fused_kernel`,
  `test_cache_matches_a_full_forward_pass` (both fused and written-out paths)
  and `test_rope_matches_mlx_fast_kernel`.
- Chapter 03 training: **L2** — 300 steps of preset `small` on TinyShakespeare:
  loss 7.0818 → 4.1205, final validation loss 4.6077 (perplexity 100.3),
  11,700–12,200 tokens/sec sustained, checkpoint written and resumable.
- Chapter 03 test suite: **L1** — 19 passed in 1.29s, including
  `test_the_loss_actually_goes_down` and `test_resuming_continues_the_same_run`.
- Chapter 03 resume: **L2** — real CLI interrupt at step 150 of a 300-step run,
  resumed to 300. Straight-through vs resumed: 4.6638/4.6629 at step 160,
  4.1205/4.1273 at step 300, final val 4.6077/4.6067.
- `bench/run.py`: **L2** — ran on this machine: 11,255 train tok/s,
  377.3 generate tok/s, 2,366 MiB peak. Row is in `bench/README.md`.
- Chapter 04 predictor + sweep: **L2** — 44 of 48 combinations run on this
  machine. measured/predicted spans 0.52x–1.24x; the estimator is deliberately
  biased toward over-predicting, because the safe failure for a "will this fit"
  tool is declining a config that would have run. `base` (78.7M) trains at
  batch 8×512 or 16×256 on 16 GiB and stops there.
- Chapter 04 test suite: **L1** — 24 passed.
- `docs/00`–`docs/04` + `docs/REPRODUCIBILITY.md` written. **Every chapter gets
  a docs page — do not skip it.**
- Chapter 05 LoRA + SFT: **L2** — real run against the chapter 03 checkpoint.
  Rank 8 = 114,688 trainable of 25,027,072 (0.46%). Held-out loss 6.5296 →
  5.0644 in 200 steps / 4.5s. Control run confirms the base model never stops
  (60/60 tokens) and the fine-tuned one stops on `<|endoftext|>`.
- Chapter 05 test suite: **L1** — 19 passed.
- Chapter 06 metrics + probes: **L2** — ran against the step-300 checkpoint and
  an untrained control. Held out: loss 4.6032, perplexity 99.8, **2.134 bits per
  byte**, top-1 24.0%, top-5 38.8%; control 8.8388 / 6,896.9 / 4.098 / 0.0%.
  Probes over six seeds: context gain trained +0.0312 [+0.0179, +0.0477] vs
  control −0.0135 [−0.1454, +0.1281]; induction trained −0.0222, straddling
  zero — **no induction circuit at 300 steps, and that negative result stays
  in the notes.**
- Chapter 06 test suite: **L1** — 25 passed in 0.62s, including
  `test_bits_per_byte_survives_a_change_of_tokenizer` (the chapter's thesis)
  and regression guards that fail loudly if either probe goes back to running
  on a slice.
- Chapter 07 quantization: **L2** — ran against the step-300 checkpoint. 4 bits
  in groups of 64: 95.04 MiB → 14.88 MiB (**6.39x**), peak memory during
  generation 97 → 18 MiB, bits per byte 2.134 → 2.136, top-1 24.0% unchanged.
  **99.9% of the learning survives.** Chapter 06's probes run before and after:
  context gain positive **6/6 seeds both ways** (+0.0312 → +0.0316), induction
  absent both ways, memorisation gap +0.3207 → +0.3183.
- Chapter 07 speed: **L2, and it was wrong once.** Measured sequentially the
  answer was "no difference" (seven runs, ordering flipped four times, spread
  244–379 tok/s inside one model). Measured **paired** — both models resident,
  alternating rounds, medians — 4-bit is ahead in all seven measurements,
  ratios 1.02–1.10. Canonical run 346.4 vs 368.7 tok/s. Report the sign, not
  the magnitude. `docs/LESSONS.md` L13.
- Chapter 07 equivalence: **L3** — byte-identical to `mx.quantize` across all
  24,904,192 weights of the real checkpoint, at bits 2/3/4/5/6/8 x group 32/64:
  0 differing words, 0 differing scales, 0 differing biases. Adversarial pass
  found the tie-rounding divergence (L12) that the small-matrix tests missed.
- Chapter 07 test suite: **L1** — 33 passed in 0.42s, including
  `test_the_packed_words_match_mlx_exactly` and
  `test_a_tie_breaks_away_from_zero_the_way_the_kernel_does`.
- Chapter 07 save/load: **L2** — `save_quantized`/`load_quantized` round-trip to
  bit-identical logits and identical greedy generation. This is the file chapter
  08 loads from Swift.
- Chapter 08 Swift port: **L2** — 32 XCTest cases green via `xcodebuild test`
  (`Executed 32 tests, with 0 failures in 1.205 seconds`). Python against Swift:
  the committed 106,496-weight golden model reproduces **bit-identically** in
  both float32 and 4-bit (`worst |Δ| = 0.000e+00` over 1,280 logits, greedy
  generation identical over 24 tokens); the real 24.9M checkpoint agrees to
  `1.431e-06` with identical argmax and identical 16-token greedy output, and
  its embedding output is bit-identical. Weights held: **14.88 MiB**.
- Chapter 08 tokenizer: **L2** — the hand-written Swift splitter matches
  `re.findall(SPLIT_PATTERN, …)` on all 49 fixture texts, including the ones
  chosen to break an ICU-based port (combining marks, connector punctuation,
  superscripts, ZWJ emoji, `U+001C`–`U+001F`). Encoding and decoding match the
  real 4,097-token tokenizer on the same texts.
- Chapter 08 app: **L1 on the phone, L2 on the Mac.** The Xcode project builds
  for the simulator and installs, and MLX aborts on launch there (see landmines
  — this is not a bug in the app). **Nothing has run on a physical device**;
  tokens/sec, peak memory and the paired 4-bit-vs-float32 comparison on a phone
  are the chapter's open items, and `docs/08-ship_ios.md` marks them as not
  measured rather than filling them with laptop numbers.
- Chapter 08 has **no Python tests, on purpose** — its tests are Swift, because
  the thing under test is Swift. `export_golden.py` and `bundle_model.py` are
  fixture and packaging scripts, exercised every time the Swift suite runs.

## Probe rule

**A probe scores a slice; it never *runs* on a slice.** Both chapter 06 probes
were feeding the model only the region they graded, which hid the prefix from
the context probe and the first copy from the induction probe. Use
`nll_from_logits` — full window through the model, grading restricted
afterwards. The tell was a gain of exactly `+0.0000`; an implausibly clean
number is evidence about the instrument, not the world. `docs/LESSONS.md` L9.

**And no probe result is reported from one seed.** Six, with the range printed,
compared against the untrained control's own spread. Where the effect is
smaller than the control's spread — as the context gain is — the claim is the
consistency of the *sign*, never the magnitude.
- Repository is **live and public**: `github.com/selimfedakar/handmade-llm`,
  34 commits on `main`. README opens with `docs/assets/hero.svg` and shows
  `docs/assets/loss-curve.svg` from a real run. The terminal GIF is still
  missing and still the highest-value launch asset — `docs/assets/demo.tape`
  is written and waiting for `brew install vhs`.
- `.gitignore` used to list `data/*.txt|json|bin`, and `data/tokens/*.npy`
  walked through the gap. Now `data/*` with `!data/download.py`. Check
  `git status` before every push; a corpus in a teaching repository is
  embarrassing in a way that is hard to undo.

## The vocabulary rule (learned the hard way, 2026-07-26)

**The tokenizer decides the vocabulary — never a preset default.** An id the
embedding has no row for does not raise in MLX; it returns a number that
changes with the batch shape, and training continues with a quietly wrong loss.

- `prepare()` writes `vocab_size` into `data/tokens/meta.json`; `train()` reads
  it and overrides the preset. Do not remove that.
- Guards: `check_ids_fit_the_model` at dataset-build time in chapter 05, and a
  once-per-run check against both splits in chapter 03.
- Chapter 03 had this bug and survived on luck for two chapters — its corpus
  never contained the special token. `docs/LESSONS.md` L7 is the full trail.

## The reproducibility finding (2026-07-26)

Two runs with the same seed, same config, same data, different processes:
identical at step 10 and 30, apart by 0.0016 at step 60. **MLX training is not
bit-reproducible across processes** — GPU reduction order is not fixed, and
float addition is not associative.

Consequences, all of them load-bearing:
- Resume cannot be asserted bit-identical, because an uninterrupted run is not
  bit-identical to itself. `test_resuming_continues_the_same_run` uses rel=1e-4
  and `docs/REPRODUCIBILITY.md` is the receipt for that tolerance. Do not
  tighten it without re-running the two-run experiment.
- Any A/B comparison of two configurations needs a gap larger than this noise
  floor, or it is measuring the GPU scheduler. Chapter 04 has to respect this.
- What *is* deterministic: the batch sampler (a function of seed and step), the
  learning-rate schedule, and a forward pass within one process. All three have
  tests.

## Whole-suite check

`python -m pytest -q` → 180 passed in 2.07s (33 + 27 + 19 + 24 + 19 + 25 + 33).

Chapter 08 is not in that number and cannot be — it is Swift:

```
cd 08_ship_ios/HandmadeLLM && xcodebuild test -scheme HandmadeLLM \
  -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation
→ Executed 32 tests, with 0 failures (0 unexpected) in 1.205 seconds
```

**212 assertions across the two suites.** Paste the real counts when they
change; never recall them.
- README GIF: **missing.** Cannot be recorded before chapter 03 exists. It is
  the single highest-value launch asset; do not launch without it. Chapter 08
  no longer blocks it.
