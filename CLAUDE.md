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
| `08_ship_ios/` | mlx-swift + SwiftUI — the finish line |
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
- **`data/` is gitignored.** Never commit a corpus or a trained tokenizer.
- **Python here is Anaconda 3.10** at `/Users/selimfedakar/anaconda3/bin/python3`.

## State as of 2026-07-26

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
- Chapters 06–08: **not started.** Next is `06_eval`.

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

`python -m pytest -q` → 122 passed (33 + 27 + 19 + 24 + 19). Paste the real
count when it changes; never recall it.
- README GIF: **missing.** Cannot be recorded before chapter 03 exists. It is
  the single highest-value launch asset; do not launch without it.
