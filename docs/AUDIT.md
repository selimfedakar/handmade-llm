# The closing pass

Before this repository was announced, I read it end to end as a stranger would
and re-ran everything it claims. This page is what that found. It is here for the
same reason `docs/LESSONS.md` is: a repository that argues for measuring things
should be able to show its own last measurement, including the parts that came
back wrong.

Run on 2026-07-29. M1 Pro, 16 GiB, macOS 26.5.1, Xcode 26.6, Python 3.10, mlx
0.31.2, mlx-swift 0.31.6.

## The suites, re-run rather than recalled

```
python -m pytest -q
  180 passed in 1.90s          33 + 27 + 19 + 24 + 19 + 25 + 33

cd 08_ship_ios/HandmadeLLM && xcodebuild test -scheme HandmadeLLM \
  -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation
  Executed 32 tests, with 0 failures (0 unexpected) in 0.837 seconds
```

Both counts match what `CLAUDE.md` claimed. The recorded Swift runtime did not —
it said 1.205s — so it was replaced with a run that actually happened.

Then the replacement moved too: 0.784s on one run, 0.837s on the next. Which is
this repository's own L5 arriving at the scale of a test suite, and it changes
what the line is for. **The counts are the claim. The runtimes are decoration**,
and pinning decoration to three decimal places invites a reader to treat it as a
measurement. Both files now say so.

## What was checked

- **Every chapter has a `docs/NN-*.md` page, and every page has a "Measured on
  this machine" block with real output in it.** Nine pages, nine blocks.
- **Every number in the README resolves to a page in `docs/`.** Compression,
  bits per byte, the 1.964 the model moved, the parameter counts, the memory
  ceiling, the bench row, the cross-language tolerance, the phone ratios.
- **`docs/LESSONS.md` L1–L18 each carry the five parts** — what I expected, what
  happened, why, what changed in the repository, what it cost. L3 had the cause
  folded into the symptom and now states it separately.
- **`git status` leaks nothing.** No corpus, no `runs/`, no bundled model. The
  34 files chapter 08 adds are source, fixtures and the packaging scripts. The
  golden fixtures are 788 KiB and are meant to be there.
- **`requirements.txt` is sufficient.** Every import across the repository is
  stdlib, `mlx`, `numpy` or `pytest`.
- **No dead files, no `__pycache__`, no `.build` leaking into the tree.**
- **LICENSE, quickstart and the `bench/` table are current.**

## What it found

Three defects, all real, all fixed in this pass.

**1. An unresolved merge conflict, in `README.md`, on disk.** Left by a
`git stash pop` that was never finished — conflict markers at the end of the
file, below the licence. The content at stake was a trailing horizontal rule.
Trivial to fix and it would have shipped on the front page of the repository.

**2. Recording the README GIF destroyed the checkpoint every chapter is measured
against.** `docs/assets/demo.tape` trains a model, and `--out-dir` defaults to
`runs/latest` — where the step-300 checkpoint lived. Sixty steps silently
replaced three hundred. Recovered from a byte-exact copy that
`08_ship_ios/bundle_model.py` had made three days earlier for an unrelated
reason. The tape now writes to `runs/demo`. Full account in `docs/LESSONS.md`
L17; the hazard is in `CLAUDE.md` as a landmine, because the destructive default
is in this repository's own quickstart and not in the tape.

**3. The paired benchmark reported peak memory it could not measure.**
`Benchmark.paired` keeps both models resident so the timing is honest, and then
read a process-wide memory counter per model — printing `peak 111 vs 111 MiB`
for models that differ by 6.4x. The arrangement that makes the speed comparison
trustworthy is the same one that makes a memory comparison impossible.
`PairedResult` now reports one process-wide figure and says so. `docs/LESSONS.md`
L18.

Two of those three were caught by *looking at output that had already been
produced* — the last frame of a finished GIF, and a screenshot taken for the
README. Neither was found by reading code.

## The two things that could not be repaired, and what was done instead

The demo overwrote three files and only one of them had a backup. The weights
came back byte-exact; the **optimizer state** and the **training log** did not,
and nothing can recreate them — the run that produced them is gone, and MLX
training is not reproducible across processes, so re-running would not reproduce
them either.

The temptation was to train a fresh 300-step run into `runs/latest` and call it
repaired. That would have been the worst move available. Every published number
in chapters 06, 07 and 08 — bits per byte, the compression ratio, the golden
fixture the Swift tests assert against, the models in the app bundle, the phone
measurements — is tied to *these* weights. New weights would silently
invalidate all of it while looking like a fix.

So the damage is not repaired. It is made **loud**, in the two places where it
could otherwise be mistaken for working:

**`--resume` now refuses.** `load_checkpoint` checks for all three files before
touching MLX and explains what is missing:

```
runs/latest/checkpoint/optimizer.safetensors is missing, so this checkpoint
cannot be resumed from.
Present: ['meta.json', 'optimizer.safetensors.step60-stale', 'weights.safetensors']
Weights alone are enough to evaluate, quantize or ship a model — see chapters
06, 07 and 08 — but not to continue training it. Train a new run with --out-dir
instead of resuming this one.
```

The stale file is renamed rather than deleted so the message can show it. This is
a general improvement, not a patch for one directory: resuming with a missing
optimizer restarts Adam's moment estimates at zero underneath a model three
hundred steps in, which diverges quietly instead of failing — the worst available
outcome, and previously it would have arrived as a bare file-not-found from
inside MLX.

**`plot_loss.py` now cannot overwrite the good curve with a bad one.** The
sixty-step log is renamed to `log.jsonl.demo-60-steps`, so:

```
$ python 03_train/plot_loss.py
No log at .../runs/latest/log.jsonl. Run chapter 03 first.
$ echo $?
1
```

`docs/assets/loss-curve.svg` is unchanged — still `steps 10-300 (6.97 to 4.17)`,
verified by checksum before and after. Had the stale log stayed in place the
script would have exited 0 and written a sixty-step curve over it. The title it
generates does carry the step range, so it would not have been *silent* — but
"not silent" is a bad place to keep the front page of a repository.

## What is still open, and stays open

- **The phone results are one device.** An iPhone 17 Pro Max is not the
  bandwidth-starved machine chapter 07's question had in mind.
- **Model load time on the phone was never timed.**
- **`runs/latest` is not resumable and will not become resumable.** Continuing
  training means a new run in a new `--out-dir`, which is what the flag is for.

## An accident worth keeping

The GIF was recorded twice, which left two independent 60-step training runs in
two processes with the same seed and the same config. They are byte-identical at
steps 10, 20 and 30 and diverge from step 40:

```
step 40    5.873813152313    5.873812675476
step 50    5.785552978516    5.785551071167
step 60    5.535792350769    5.535791397095
```

That is `docs/REPRODUCIBILITY.md`'s central finding, reproduced by accident,
from runs made for a completely different purpose. The GIF displays four decimal
places, so on screen the two runs look identical — which is its own small lesson
about reading a rounded number as agreement.
