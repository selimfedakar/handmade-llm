# Commit queue — `handmade-llm`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else. Every block is prefixed with the repository name,
because `scratchbench` has its own queue in its own repository.

Copy a block, run it, then move it down to **Committed**.

---

## Pending

### `handmade-llm` — the README picture and the loss curve

The GIF still needs chapter 08 and a `brew install vhs`. Until then the README
opens with a drawing and shows a real loss curve, both plain SVG, both
committable today.

```bash
cd ~/handmade-llm

git add .gitignore && git commit -m "Ignore all of data/, not only three extensions"
git add 03_train/plot_loss.py && git commit -m "Chapter 03: draw the loss curve without a plotting library"
git add docs/assets/hero.svg && git commit -m "Add the README illustration"
git add docs/assets/loss-curve.svg && git commit -m "Add a loss curve from a real run"
```

`README.md` moved to the shared block at the end, for the same reason
`docs/LESSONS.md` did: it now carries chapter 07's changes as well, and a whole
file is what gets committed.

Regenerate the curve after any run that changes the numbers:

```bash
python 03_train/plot_loss.py
```

### `handmade-llm` — chapter 06: evaluation
--burdan -----------------
```bash
cd ~/handmade-llm

git add 06_eval/metrics.py && git commit -m "Chapter 06: perplexity, bits per byte and accuracy"
git add 06_eval/probes.py && git commit -m "Chapter 06: probes for context use, induction and memorisation"
git add 06_eval/evaluate.py && git commit -m "Chapter 06: evaluate a checkpoint against an untrained control"
git add 06_eval/test_eval.py && git commit -m "Chapter 06: tests for the metrics and the probes"
git add docs/06-eval.md && git commit -m "Chapter 06: notes"
```

`docs/LESSONS.md` and `CLAUDE.md` used to be in this block and have been moved
to the shared block at the end. Chapter 07 landed before any of this was run, so
both files now hold both chapters' changes, and git commits whole files — a
`git add docs/LESSONS.md` here would put chapter 07's lessons into a commit
titled "the lesson chapter 06 cost". One commit each, once, at the end.

### `handmade-llm` — chapter 07: quantization

```bash
cd ~/handmade-llm

git add 07_quantize/quantize.py && git commit -m "Chapter 07: group-wise 4-bit quantization, written out"
git add 07_quantize/compare.py && git commit -m "Chapter 07: measure what four bits gain and what they cost"
git add 07_quantize/test_quantize.py && git commit -m "Chapter 07: tests, including byte equality with MLX's kernel"
git add docs/07-quantize.md && git commit -m "Chapter 07: notes"
```

### `handmade-llm` — chapter 08: the Swift package

The library first, because everything else depends on it. `Package.resolved`
goes in with it: pinning mlx-swift `exact: "0.31.6"` is only half a pin without
the lock file, and this chapter's whole claim is about one build of one kernel.

```bash
cd ~/handmade-llm

git add 08_ship_ios/HandmadeLLM/Package.swift && git commit -m "Chapter 08: a Swift package for the tokenizer, the model and their tests"
git add 08_ship_ios/HandmadeLLM/Package.resolved && git commit -m "Chapter 08: pin mlx-swift to the build the equivalence claims are about"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/TextSplitter.swift && git commit -m "Chapter 08: chapter 01's split pattern, written out by hand in Swift"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/BPETokenizer.swift && git commit -m "Chapter 08: the BPE encoder and decoder in Swift"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Unpack.swift && git commit -m "Chapter 08: unpack and dequantize written out, next to MLX's kernel"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Projection.swift && git commit -m "Chapter 08: one matrix, stored dense or as packed 4-bit codes"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Transformer.swift && git commit -m "Chapter 08: chapter 02's transformer in Swift, both attention paths"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Checkpoint.swift && git commit -m "Chapter 08: read what chapter 07 wrote"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Generator.swift && git commit -m "Chapter 08: generation, and holding back half a character"
git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Benchmark.swift && git commit -m "Chapter 08: time two models in one alternating loop, never one after the other"
```

### `handmade-llm` — chapter 08: what Python writes for Swift to match

`export_golden.py` produces these; they are committed because the equivalence
claim is worth nothing if checking it needs a corpus, a training run and a
Python interpreter. 788 KiB in total, and `.gitignore` carries a deliberate
exception for the two `.safetensors` — read the comment there before adding
anything else.

```bash
cd ~/handmade-llm

git add 08_ship_ios/export_golden.py && git commit -m "Chapter 08: export what Swift has to agree with"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/splits.json && git commit -m "Chapter 08: 49 texts chosen to break a regular-expression port"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tokens.json && git commit -m "Chapter 08: the same texts, all the way through the tokenizer"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32/meta.json && git commit -m "Chapter 08: the golden model's architecture"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32/weights.safetensors && git commit -m "Chapter 08: a 106,496-weight model in float32, small enough to commit"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32.json && git commit -m "Chapter 08: the logits Python got from the float32 golden model"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized/meta.json && git commit -m "Chapter 08: the golden model's quantization recipe"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized/weights.safetensors && git commit -m "Chapter 08: the same model at four bits"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized.json && git commit -m "Chapter 08: the logits Python got from the quantized golden model"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/real-quantized.json && git commit -m "Chapter 08: the 24.9M checkpoint's logits, greedy tokens and intermediates"
```

### `handmade-llm` — chapter 08: the tests

```bash
cd ~/handmade-llm

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden.swift && git commit -m "Chapter 08: find the fixtures, and report the worst difference rather than hiding it"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/TestDevice.swift && git commit -m "Chapter 08: fail with the command to run when the Metal toolchain is missing"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/SplitterTests.swift && git commit -m "Chapter 08: the Swift splitter against the Python pattern"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/TokenizerTests.swift && git commit -m "Chapter 08: the same token ids in both languages"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/ModelTests.swift && git commit -m "Chapter 08: the same logits in both languages, at a tolerance of zero"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/RealCheckpointTests.swift && git commit -m "Chapter 08: the same tests against the checkpoint that ships"
git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/ByteStreamTests.swift && git commit -m "Chapter 08: a character split across two tokens must not become U+FFFD"
```

Run them before committing, and paste the count into `CLAUDE.md` if it moved:

```bash
cd ~/handmade-llm/08_ship_ios/HandmadeLLM
xcodebuild test -scheme HandmadeLLM -destination 'platform=OS X,arch=arm64' \
  -skipPackagePluginValidation -skipMacroValidation
```

### `handmade-llm` — chapter 08: the app

```bash
cd ~/handmade-llm

git add 08_ship_ios/bundle_model.py && git commit -m "Chapter 08: copy the model into the app bundle"
git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp.xcodeproj/project.pbxproj && git commit -m "Chapter 08: an Xcode project with no signing team in it"
git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved && git commit -m "Chapter 08: pin the app's package graph to the package's"
git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/HandmadeLLMApp.swift && git commit -m "Chapter 08: an app with no network code in it"
git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/ModelStore.swift && git commit -m "Chapter 08: one background task owns the model"
git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/ContentView.swift && git commit -m "Chapter 08: a prompt, the tokens, and the two numbers that matter"
git add 08_ship_ios/HandmadeLLMApp/Resources/README.md && git commit -m "Chapter 08: what goes in the app's Resources, and why none of it is committed"
git add docs/08-ship_ios.md && git commit -m "Chapter 08: notes"
```

Before running any of these, confirm the 110 MiB of bundled model is **not** in
`git status` — `bundle_model.py` writes it into a directory whose contents are
ignored, and the only tracked file in there is the README:

```bash
git status --short 08_ship_ios/HandmadeLLMApp/Resources
```

### `handmade-llm` — the files both chapters touched

Run this last. Every command above commits one new file; these four are existing
files carrying changes from chapters 06, 07 **and 08** together — which is why
they are here and not in their chapters' blocks. Git commits whole files, and a
`git add docs/LESSONS.md` up in the chapter 06 block would put chapter 08's
lessons into a commit titled "the lessons chapter 06 cost".

```bash
cd ~/handmade-llm

git add .gitignore && git commit -m "Ignore the bundled model, keep the golden fixtures"
git add docs/LESSONS.md && git commit -m "Add the eight lessons chapters 06, 07 and 08 cost"
git add CLAUDE.md && git commit -m "Record the state after chapter 08, and the four ways its build can lie to you"
git add README.md && git commit -m "Show the phone at the end of the chain"
```

Then, once — **one push for the whole queue, not one per commit.** Twenty-five
pushes in thirty-one seconds on 2026-07-27 cost thirteen contribution squares
and cancelled twenty-four CI runs.

```bash
git push
```

Check before pushing, and actually look rather than assuming:

```bash
git status --short          # runs/ and data/ must not appear at all
git log --oneline -40       # every commit touches exactly one file
du -sh 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden   # ~788 KiB
```

### `handmade-llm` — the README GIF (blocked on chapter 08)

```bash
brew install vhs
cd ~/handmade-llm
python data/download.py                    # the download is not part of the story
vhs docs/assets/demo.tape                  # writes docs/assets/demo.gif, ~50s

git add docs/assets/demo.gif && git commit -m "Add the README demo"
git add README.md && git commit -m "Show the demo at the top of the README"
git push
```

Check the GIF before committing it: under about 8 MB, text readable at GitHub's
width, and the loss visibly falling. If it is too large, drop `Set FontSize` to
14 and `Set Width` to 1000 and record again.

---

## Known wart

`d8ff25a` is titled *"Show the demo at the top of the README"* but it only added
the README — the demo did not exist yet. Cosmetic, and the log is something
people read in a repository like this one. To fix it, in your own terminal:

```bash
cd ~/handmade-llm
git rebase -i HEAD~5        # change `pick d8ff25a` to `reword`, save, then retitle it "Add README"
git push --force-with-lease
```

Only worth doing while the repository is young and nobody has cloned it.

---

## Committed

**2026-07-27 — 34 commits, pushed to `main`.** Chapters 00–05, the community
benchmark, the reproducibility ledger, the lessons page, and the project files.
Everything through `dad6f05 Add commit queue`.

Two blocks from the original queue produced no commits, and that was correct:
the chapter 04 block tried to re-add `02_model/model.py` and `docs/02-model.md`,
and the reproducibility block tried to re-add `docs/LESSONS.md`, but each file
had already been committed with its final content by an earlier block. Nothing
was lost — `git status` was clean afterwards.
