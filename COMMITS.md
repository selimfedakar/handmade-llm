# Commit queue — `handmade-llm`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else. Every block is prefixed with the repository name,
because `scratchbench` has its own queue in its own repository.

**Rewritten 2026-07-29 to match the actual state of the tree.** Work through it
top to bottom. Each command is `add` + `commit` + `push`, one file at a time,
deliberately — the 2026-07-27 accident was **twenty-five pushes in thirty-one
seconds**, which is a rate problem, not a count problem. Take your time and it is
fine.

**Order matters in one place:** the images and `docs/AUDIT.md` are committed
*before* `README.md`, because the README links to them. Push the README first and
GitHub shows broken images until the next push.

---

## Pending — 48 commits

### `handmade-llm` — chapter 07: quantization

```bash
cd ~/handmade-llm

git add 07_quantize/quantize.py && git commit -m "Chapter 07: group-wise 4-bit quantization, written out" && git push origin main

git add 07_quantize/compare.py && git commit -m "Chapter 07: measure what four bits cost" && git push origin main

git add 07_quantize/test_quantize.py && git commit -m "Chapter 07: assert the packed words match MLX exactly" && git push origin main
```

### `handmade-llm` — chapter 08: the Swift package

```bash
cd ~/handmade-llm

git add 08_ship_ios/HandmadeLLM/Package.swift && git commit -m "Chapter 08: pin mlx-swift exactly, because the claim is about the kernel" && git push origin main

git add 08_ship_ios/HandmadeLLM/Package.resolved && git commit -m "Chapter 08: lock the package graph" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/TextSplitter.swift && git commit -m "Chapter 08: chapter 01's regular expression, written out by hand" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/BPETokenizer.swift && git commit -m "Chapter 08: the tokenizer in Swift" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Unpack.swift && git commit -m "Chapter 08: chapter 07's decoder next to MLX's kernel" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Projection.swift && git commit -m "Chapter 08: one matrix, stored two ways" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Transformer.swift && git commit -m "Chapter 08: chapter 02's network in Swift" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Checkpoint.swift && git commit -m "Chapter 08: read the file chapter 07 wrote" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Generator.swift && git commit -m "Chapter 08: sampling, streaming, and holding back half a character" && git push origin main

git add 08_ship_ios/HandmadeLLM/Sources/HandmadeLLM/Benchmark.swift && git commit -m "Chapter 08: two models, one alternating loop" && git push origin main
```

### `handmade-llm` — chapter 08: the golden fixtures

788 KiB total, and they are the reason `xcodebuild test` means anything on a
fresh clone with no corpus, no training run and no phone. Check the size before
adding anything else here.

```bash
cd ~/handmade-llm

git add 08_ship_ios/export_golden.py && git commit -m "Chapter 08: write down what Swift has to agree with" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/splits.json && git commit -m "Chapter 08: 49 texts chosen to break a regex port" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tokens.json && git commit -m "Chapter 08: the same texts as token ids" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32/meta.json && git commit -m "Chapter 08: the tiny model's architecture" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32/weights.safetensors && git commit -m "Chapter 08: a 106,496-weight model in float32" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-float32.json && git commit -m "Chapter 08: the logits Python got from it" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized/meta.json && git commit -m "Chapter 08: the same model, quantized" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized/weights.safetensors && git commit -m "Chapter 08: its 4-bit weights" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/tiny-quantized.json && git commit -m "Chapter 08: and the logits those produce" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden/real-quantized.json && git commit -m "Chapter 08: three points from inside a forward pass of the real checkpoint" && git push origin main
```

### `handmade-llm` — chapter 08: the tests

```bash
cd ~/handmade-llm

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/TestDevice.swift && git commit -m "Chapter 08: fail with the command to run, never skip" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden.swift && git commit -m "Chapter 08: load the fixtures Python wrote" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/ByteStreamTests.swift && git commit -m "Chapter 08: a character split across two tokens" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/SplitterTests.swift && git commit -m "Chapter 08: assert Swift splits text the way Python's re does" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/TokenizerTests.swift && git commit -m "Chapter 08: assert the encoder and decoder agree across the language boundary" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/ModelTests.swift && git commit -m "Chapter 08: bit-identical logits, at a tolerance of exactly zero" && git push origin main

git add 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/RealCheckpointTests.swift && git commit -m "Chapter 08: the 24.9M checkpoint, and where the float32 floor is" && git push origin main
```

### `handmade-llm` — chapter 08: the app

Before running these, confirm the 110 MiB of bundled model is **not** staged. The
directory's contents are ignored and the only tracked file in it is the README:

```bash
git status --short 08_ship_ios/HandmadeLLMApp/Resources
```

```bash
cd ~/handmade-llm
-- burdan 
git add 08_ship_ios/bundle_model.py && git commit -m "Chapter 08: copy the model into the app bundle" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/Resources/README.md && git commit -m "Chapter 08: what goes in the app's Resources, and why none of it is committed" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp.xcodeproj/project.pbxproj && git commit -m "Chapter 08: an Xcode project with no signing team in it" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved && git commit -m "Chapter 08: pin the app's package graph to the package's" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/HandmadeLLMApp.swift && git commit -m "Chapter 08: an app with no network code in it" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/ModelStore.swift && git commit -m "Chapter 08: one background task owns the model" && git push origin main

git add 08_ship_ios/HandmadeLLMApp/HandmadeLLMApp/ContentView.swift && git commit -m "Chapter 08: a prompt, the tokens, and the two numbers that matter" && git push origin main
```

### `handmade-llm` — chapter 03: the fallout from L17

`03_train/train.py` was already committed in the first batch, so this change gets
its own commit rather than a chapter block. `load_checkpoint` now checks for all
three checkpoint files before touching MLX and explains what is missing.

Worth a commit of its own because resuming without an optimizer does not start
Adam fresh — it starts Adam's moment estimates at zero underneath a model three
hundred steps in, so it **diverges quietly** rather than failing.

```bash
cd ~/handmade-llm

git add 03_train/train.py && git commit -m "Chapter 03: refuse to resume a checkpoint with no optimizer state" && git push origin main
```

### `handmade-llm` — the chapter notes

```bash
cd ~/handmade-llm

git add docs/07-quantize.md && git commit -m "Chapter 07: notes" && git push origin main

git add docs/08-ship_ios.md && git commit -m "Chapter 08: notes, and the phone numbers" && git push origin main
```

### `handmade-llm` — the launch assets

⚠ **If you ever re-record the GIF, read this first.** `vhs docs/assets/demo.tape`
trains a model, and until 2026-07-29 it trained into `runs/latest` — where the
step-300 checkpoint lives, the one every number in chapters 06, 07 and 08 was
measured against. Recording the GIF replaced three hundred steps with sixty and
said nothing. The tape now passes `--out-dir runs/demo`; if you edit that line
out, you will destroy the checkpoint again. `docs/LESSONS.md` L17.

```bash
cd ~/handmade-llm

git add docs/assets/demo.tape && git commit -m "Record the README demo as a script rather than a screen capture" && git push origin main

git add docs/assets/demo.gif && git commit -m "Add the README demo" && git push origin main

git add docs/assets/phone.png && git commit -m "Add the model generating on a phone" && git push origin main

git add docs/AUDIT.md && git commit -m "Add the closing pass, and the three things it found" && git push origin main
```

### `handmade-llm` — the files several chapters touched

These carry content from chapters 06, 07 **and** 08 together, which is why they
are here and not in a chapter block: git commits whole files, and a
`git add docs/LESSONS.md` up in the chapter 07 block would put chapter 08's
lessons into a commit titled for chapter 07.

`README.md` is **last** — it links the GIF, the phone screenshot and
`docs/AUDIT.md`, all of which are pushed by the block above.

```bash
cd ~/handmade-llm

git add docs/LESSONS.md && git commit -m "Add the lessons chapters 07 and 08 cost, L17 and L18 included" && git push origin main

git add CLAUDE.md && git commit -m "Record the state after chapter 08, measured on a phone" && git push origin main

git add COMMITS.md && git commit -m "Rewrite the commit queue against the real state of the tree" && git push origin main

git add README.md && git commit -m "Show the phone at the end of the chain" && git push origin main
```

---

## When the queue is empty

```bash
git status --short          # nothing but ignored paths; runs/ and data/ must not appear
git log --oneline -50       # every commit after f834d96 touches exactly one file
du -sh 08_ship_ios/HandmadeLLM/Tests/HandmadeLLMTests/Golden   # ~788 KiB
```

Then open the README **on GitHub** and look at it. The GIF and the phone
screenshot render differently there than in a local preview, and the HTML
`<p align="center">` block around the screenshot is the part most likely to
surprise you.

The stash can go once the tree is clean. Its four files are byte-identical to
what is committed — nothing in it is unaccounted for:

```bash
git stash list
git stash drop stash@{0}
```

Topics are still empty on the repository. Fifteen, from the launch text:

```bash
gh repo edit selimfedakar/handmade-llm \
  --add-topic mlx --add-topic apple-silicon --add-topic llm \
  --add-topic from-scratch --add-topic transformer --add-topic tokenizer \
  --add-topic bpe --add-topic language-model --add-topic machine-learning \
  --add-topic deep-learning --add-topic macos --add-topic ios \
  --add-topic on-device-ai --add-topic gpt --add-topic education
```

---

## Known warts in the history

**`f834d96` — "fix: resolve merge conflict in README.md"** touches four files
(`.gitignore`, `CLAUDE.md`, `COMMITS.md`, `docs/LESSONS.md`) and its message
describes none of them. It happened while clearing a `git stash pop` conflict:
those four had staged content from the stash, and committing the resolution took
them along.

It also committed those four at their **stash-era** state rather than their
current one, which is why they appear again in the queue above. Nothing is lost —
the later commits bring them current.

The same command, `git checkout --ours README.md`, resolved the conflict toward
the **wrong side** and reverted `README.md` to its pre-chapter-05 state — the one
saying *"Chapters 00–04 work today"*, with no GIF and no phone screenshot. It was
recovered from the conflict's stage-3 blob, which git still had:

```bash
git cat-file -p 150a751d07bee0aaeb0000531bacc6795b39d44a > README.md
```

Transferable, and worth more than the fix: in a `git stash pop` conflict
`--ours` is **HEAD**, not your work. Your work is `--theirs`. Reaching for
`--ours` to "keep mine" does the opposite, and it does it silently, because the
result is a valid file that commits cleanly.

**`d8ff25a` — "Show the demo at the top of the README"** only added the README;
the demo did not exist yet. Cosmetic. Both warts are cheap to leave and only
worth rewriting while the repository is young and nobody has cloned it.

---

## Committed

**2026-07-29 — `f834d96`.** `.gitignore`, `CLAUDE.md`, `COMMITS.md` and
`docs/LESSONS.md` at their stash-era state, plus the README conflict resolution.
See the warts above.

**2026-07-29 — chapter 06 and the first README assets, pushed to `main`.**
`06_eval/metrics.py`, `06_eval/probes.py`, `06_eval/evaluate.py`,
`06_eval/test_eval.py`, `docs/06-eval.md`, `03_train/plot_loss.py`,
`docs/assets/hero.svg`, `docs/assets/loss-curve.svg`. Selim's `--burdan ---`
bookmark lived in this block; the block is done and the bookmark is retired.

**2026-07-27 — 34 commits, pushed to `main`.** Chapters 00–05, the community
benchmark, the reproducibility ledger, the lessons page, and the project files.
Everything through `dad6f05 Add commit queue`.

Two blocks from the original queue produced no commits, and that was correct:
the chapter 04 block tried to re-add `02_model/model.py` and `docs/02-model.md`,
and the reproducibility block tried to re-add `docs/LESSONS.md`, but each file
had already been committed with its final content by an earlier block. Nothing
was lost — `git status` was clean afterwards.
