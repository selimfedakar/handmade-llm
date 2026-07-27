# Commit queue — `handmade-llm`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else. Every block is prefixed with the repository name,
because `scratchbench` has its own queue in its own repository.

Copy a block, run it, then move the block down to **Committed** with the date.

---

## Pending

### `handmade-llm` — first push: chapters 00–01

```bash
cd ~/handmade-llm && git init

git add LICENSE && git commit -m "Add MIT license"
git add .gitignore && git commit -m "Add gitignore for data, checkpoints and Xcode artifacts"
git add requirements.txt && git commit -m "Add dependencies"
git add data/download.py && git commit -m "Add corpus downloader"
git add 00_setup/check_mac.py && git commit -m "Chapter 00: measure the machine before trusting it"
git add docs/00-setup.md && git commit -m "Chapter 00: notes"
git add 01_tokenizer/bpe.py && git commit -m "Chapter 01: byte-level BPE from scratch"
git add 01_tokenizer/test_bpe.py && git commit -m "Chapter 01: tests, including fast-path equivalence proof"
git add 01_tokenizer/train_tokenizer.py && git commit -m "Chapter 01: train a tokenizer on your own text"
git add docs/01-tokenizer.md && git commit -m "Chapter 01: notes"
```

### `handmade-llm` — chapter 02: the transformer

```bash
cd ~/handmade-llm

git add 02_model/model.py && git commit -m "Chapter 02: transformer in MLX, one file"
git add 02_model/test_model.py && git commit -m "Chapter 02: tests for causality, cache equivalence and fused attention"
git add 02_model/demo.py && git commit -m "Chapter 02: build a model and watch it generate"
git add docs/02-model.md && git commit -m "Chapter 02: notes"
```

### `handmade-llm` — chapter 03: training

```bash
cd ~/handmade-llm

git add 03_train/data.py && git commit -m "Chapter 03: deterministic batches from a tokenized corpus"
git add 03_train/train.py && git commit -m "Chapter 03: training loop with warmup, cosine decay and resumable checkpoints"
git add 03_train/test_train.py && git commit -m "Chapter 03: tests for the schedule, the loader, checkpoints and a falling loss"
git add docs/03-training.md && git commit -m "Chapter 03: notes"
```



### `handmade-llm` — chapter 04: scale

```bash
cd ~/handmade-llm

git add 04_scale/memory.py && git commit -m "Chapter 04: predict what a configuration will cost before running it"
git add 04_scale/sweep.py && git commit -m "Chapter 04: measure presets against the prediction"
git add 04_scale/test_scale.py && git commit -m "Chapter 04: tests for the estimator and the sweep"
git add docs/04-scale.md && git commit -m "Chapter 04: notes"
git add 02_model/model.py && git commit -m "Chapter 02: record what the fused kernel actually costs"
git add docs/02-model.md && git commit -m "Chapter 02: note the measured memory cost of fused attention"
```

### `handmade-llm` — chapter 05: fine-tuning

```bash
cd ~/handmade-llm

git add 05_finetune/lora.py && git commit -m "Chapter 05: LoRA written out, with merging"
git add 05_finetune/sft.py && git commit -m "Chapter 05: supervised fine-tuning with a masked loss"
git add 05_finetune/test_finetune.py && git commit -m "Chapter 05: tests for adapters, merging and the loss mask"
git add docs/05-finetune.md && git commit -m "Chapter 05: notes"
git add 03_train/data.py && git commit -m "Chapter 03: write the tokenizer's vocabulary next to the tokens"
git add 03_train/train.py && git commit -m "Chapter 03: take the vocabulary from the tokenizer, not the preset"
git add docs/LESSONS.md && git commit -m "Add the two lessons chapter 05 cost"
```

### `handmade-llm` — the community benchmark

```bash
cd ~/handmade-llm

git add bench/run.py && git commit -m "Add the community benchmark: one command, no setup"
git add bench/README.md && git commit -m "Add the benchmark table"
```

### `handmade-llm` — the reproducibility ledger and the lessons page

```bash
cd ~/handmade-llm

git add docs/REPRODUCIBILITY.md && git commit -m "Document what is reproducible here and what is not"
git add docs/LESSONS.md && git commit -m "Add the lessons page: what each mistake cost and what it changed"
```

### `handmade-llm` — the README GIF

Record it before the first public push. It is the single highest-value asset in
the repository and the README has a placeholder waiting for it.

```bash
brew install vhs
cd ~/handmade-llm
python data/download.py                    # the download is not part of the story
vhs docs/assets/demo.tape                  # writes docs/assets/demo.gif, ~50s

git add docs/assets/demo.tape && git commit -m "Add the demo recording script"
git add docs/assets/demo.gif && git commit -m "Add the README demo"
git add README.md && git commit -m "Show the demo at the top of the README"
```

Check the GIF before committing it: under about 8 MB, text readable at GitHub's
width, and the loss visibly falling. If it is too large, drop `Set FontSize` to
14 and `Set Width` to 1000 and record again.

### `handmade-llm` — project files, last

```bash
cd ~/handmade-llm

git add CLAUDE.md && git commit -m "Add project instructions and verification rules"
git add COMMITS.md && git commit -m "Add commit queue"
git add README.md && git commit -m "Add README"

--burdan 
```

### `handmade-llm` — create the public repository and push

```bash
cd ~/handmade-llm
git branch -M main

# with the GitHub CLI:
gh repo create selimfedakar/handmade-llm --public --source=. --remote=origin \
  --description "Train a real LLM from scratch on the Mac you already own — then put it on your iPhone."
git push -u origin main

# or, if the repository already exists on github.com:
git remote add origin https://github.com/selimfedakar/handmade-llm.git
git push -u origin main
```

Before pushing publicly, check that `data/`, `runs/` and `checkpoints/` are
untracked: `git status --porcelain --ignored | head`.

---

## Committed

_Nothing yet._
