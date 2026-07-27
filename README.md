# handmade-llm

**Train a real language model from scratch on the Mac you already own — then put it on your iPhone.**

No CUDA. No cloud bill. No rented A100s. A laptop, eight chapters, and a model that ends up talking to you from your own pocket.

<!-- TODO(launch): 20s terminal GIF here — clone to first generated sentence. This is the single most important asset in the repo. -->

---

## Why this exists

I kept running into the same wall. Every good "build an LLM from scratch" repository assumes a machine I did not have. `nvidia-smi`, a rented GPU, an eight-card node. The code was excellent and none of it ran on my desk.

So I wrote it on what was on my desk: an M1 Pro with 16 GB of unified memory. The smallest Apple Silicon machine you can reasonably buy. If it runs there, it runs everywhere above it.

That constraint turned out to be the interesting part. When you cannot brute-force anything, you have to actually understand where the memory goes, why the batch size matters, what a KV-cache really costs you. The limit teaches you. I did not plan that — it just happened that way, and I would not change it now.

## What you build

Eight chapters. Each one runs on its own, each one is a few hundred readable lines, each one ends with something you can see working.

| | Chapter | What comes out of it |
|---|---|---|
| 00 | `00_setup` | Your machine, measured. Memory budget and a tokens/sec baseline. |
| 01 | `01_tokenizer` | Byte-level BPE, from scratch. Trained on your own text. |
| 02 | `02_model` | A transformer in MLX. One file. No framework underneath it. |
| 03 | `03_train` | The training loop. Loss curve, checkpoints, resume, 16 GB survival. |
| 04 | `04_scale` | Which chip holds which config. Measured, not guessed. |
| 05 | `05_finetune` | SFT and LoRA. Your model starts answering instead of rambling. |
| 06 | `06_eval` | Did it actually learn something, or does it just sound like it did? |
| 07 | `07_quantize` | 4-bit. What you gain, what you give up, in numbers. |
| 08 | `08_ship_ios` | mlx-swift and SwiftUI. The model runs on your phone, offline. |

That last chapter is the one I have not seen anywhere else. Plenty of repositories teach you to train a model. Almost none of them hand it back to you as an app you can open on the train.

Each chapter has a short page in [`docs/`](docs/) — what it does, why it is built that way, and the numbers it produced on my machine. And [`docs/LESSONS.md`](docs/LESSONS.md) is the other half of that: the things I got wrong first, including the benchmark number that was fifty times off and nearly shipped. The fixes are in the code; the reasons are only there.

## Two claims, and the receipts

Most repositories tell you their fast path is equivalent to the obvious one, and that their checkpoints resume cleanly. I wrote the tests instead.

**Every shortcut is proved equivalent to the version you can read.** Chapter 01 keeps the textbook BPE trainer next to the fast one and asserts they learn byte-identical merges — the fast one turned ten minutes into 6.2 seconds. Chapter 02 keeps an explicit softmax next to MLX's fused kernel and asserts they agree. The readable version explains, the fast version ships, and the test is what makes that an argument rather than a promise.

**Training here is not bit-reproducible, and I can show you.** Same seed, same config, two runs: identical at step 10, identical at step 30, apart by 0.0016 at step 60. That is what a GPU reduction does to floating-point addition, it is not a bug, and it decides what a checkpoint can honestly claim. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) has the experiments, the numbers, and the parts that *are* deterministic — the data pipeline, the schedule, the forward pass.

I have not found another training repository that checks this. Everyone sets the seed and moves on.

## Quickstart

```bash
git clone https://github.com/selimfedakar/handmade-llm
cd handmade-llm
pip install -r requirements.txt

python 00_setup/check_mac.py          # what your machine can handle
python data/download.py               # a small corpus to start on
python 01_tokenizer/train_tokenizer.py --vocab-size 4096
python 02_model/demo.py               # an untrained model, generating
python 03_train/train.py --steps 300  # the loss starts falling
```

On an M1 Pro with 16 GB that is about four minutes from clone to a falling loss curve. Chapters 00–04 work today; the rest are landing in order.

Before you pick a size, ask what your machine will hold:

```bash
python 04_scale/sweep.py               # predicted vs measured, on your chip
```

On a 16 GB M1 Pro: 78.7M parameters train at batch 8 × 512, and stop there.

Training on a laptop you also need for other things:

```bash
python 03_train/train.py --steps 5000 --stop-after 500      # an hour tonight
python 03_train/train.py --steps 5000 --resume runs/latest/checkpoint   # tomorrow
```

`--steps` describes the whole run, so the schedule does not care how many sittings you take.

## Why MLX and not PyTorch

PyTorch on MPS works, mostly. But "mostly" is where the evenings go: an operator that silently falls back to CPU, a dtype that behaves differently than it does on CUDA, a memory model built for a machine with a separate GPU card.

MLX was built for this hardware. Unified memory is the default assumption, not a workaround. On an M1 Pro the difference is not subtle.

And there is a second reason, which only shows up in chapter 08: `mlx-swift` exists. That is the bridge that lets a model you trained in Python on Monday run inside a Swift app on Tuesday. Without it, the last chapter cannot be written at all.

Where the PyTorch equivalent is meaningfully different, each chapter says so.

## This is not a nanoGPT replacement

Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) are the reason a repository like this one can exist at all. I learned from both. If you have a CUDA machine and you want to reproduce GPT-2 properly, go there.

This repository answers a narrower question: *what is the complete chain, from raw bytes to an app on your phone, on a laptop with no discrete GPU?*

Different promise. Same family.

## The benchmark table

[`bench/`](bench/) holds a community table: the same 24.9M model on every chip. No corpus, no tokenizer, nothing to download — one command, about a minute, one row.

```bash
python bench/run.py --submit
```

| Chip | GiB | Train tok/s | Generate tok/s | Peak MiB |
|---|---:|---:|---:|---:|
| Apple M1 Pro | 16 | 11,255 | 377.3 | 2,366 |

I have one Mac. The table is only useful if it has more than one Mac in it — so if you run this, send the numbers back. That table is the thing I would have wanted before buying anything.

## Who is writing this

I am Selim Fedakar, a computer science student in Los Angeles, co-founder and CTO of two small hardware-and-AI companies, with two apps live on the App Store. I have been working through Stanford's CS336 on my own for a while now, and this repository is where the parts that finally clicked get written down properly.

I am not a researcher. I am someone who wanted the whole chain to fit on one laptop, and kept going until it did.

## License

MIT. Take it, teach with it, ship with it.

---

*Bir başka gün, bir başka yerde, bir başka zaman ve bir başka mekânda, tekrar görüşünceye kadar kendinize çok iyi bakın.*
