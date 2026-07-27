# The table

What each Apple Silicon chip actually does with the same 24.9M model. One
command, no setup, one row.

```bash
python bench/run.py --submit
```

I own one Mac. Everything below the first row came from someone else, and the
table is only worth anything because of that. If you run it, send the row back.

## Results

Benchmark v1 · 8 layers × 512, 8 query heads / 4 KV heads, batch 8 × 256 tokens,
25 measured steps after 5 warm-up steps, greedy generation of 64 tokens.

| Chip | GiB | Train tok/s | Generate tok/s | Peak MiB | MLX | macOS |
|---|---:|---:|---:|---:|---|---|
| Apple M1 Pro | 16 | 11,255 | 377.3 | 2,366 | 0.31.2 | 26.5.1 |

Sorted by chip. Add yours in place.

## What is being measured

**Train tok/s** — steady-state forward, backward, gradient clip and AdamW
update, with `mx.eval()` inside the loop so the number is real. MLX is lazy;
a training benchmark without that call measures graph construction and reports
something wonderful and false.

**Generate tok/s** — single-stream greedy decoding with a warm KV-cache. This
is memory-bandwidth bound, not compute bound, which is why the ordering here
often differs from the training column. Watch for that — it is the most
interesting thing in the table.

**Peak MiB** — `mx.get_peak_memory()` over the measured window, reset after
warm-up. Unified memory, so this is the real number and not a GPU-only slice.

## The rules

- **The configuration is frozen.** Changing it makes rows incomparable. If it
  ever has to change, `BENCH_VERSION` goes up and old rows keep their version.
- **Synthetic tokens.** No corpus, no tokenizer, nothing to download. Random ids
  train just as slowly as real ones, and the point here is speed, not loss.
- **Close what you can.** Not a lab, just the obvious: a browser mid-video will
  cost you a thousand tokens per second and it will not be the chip's fault.
- **Report what you have.** A thermally throttled MacBook Air is a real machine
  that real people own. Note it in the pull request and the row still belongs.

## What this is not

Chapter 04 explores *your* machine: which configurations fit, where the memory
goes, where the prediction and the measurement disagree. This directory does
the opposite — one frozen configuration everywhere, so the only variable left
is the hardware.
