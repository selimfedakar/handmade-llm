# Chapter 00 — Measure the machine

**What it does.** `00_setup/check_mac.py` reads the chip and memory out of
`sysctl`, times a 2048×2048 float16 matmul through MLX, and turns the memory
figure into a parameter ceiling.

**Why it is chapter zero.** Every chapter after this one asks "will this fit?",
and the answer is different on a 16 GB M1 Pro than on a 128 GB M3 Max. I would
rather open with a measurement than with an assumption. It also means that when
chapter 03 feels slow six weeks from now, there is a number from today to
compare against.

**Technologies.** MLX for the throughput test, `sysctl` for the hardware, and
nothing else — this file has to run before anything is installed properly.

**Decisions.**

- *Measure, do not look up.* Published TFLOP/s figures are for a machine in a
  lab, not the one with a browser and an editor open on it.
- *A quarter of memory goes to the system, minimum 4 GB.* Rough, and honest
  about being rough. macOS plus a browser really does take that.
- *16 bytes per parameter* for the ceiling: mixed-precision Adam with
  activations counted in. The printout says out loud that this is a ceiling and
  not a target, because someone will read it as a target otherwise.

**The MLX thing to know.** MLX is lazy — the graph builds and nothing runs until
`mx.eval()`. Three warm-up evals happen before the timer starts. Without them
you time graph construction and report a number ten times too good. This trap
comes back in chapter 03.

**Measured on this machine.**

```
Apple M1 Pro, 8 CPU cores, 16 GiB unified, MLX 0.31.2
2048x2048 float16 matmul: 4.4 ms  ->  3.89 TFLOP/s
usable for training: 12.0 GiB  ->  roughly 805M parameters
```
