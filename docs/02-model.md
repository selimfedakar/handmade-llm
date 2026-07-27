# Chapter 02 — The transformer

**What it does.** `02_model/model.py` is the whole network in one file:
RMSNorm, rotary positions, grouped-query attention with a KV-cache, SwiGLU,
pre-norm blocks, tied embeddings, and sampling. `demo.py` builds one and
generates from it.

**Why one file.** Because you can hold it in your head. Split across six
modules it becomes something you navigate instead of something you read, and
this chapter's only job is to be read.

**Technologies.** MLX. `nn.Linear` and `nn.Embedding` are imported — a matrix
multiply is a matrix multiply — and everything with an idea in it is written
out.

**Decisions, and what each one bought.**

- *RMSNorm over LayerNorm.* Cheaper, and it turns out the mean-centring was
  never doing the work people assumed. Computed in float32: the sum of squares
  is exactly the operation that overflows float16.
- *RoPE over learned positional embeddings.* Position lives in the rotation of
  q and k, so the dot product carries the *distance* between two tokens rather
  than their absolute slots. There is a test for that property, because it is
  the entire reason the technique works.
- *Grouped-query attention.* Several query heads share one KV head. The
  KV-cache is the thing that grows during generation, and on a 16 GB laptop
  that is the constraint that bites first.
- *SwiGLU over ReLU.* A gate that decides per-dimension how much signal passes.
  Better at equal parameter count.
- *Pre-norm.* Each sublayer reads a normalised copy and adds its result back,
  so the residual stream stays a clean path from input to loss. This is why
  deep transformers train at all.
- *Tied embeddings.* On a 25M model the embedding table is a large share of the
  parameters. Reusing it as the output head saves `vocab_size × d_model`, and
  a test asserts exactly that number.

**Attention, twice.** Once written out with an explicit softmax, once through
`mx.fast.scaled_dot_product_attention`. `fused_attention=False` selects the
readable path, and `test_written_out_attention_matches_the_fused_kernel` proves
they agree. Same move as chapter 01.

What the fused kernel actually buys is speed. Chapter 04 measured the peak
memory both ways — 1.236 GiB fused against 1.257 GiB written out, under two
percent — so it will not rescue a configuration that does not fit. MLX skips
the score matrix going forward; the backward pass wants it regardless. Worth
saying plainly, because "fused" reads like "cheaper" and here it is not.

**The bug this chapter is built to prevent.** During generation there is one
query and a growing pile of keys, so the query sits at the *end* of the key
range, not the start. Get that alignment wrong and nothing crashes — generation
just quietly gets worse, and training metrics never mention it. So
`test_cache_matches_a_full_forward_pass` runs both paths, fused and written
out, and demands that feeding tokens one at a time produces exactly what one
pass over the whole sequence produces.

**Measured on this machine.**

```
preset "small": 8 layers x 512, 8 query heads / 4 KV heads, 4,097 vocab
24.9M parameters, 95 MiB at float32
40 tokens generated in 0.33s  ->  122.8 tokens/sec (untrained, cold cache)
27 tests pass in 1.62s
```
