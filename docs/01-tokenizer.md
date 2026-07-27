# Chapter 01 — Byte-level BPE

**What it does.** `01_tokenizer/bpe.py` trains a byte-pair-encoding tokenizer
from scratch and encodes and decodes with it. `train_tokenizer.py` runs it on a
text file of your choice.

**Why it is first.** Before a model learns anything, text has to become
integers, and how you do that decides how much the model has left to learn.
Start at raw UTF-8 bytes and there is no unknown token, ever, in any language.
Everything else is compression on top of that.

**Technologies.** The Python standard library. No `tiktoken`, no
`tokenizers`, not even `regex` — the split pattern is written for stdlib `re`,
which means letters are spelled `[^\W\d_]` instead of `\p{L}`. The file is
meant to be read, and a reader should not have to install anything to run it.

**Decisions.**

- *Byte-level, not character-level.* 256 starting symbols covers every language
  and every emoji. A character-level vocabulary has to decide in advance which
  characters exist.
- *Split before merging.* Merges never cross the split pattern, which is what
  stops tokens like `dog.` or ` the(` from forming. The pattern also caps digit
  runs at three, the way GPT-4 does.
- *Deterministic tie-breaking.* Equally frequent pairs resolve toward the lower
  token ids. Train twice, get the same tokenizer — and a test asserts it.
- *Special tokens are opt-in.* By default `<|endoftext|>` in user text is just
  text. Anything else is an injection waiting to happen.

**The bug that shaped the chapter.** The textbook loop re-counts every pair in
the corpus before every merge. On 1.1 MB with 3,840 merges it had not finished
after ten minutes. Two fixes, neither of which changes a single learned merge:
deduplicate chunks (` the` appears thousands of times and merges identically
every time), and keep a running pair counter with an index from each pair to
the chunks containing it, so a merge only touches what changed.

**10 minutes to 6.2 seconds.** And because a speedup that quietly changes the
output is worse than no speedup,
`test_fast_training_matches_the_slow_obvious_way` runs the textbook version and
asserts the merges are byte-identical. That pattern — readable version, fast
version, equivalence test — became the signature of this repository. Chapter 02
does the same thing for attention.

**Measured on this machine.**

```
TinyShakespeare, 1,115,394 bytes, vocab 4,096
3,840 merges in 6.2s   ->   344,092 tokens   ->   3.24 bytes per token
longest tokens learned: ' Northumberland', 'NORTHUMBERLAND', ' BOLINGBROKE'
33 tests pass in 0.07s
```

Those longest tokens are worth a look. The tokenizer decided, on its own, that
this corpus is mostly people shouting names at each other. It is not wrong.
