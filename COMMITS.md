# Commit queue — `handmade-llm`

One file per commit. Selim runs these — nothing in this repository is committed
or pushed by anyone else. Every block is prefixed with the repository name,
because `scratchbench` has its own queue in its own repository.

**The queue is empty as of 2026-07-30.** Everything through chapter 08 is
committed and pushed. What is left before the repository is announced is not git
work, and it is listed below.

---

## Pending

Nothing.

Two commits will exist by the time you read this that are not in the log above:
this file, cleaned up after the queue drained, and whatever you touch next.

```bash
cd ~/handmade-llm

git add COMMITS.md && git commit -m "Empty the commit queue" && git push origin main
```

---

## Not git work, and still open

**Topics are empty.** `gh repo view --json repositoryTopics` returns nothing.
Fifteen, from the launch text:

```bash
gh repo edit selimfedakar/handmade-llm \
  --add-topic mlx --add-topic apple-silicon --add-topic llm \
  --add-topic from-scratch --add-topic transformer --add-topic tokenizer \
  --add-topic bpe --add-topic language-model --add-topic machine-learning \
  --add-topic deep-learning --add-topic macos --add-topic ios \
  --add-topic on-device-ai --add-topic gpt --add-topic education
```

**Read the README on GitHub, not locally.** The GIF and the phone screenshot
render differently there, and the HTML `<p align="center">` block around the
screenshot is the part most likely to surprise you.

**The stash can go.** Its four files are byte-identical to what is committed:

```bash
git stash list
git stash drop stash@{0}
```

---

## If you ever re-record the GIF

`vhs docs/assets/demo.tape` **trains a model.** Until 2026-07-29 it trained into
`runs/latest` — where the step-300 checkpoint lives, the one every number in
chapters 06, 07 and 08 was measured against. Recording the GIF replaced three
hundred steps with sixty and said nothing about it. The tape now passes
`--out-dir runs/demo`; if you edit that line out, you will destroy the checkpoint
again. `docs/LESSONS.md` L17.

Then **look at the result** rather than checking its size: under about 8 MB, text
readable at GitHub's width, the loss visibly falling, and no path in the last
frame that you did not mean to record.

---

## Known warts in the history

**`f834d96` — "fix: resolve merge conflict in README.md"** touches four files
(`.gitignore`, `CLAUDE.md`, `COMMITS.md`, `docs/LESSONS.md`) and its message
describes none of them. It happened while clearing a `git stash pop` conflict:
those four had staged content from the stash, and committing the resolution took
them along. It also committed them at their stash-era state rather than their
current one, which is why they appear again later in the log. Nothing was lost.

The same command, `git checkout --ours README.md`, resolved that conflict toward
the **wrong side** and reverted `README.md` to its pre-chapter-05 state — the one
saying *"Chapters 00–04 work today"*, with no GIF and no phone screenshot. It was
recovered from the conflict's stage-3 blob, which git still had:

```bash
git cat-file -p 150a751d07bee0aaeb0000531bacc6795b39d44a > README.md
```

Transferable, and worth more than the fix: in a `git stash pop` conflict
`--ours` is **HEAD**, not your work. Your work is `--theirs`. Reaching for
`--ours` to "keep mine" does the opposite, and it does it silently, because the
result is a valid file that commits cleanly and looks fine until you read it.

**`d8ff25a` — "Show the demo at the top of the README"** only added the README;
the demo did not exist yet. Cosmetic.

Both are cheap to leave. Only worth rewriting while the repository is young and
nobody has cloned it.

---

## Committed

**2026-07-30 — 48 commits, pushed to `main`, one file each.** Chapter 07, the
whole of chapter 08 (package, golden fixtures, tests, app), both chapter notes,
`docs/AUDIT.md`, the demo GIF and its tape, the phone screenshot, the resume
guard in `03_train/train.py`, and the four files several chapters touched.
Everything through `0491161 Show the phone at the end of the chain`.

**2026-07-29 — `f834d96`.** `.gitignore`, `CLAUDE.md`, `COMMITS.md` and
`docs/LESSONS.md` at their stash-era state, plus the README conflict resolution.
See the warts above.

**2026-07-29 — chapter 06 and the first README assets.** `06_eval/metrics.py`,
`06_eval/probes.py`, `06_eval/evaluate.py`, `06_eval/test_eval.py`,
`docs/06-eval.md`, `03_train/plot_loss.py`, `docs/assets/hero.svg`,
`docs/assets/loss-curve.svg`.

**2026-07-27 — 34 commits.** Chapters 00–05, the community benchmark, the
reproducibility ledger, the lessons page, and the project files. Everything
through `dad6f05 Add commit queue`.

Two blocks from the original queue produced no commits, and that was correct:
the chapter 04 block tried to re-add `02_model/model.py` and `docs/02-model.md`,
and the reproducibility block tried to re-add `docs/LESSONS.md`, but each file
had already been committed with its final content by an earlier block.
