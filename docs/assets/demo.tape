# The README GIF, as a script rather than a lucky screen recording.
#
#   brew install vhs
#   vhs docs/assets/demo.tape
#
# Regenerate it whenever the output changes. A GIF nobody can reproduce goes
# stale the first time a chapter moves, and then the front page of the
# repository is quietly lying about what the code does.
#
# Run `python data/download.py` first — the download is not part of the story.
#
# `--out-dir runs/demo` is not cosmetic. Without it this tape trains into
# `runs/latest`, which is where chapters 06, 07 and 08 read the checkpoint every
# number in this repository was measured against — and sixty steps would quietly
# replace three hundred. Recording the demo would destroy the thing the demo is
# advertising. `docs/LESSONS.md` L17.

Output docs/assets/demo.gif

Set Shell "bash"
Set FontSize 15
Set Width 1180
# 560, not 760: the tallest screen in this recording is the tokenizer's, and it
# ends around row 24. GitHub scales a README image down to about 890px wide, so
# every blank row costs readable pixels in the part that is not blank.
Set Height 560
Set Padding 18
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

# --- 1. a tokenizer, trained on your own text, in six seconds -------------

Type "python 01_tokenizer/train_tokenizer.py --vocab-size 4096"
Sleep 400ms
Enter
Sleep 12s

Type "clear"
Enter
Sleep 500ms

# --- 2. the loss starts falling ------------------------------------------

Type "python 03_train/train.py --preset small --steps 60 --out-dir runs/demo"
Sleep 400ms
Enter

# 60 steps at roughly 12,000 tokens/sec on an M1 Pro. Give it room; a GIF that
# cuts off mid-run is worse than a longer one.
Sleep 30s

# Let the final validation line sit on screen long enough to read.
Sleep 4s
