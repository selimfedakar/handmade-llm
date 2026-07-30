"""Chapter 08, part two — putting the model inside the app.

`runs/` and `data/` are gitignored, and correctly so: a corpus and a checkpoint
do not belong in a repository people clone to learn from. But the app has to
ship with a model inside it, offline being the whole point, so something has to
carry 14.9 MiB of weights across that line. This is that something.

    python 08_ship_ios/bundle_model.py

It copies into `08_ship_ios/HandmadeLLMApp/Resources/`, which is also
gitignored:

    tokenizer.json                   what the app encodes text with
    model-quantized.safetensors      14.9 MiB, the model that ships
    model-quantized.json             its meta.json, renamed to sit beside it
    model-float32.safetensors        95 MiB, only for the paired measurement
    model-float32.json

The float32 pair is optional and `--quantized-only` leaves it out. It is not
there to be used — nobody ships a 95 MiB model in a phone app — it is there so
that the app can time the two against each other **on the phone**, which is the
question chapter 07 left open and chapter 08 exists to answer.

An app bundle is flat, which is why the two files of each pair are renamed to
share a stem instead of staying in a directory. iOS can be persuaded to keep a
folder as a folder; it is one more thing to get wrong for no gain.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESOURCES = ROOT / "08_ship_ios/HandmadeLLMApp/Resources"


def copy_pair(source: Path, stem: str) -> int:
    """Copy `weights.safetensors` + `meta.json` in under one stem."""
    weights = source / "weights.safetensors"
    metadata = source / "meta.json"
    if not weights.exists() or not metadata.exists():
        raise SystemExit(
            f"{source} is not a checkpoint directory — expected weights.safetensors and meta.json"
        )

    RESOURCES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, RESOURCES / f"{stem}.safetensors")
    shutil.copy2(metadata, RESOURCES / f"{stem}.json")

    size = weights.stat().st_size
    quantization = json.loads(metadata.read_text(encoding="utf-8")).get("quantization")
    described = (
        f"{quantization['bits']}-bit, group {quantization['group_size']}"
        if quantization
        else "float32"
    )
    print(f"  {stem}.safetensors  {size / 1_048_576:6.2f} MiB  ({described})")
    return size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized", type=Path, default=ROOT / "runs/latest/quantized")
    parser.add_argument("--float32", type=Path, default=ROOT / "runs/latest/checkpoint")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer.json")
    parser.add_argument(
        "--quantized-only",
        action="store_true",
        help="leave out the 95 MiB float32 model; the app's comparison then has nothing to compare",
    )
    args = parser.parse_args()

    if not args.tokenizer.exists():
        raise SystemExit(
            f"no {args.tokenizer}. Run:\n"
            "  python data/download.py\n"
            "  python 01_tokenizer/train_tokenizer.py --vocab-size 4096"
        )

    print(f"bundling into {RESOURCES.relative_to(ROOT)}")
    RESOURCES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tokenizer, RESOURCES / "tokenizer.json")
    print(f"  tokenizer.json   {args.tokenizer.stat().st_size / 1024:6.1f} KiB")

    total = copy_pair(args.quantized, "model-quantized")
    if not args.quantized_only:
        if args.float32.exists():
            total += copy_pair(args.float32, "model-float32")
        else:
            print(f"  no float32 checkpoint at {args.float32} — the app's comparison will be off")

    print(f"\n  app payload: {total / 1_048_576:.1f} MiB")
    if not args.quantized_only:
        print("  (--quantized-only ships just the 4-bit model, which is what a release would do)")


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        raise SystemExit("python 3.9 or newer")
    main()
