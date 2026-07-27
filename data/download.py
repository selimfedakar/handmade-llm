"""Fetch a small corpus so chapter 01 has something to chew on.

    python data/download.py

TinyShakespeare is about 1.1 MB. Small enough to train a tokenizer on in
seconds, large enough that the merges it learns are not noise.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
SOURCES = {
    "tinyshakespeare.txt": (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    ),
}


def download(name: str, url: str) -> int:
    target = DATA_DIR / name
    if target.exists():
        print(f"{name} already here ({target.stat().st_size:,} bytes)")
        return 0

    print(f"Downloading {name} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not download {name}: {exc}", file=sys.stderr)
        return 1

    target.write_bytes(payload)
    print(f"Saved {name} ({len(payload):,} bytes)")
    return 0


def main() -> int:
    return max(download(name, url) for name, url in SOURCES.items())


if __name__ == "__main__":
    raise SystemExit(main())
