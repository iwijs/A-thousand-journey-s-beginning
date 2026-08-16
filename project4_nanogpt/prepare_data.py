"""Download the public Tiny Shakespeare corpus used by Karpathy's char-rnn."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .utils import text_sha256


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Tiny Shakespeare")
    parser.add_argument("--output", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        text = args.output.read_text(encoding="utf-8")
        print(f"exists={args.output} characters={len(text):,} sha256={text_sha256(text)}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    text = payload.decode("utf-8")
    args.output.write_text(text, encoding="utf-8")
    print(f"saved={args.output} characters={len(text):,} sha256={text_sha256(text)} source={DATA_URL}")


if __name__ == "__main__":
    main()
