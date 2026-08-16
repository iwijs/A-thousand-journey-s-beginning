"""Extract selected character speeches from Tiny Shakespeare."""

from __future__ import annotations

import argparse
from pathlib import Path

from project4_nanogpt.data import load_corpus
from project4_nanogpt.utils import text_sha256


def extract_speeches(text: str, speakers: set[str]) -> str:
    selected = []
    for block in text.split("\n\n"):
        first_line, _, _ = block.partition("\n")
        if first_line.endswith(":") and first_line[:-1] in speakers:
            selected.append(block.strip())
    if not selected:
        raise ValueError(f"no speeches found for {sorted(speakers)}")
    return "\n\n".join(selected) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a character-specific fine-tuning corpus")
    parser.add_argument("--source", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/romeo_juliet_speeches.txt"))
    parser.add_argument("--speakers", nargs="+", default=["ROMEO", "JULIET"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        text = args.output.read_text(encoding="utf-8")
        print(f"exists={args.output} characters={len(text):,} sha256={text_sha256(text)}")
        return
    domain = extract_speeches(load_corpus(args.source), set(args.speakers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(domain, encoding="utf-8")
    print(
        f"saved={args.output} speakers={','.join(args.speakers)} "
        f"characters={len(domain):,} sha256={text_sha256(domain)}"
    )


if __name__ == "__main__":
    main()
