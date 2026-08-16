"""Generate text from a validation-selected character GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from project2_cifar10.engine import resolve_device

from .data import CharTokenizer
from .models import GPTConfig, CharGPT
from .utils import load_checkpoint, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with the W5 character GPT")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    set_seed(args.seed, deterministic=False)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tokenizer = CharTokenizer(checkpoint["characters"])
    model = CharGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved={args.output}")
    print(text)


if __name__ == "__main__":
    main()
