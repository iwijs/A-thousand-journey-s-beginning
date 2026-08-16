"""Generate text from a W6 LoRA checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from project2_cifar10.engine import resolve_device
from project2_cifar10.utils import set_seed
from project4_nanogpt.data import CharTokenizer
from project4_nanogpt.models import GPTConfig, CharGPT

from .lora import LoRAConfig, inject_lora


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from a LoRA-fine-tuned character GPT")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer = CharTokenizer(checkpoint["characters"])
    model = CharGPT(GPTConfig(**checkpoint["model_config"])).to(device)
    inject_lora(model, LoRAConfig(**checkpoint["lora_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated = model.generate(prompt, args.max_new_tokens, args.temperature, args.top_k)
    text = tokenizer.decode(generated[0].tolist())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"saved={args.output}")
    print(text)


if __name__ == "__main__":
    main()
