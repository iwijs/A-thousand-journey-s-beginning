"""Character tokenization and random next-character training batches."""

from __future__ import annotations

from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, characters: list[str]):
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("characters must be a non-empty unique list")
        self.characters = list(characters)
        self.stoi = {character: index for index, character in enumerate(characters)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    def encode(self, text: str) -> list[int]:
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"prompt contains unknown characters: {unknown!r}")
        return [self.stoi[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.characters[token_id] for token_id in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.characters)


def load_corpus(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"corpus not found: {path}; run python -m project4_nanogpt.prepare_data"
        )
    text = path.read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("corpus must contain at least 100 characters")
    return text


def encode_and_split(text: str, train_fraction: float = 0.9):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(encoded) * train_fraction)
    return encoded[:split], encoded[split:], tokenizer


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 1 or len(data) <= block_size:
        raise ValueError("data must be 1D and longer than block_size")
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)
