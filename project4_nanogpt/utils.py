"""Reproducibility, CSV, and checkpoint helpers for W5."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import torch

from project2_cifar10.utils import capture_rng_state, restore_rng_state, set_seed


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_history(path: str | Path, row: dict) -> None:
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def history_last_step(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("history is empty")
    return int(rows[-1]["step"])


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    step: int,
    best_val_loss: float,
    experiment_config: dict,
    model_config: dict,
    characters: list[str],
    corpus_hash: str,
) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "selection_metric": "validation_loss",
            "experiment_config": experiment_config,
            "model_config": model_config,
            "characters": characters,
            "corpus_sha256": corpus_hash,
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(Path(path), map_location=device, weights_only=True)


__all__ = [
    "append_history",
    "history_last_step",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "save_json",
    "set_seed",
    "text_sha256",
]
