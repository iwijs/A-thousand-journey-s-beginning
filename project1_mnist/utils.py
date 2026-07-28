"""Reproducibility, history, and checkpoint helpers."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    epoch: int,
    best_accuracy: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )


def append_history(path: str | Path, row: dict[str, float | int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
