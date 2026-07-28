"""Render loss/accuracy curves from the CSV history produced by train.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MNIST training history")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.history.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"history is empty: {args.history}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    test_loss = [float(row["test_loss"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    test_accuracy = [float(row["test_accuracy"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, test_loss, marker="o", label="test")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross entropy")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="train")
    axes[1].plot(epochs, test_accuracy, marker="o", label="test")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
