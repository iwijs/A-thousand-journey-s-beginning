"""Render train/validation loss and accuracy curves from CSV history."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(history_path: str | Path, output_path: str | Path) -> Path:
    history_path = Path(history_path)
    output_path = Path(output_path)
    with history_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"history is empty: {history_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_accuracy = [float(row["train_accuracy"]) for row in rows]
    val_accuracy = [float(row["val_accuracy"]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="validation")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross entropy")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracy, marker="o", label="train")
    axes[1].plot(epochs, val_accuracy, marker="o", label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CIFAR-10 training history")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("curves.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_history(args.history, args.output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
