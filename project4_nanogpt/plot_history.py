"""Plot training and validation cross-entropy from W5 CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_history(
    history_path: str | Path,
    output_path: str | Path,
    title: str = "Character GPT loss",
) -> Path:
    history_path = Path(history_path)
    output_path = Path(output_path)
    with history_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("history is empty")
    steps = [int(row["step"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(steps, train_loss, marker="o", label="train")
    axis.plot(steps, val_loss, marker="o", label="validation")
    axis.set(title=title, xlabel="optimizer steps", ylabel="cross entropy")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a W5 history CSV")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("curves.png"))
    args = parser.parse_args()
    print(f"saved={plot_history(args.history, args.output)}")


if __name__ == "__main__":
    main()
