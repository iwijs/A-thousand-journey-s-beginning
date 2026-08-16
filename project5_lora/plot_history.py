"""Plot LoRA fine-tuning loss from CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from project4_nanogpt.plot_history import plot_history as _plot_history


def plot_history(history_path: str | Path, output_path: str | Path) -> Path:
    return _plot_history(
        history_path,
        output_path,
        title="LoRA domain fine-tuning loss",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a W6 LoRA history CSV")
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("curves.png"))
    args = parser.parse_args()
    print(f"saved={plot_history(args.history, args.output)}")


__all__ = ["plot_history"]

if __name__ == "__main__":
    main()
