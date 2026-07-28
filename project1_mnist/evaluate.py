"""Evaluate a saved checkpoint on MNIST or synthetic test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from .data import make_dataloaders
from .engine import evaluate, resolve_device
from .models import build_model
from .utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MNIST checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", choices=["mnist", "synthetic"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-test", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    model_name = config["model"]
    dataset_name = args.dataset or config["dataset"]

    _, test_loader = make_dataloaders(
        dataset_name=dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=int(config.get("seed", 42)),
        limit_train=1 if dataset_name == "synthetic" else None,
        limit_test=args.limit_test,
    )
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(
        f"model={model_name} epoch={checkpoint['epoch']} "
        f"test_loss={metrics['loss']:.4f} "
        f"test_accuracy={metrics['accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
