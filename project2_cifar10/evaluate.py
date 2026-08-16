"""Evaluate a saved checkpoint once on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from .data import make_test_loader
from .engine import evaluate, resolve_device
from .models import build_model
from .utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", choices=["cifar10", "synthetic"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-test", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    if checkpoint.get("selection_metric") != "validation_accuracy":
        raise ValueError("checkpoint was not selected by validation accuracy")

    config = checkpoint["config"]
    model_name = config["model"]
    dataset_name = args.dataset or config["dataset"]
    test_loader = make_test_loader(
        dataset_name=dataset_name,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(config.get("seed", 42)),
        limit_test=args.limit_test,
        pin_memory=device.type == "cuda",
    )
    model = build_model(model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(
        f"model={model_name} selected_epoch={checkpoint['epoch']} "
        "selected_by=validation_accuracy "
        f"test_loss={metrics['loss']:.4f} "
        f"test_accuracy={metrics['accuracy']:.2%} "
        f"test_samples={len(test_loader.dataset)}"
    )


if __name__ == "__main__":
    main()
