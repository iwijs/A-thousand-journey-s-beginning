"""Evaluate a validation-selected W4 checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from .data import make_test_loader
from .engine import evaluate, resolve_device
from .models import build_model
from .utils import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR ResNet checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", choices=["cifar10", "synthetic"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-test", type=int)
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    loader = make_test_loader(
        dataset_name=args.dataset or config["dataset"],
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=int(config["seed"]),
        limit_test=args.limit_test,
        pin_memory=device.type == "cuda",
    )
    model = build_model(
        config["model"],
        base_channels=int(config["base_channels"]),
        use_batch_norm=bool(config["use_batch_norm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    print(
        f"model={config['model']} selected_epoch={checkpoint['epoch']} "
        f"selected_by=validation_accuracy test_loss={metrics['loss']:.4f} "
        f"test_accuracy={metrics['accuracy']:.2%} test_samples={len(loader.dataset)}"
    )


if __name__ == "__main__":
    main()
