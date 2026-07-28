"""Command-line entry point for reproducible MLP/CNN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .data import make_dataloaders
from .engine import evaluate, resolve_device, train_one_epoch
from .models import build_model
from .utils import append_history, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP or CNN on MNIST")
    parser.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    parser.add_argument(
        "--dataset",
        choices=["mnist", "synthetic"],
        default="mnist",
        help="synthetic is only for an offline smoke test",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mnist"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace artifacts in an existing model run directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device} model={args.model} dataset={args.dataset}")

    train_loader, test_loader = make_dataloaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_dir = args.output_dir / args.model
    history_path = run_dir / "history.csv"
    if history_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{run_dir} already contains a run; choose another "
                "--output-dir or pass --overwrite"
            )
        history_path.unlink()

    best_accuracy = -1.0
    config = vars(args).copy()
    config["data_dir"] = str(config["data_dir"])
    config["output_dir"] = str(config["output_dir"])

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        test_metrics = evaluate(model, test_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        append_history(history_path, row)

        improved = test_metrics["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, test_metrics["accuracy"])
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_accuracy,
            config,
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_accuracy,
                config,
            )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2%} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.2%}"
        )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"done: best_test_accuracy={best_accuracy:.2%}, "
        f"parameters={parameter_count:,}, artifacts={run_dir}"
    )


if __name__ == "__main__":
    main()
