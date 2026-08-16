"""Command-line entry point for validation-driven CIFAR-10 training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .data import AUGMENTATION_CHOICES, make_train_val_loaders
from .engine import evaluate, resolve_device, train_one_epoch
from .models import build_model
from .plot_history import plot_history
from .utils import (
    append_history,
    capture_rng_state,
    history_last_epoch,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline CNN on CIFAR-10")
    parser.add_argument("--model", choices=["cnn"], default="cnn")
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "synthetic"],
        default="cifar10",
        help="synthetic is only for offline smoke tests",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cifar10"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation",
        choices=AUGMENTATION_CHOICES,
        default="basic",
        help="none, basic (crop+flip), or strong (basic+AutoAugment+erasing)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="prefer reproducible kernels; exact CPU/GPU equality is not guaranteed",
    )
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "none"],
        default="cosine",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace generated artifacts in an existing run directory",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume the run stored in this checkpoint; --epochs is the total target",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if args.validation_size <= 0:
        raise ValueError("validation_size must be positive")
    if args.resume is not None and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")


def _serializable_config(args: argparse.Namespace, device: torch.device) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["resolved_device"] = str(device)
    config["selection_metric"] = "validation_accuracy"
    return config


def _prepare_run_directory(run_dir: Path, overwrite: bool, resume: bool) -> None:
    generated_names = (
        "history.csv",
        "config.json",
        "best.pt",
        "last.pt",
        "curves.png",
    )
    existing = [run_dir / name for name in generated_names if (run_dir / name).exists()]
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        return
    if existing and not overwrite:
        raise FileExistsError(
            f"{run_dir} already contains a run; choose another --output-dir "
            "or pass --overwrite"
        )
    if overwrite:
        for path in existing:
            path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    run_dir = args.resume.parent if args.resume is not None else args.output_dir / args.model
    _prepare_run_directory(run_dir, args.overwrite, resume=args.resume is not None)

    config = _serializable_config(args, device)
    print(
        f"device={device} model={args.model} dataset={args.dataset} "
        "selection_metric=validation_accuracy"
    )

    train_loader, validation_loader = make_train_val_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_size=args.validation_size,
        num_workers=args.num_workers,
        seed=args.seed,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        augmentation=args.augmentation,
        pin_memory=device.type == "cuda",
    )
    print(
        f"split_sizes: train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)}; test=not_loaded"
    )
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    history_path = run_dir / "history.csv"
    best_val_accuracy = -1.0
    start_epoch = 1

    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        previous_config = checkpoint["config"]
        compatibility_keys = (
            "model",
            "dataset",
            "augmentation",
            "seed",
            "scheduler",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "validation_size",
            "num_workers",
            "limit_train",
            "limit_val",
            "deterministic",
        )
        for key in compatibility_keys:
            if previous_config.get(key) != config.get(key):
                raise ValueError(
                    f"resume mismatch for {key}: checkpoint={previous_config.get(key)!r}, "
                    f"current={config.get(key)!r}"
                )
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            if checkpoint["scheduler_state"] is None:
                raise ValueError("checkpoint has no scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        checkpoint_epoch = int(checkpoint["epoch"])
        if history_last_epoch(history_path) != checkpoint_epoch:
            raise ValueError("history last epoch does not match resume checkpoint")
        start_epoch = checkpoint_epoch + 1
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        train_loader.generator.set_state(checkpoint["train_loader_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        if start_epoch > args.epochs:
            raise ValueError(
                f"checkpoint already completed epoch {checkpoint_epoch}; "
                f"--epochs must be at least {start_epoch}"
            )
        print(f"resumed_from={args.resume} completed_epoch={checkpoint_epoch}")
        config["output_dir"] = previous_config["output_dir"]
        config["resumed_from_epoch"] = checkpoint_epoch

    save_config(run_dir / "config.json", config)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_metrics = evaluate(
            model, validation_loader, loss_fn, device
        )
        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": learning_rate,
            "epoch_seconds": epoch_seconds,
        }
        append_history(history_path, row)

        improved = val_metrics["accuracy"] > best_val_accuracy
        best_val_accuracy = max(best_val_accuracy, val_metrics["accuracy"])
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_accuracy,
            config,
            capture_rng_state(),
            train_loader.generator.get_state(),
        )
        if improved:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_accuracy,
                config,
                capture_rng_state(),
                train_loader.generator.get_state(),
            )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2%} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.2%} "
            f"lr={learning_rate:.6g} time={epoch_seconds:.1f}s"
        )

    curves_path = plot_history(history_path, run_dir / "curves.png")
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"done: best_val_accuracy={best_val_accuracy:.2%}, "
        f"parameters={parameter_count:,}, artifacts={run_dir}, "
        f"curves={curves_path}"
    )


if __name__ == "__main__":
    main()
