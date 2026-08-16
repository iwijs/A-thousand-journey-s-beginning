"""Train a CIFAR ResNet-18 or its plain-network ablation."""

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
    parser = argparse.ArgumentParser(description="Reproduce ResNet-18 on CIFAR-10")
    parser.add_argument("--model", choices=["resnet18", "plain18"], default="resnet18")
    parser.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet_cifar10"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augmentation", choices=AUGMENTATION_CHOICES, default="basic")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.base_channels <= 0:
        raise ValueError("epochs, batch_size, and base_channels must be positive")
    if args.learning_rate <= 0 or not 0 <= args.momentum < 1:
        raise ValueError("learning_rate must be positive and momentum must be in [0, 1)")
    if args.weight_decay < 0 or args.validation_size <= 0:
        raise ValueError("weight_decay must be non-negative and validation_size positive")
    if args.resume is not None and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")


def _config(args: argparse.Namespace, device: torch.device) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["use_batch_norm"] = not args.no_batch_norm
    config["resolved_device"] = str(device)
    config["selection_metric"] = "validation_accuracy"
    return config


def _prepare_run(run_dir: Path, overwrite: bool, resume: bool) -> None:
    generated = [run_dir / name for name in ("config.json", "history.csv", "best.pt", "last.pt", "curves.png")]
    existing = [path for path in generated if path.exists()]
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        return
    if existing and not overwrite:
        raise FileExistsError(f"{run_dir} already contains a run")
    if overwrite:
        for path in existing:
            path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    run_dir = args.resume.parent if args.resume else args.output_dir / args.model
    _prepare_run(run_dir, args.overwrite, args.resume is not None)
    config = _config(args, device)

    train_loader, val_loader = make_train_val_loaders(
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
    model = build_model(
        args.model,
        base_channels=args.base_channels,
        use_batch_norm=not args.no_batch_norm,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_path = run_dir / "history.csv"
    start_epoch = 1
    best_val_accuracy = -1.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        previous = checkpoint["config"]
        compatibility_keys = (
            "model",
            "dataset",
            "augmentation",
            "seed",
            "base_channels",
            "use_batch_norm",
            "scheduler",
            "batch_size",
            "learning_rate",
            "momentum",
            "weight_decay",
            "validation_size",
            "num_workers",
            "limit_train",
            "limit_val",
            "deterministic",
        )
        for key in compatibility_keys:
            if previous.get(key) != config.get(key):
                raise ValueError(f"resume mismatch for {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        completed_epoch = int(checkpoint["epoch"])
        if history_last_epoch(history_path) != completed_epoch:
            raise ValueError("history/checkpoint epoch mismatch")
        start_epoch = completed_epoch + 1
        if start_epoch > args.epochs:
            raise ValueError("--epochs must exceed the checkpoint epoch")
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        train_loader.generator.set_state(checkpoint["train_loader_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"resumed_from={args.resume} completed_epoch={completed_epoch}")
        config["output_dir"] = previous["output_dir"]
        config["resumed_from_epoch"] = completed_epoch

    save_config(run_dir / "config.json", config)
    print(
        f"device={device} model={args.model} batch_norm={not args.no_batch_norm} "
        f"train={len(train_loader.dataset)} validation={len(val_loader.dataset)} test=not_loaded"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        seconds = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": learning_rate,
            "epoch_seconds": seconds,
        }
        append_history(history_path, row)
        improved = val_metrics["accuracy"] > best_val_accuracy
        best_val_accuracy = max(best_val_accuracy, val_metrics["accuracy"])
        if scheduler is not None:
            scheduler.step()
        state_args = (
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_accuracy,
            config,
            capture_rng_state(),
            train_loader.generator.get_state(),
        )
        save_checkpoint(run_dir / "last.pt", *state_args)
        if improved:
            save_checkpoint(run_dir / "best.pt", *state_args)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.2%} val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.2%} lr={learning_rate:.6g} time={seconds:.1f}s"
        )

    plot_history(history_path, run_dir / "curves.png")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"done: best_val_accuracy={best_val_accuracy:.2%} parameters={parameters:,} artifacts={run_dir}")


if __name__ == "__main__":
    main()
