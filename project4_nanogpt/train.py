"""Train a character-level GPT with validation-selected checkpoints."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from project2_cifar10.engine import resolve_device

from .data import encode_and_split, get_batch, load_corpus
from .models import GPTConfig, CharGPT
from .plot_history import plot_history
from .utils import (
    append_history,
    history_last_step,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    save_json,
    set_seed,
    text_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny character GPT")
    parser.add_argument("--data", type=Path, default=Path("data/tinyshakespeare.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/nanogpt"))
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate(args: argparse.Namespace) -> None:
    positive = (args.max_steps, args.eval_interval, args.eval_iters, args.log_interval, args.batch_size, args.block_size)
    if any(value <= 0 for value in positive):
        raise ValueError("step, interval, batch, and block values must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer hyperparameters")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")


def _experiment_config(args: argparse.Namespace, device: torch.device) -> dict:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config["resolved_device"] = str(device)
    config["selection_metric"] = "validation_loss"
    return config


def _prepare_run(run_dir: Path, overwrite: bool, resume: bool) -> None:
    names = ("config.json", "history.csv", "best.pt", "last.pt", "curves.png")
    existing = [run_dir / name for name in names if (run_dir / name).exists()]
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


@torch.inference_mode()
def estimate_loss(model, train_data, val_data, args, device) -> dict[str, float]:
    model.eval()
    result = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(args.eval_iters)
        for index in range(args.eval_iters):
            inputs, targets = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(inputs, targets)
            losses[index] = loss.detach().cpu()
        result[name] = losses.mean().item()
    model.train()
    return result


def main() -> None:
    args = parse_args()
    _validate(args)
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    text = load_corpus(args.data)
    train_data, val_data, tokenizer = encode_and_split(text)
    if len(val_data) <= args.block_size:
        raise ValueError("validation split must be longer than block_size")
    corpus_hash = text_sha256(text)
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    run_dir = args.resume.parent if args.resume else args.output_dir
    _prepare_run(run_dir, args.overwrite, args.resume is not None)
    experiment_config = _experiment_config(args, device)
    model = CharGPT(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history_path = run_dir / "history.csv"
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        previous_experiment = checkpoint["experiment_config"]
        compatibility_keys = (
            "eval_interval",
            "eval_iters",
            "log_interval",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "seed",
            "deterministic",
        )
        for key in compatibility_keys:
            if previous_experiment.get(key) != experiment_config.get(key):
                raise ValueError(
                    f"resume mismatch for {key}: checkpoint={previous_experiment.get(key)!r}, "
                    f"current={experiment_config.get(key)!r}"
                )
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("resume model configuration mismatch")
        if checkpoint["characters"] != tokenizer.characters:
            raise ValueError("resume tokenizer mismatch")
        if checkpoint["corpus_sha256"] != corpus_hash:
            raise ValueError("resume corpus hash mismatch")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_step = int(checkpoint["step"])
        if history_last_step(history_path) != completed_step:
            raise ValueError("history/checkpoint step mismatch")
        start_step = completed_step
        best_val_loss = float(checkpoint["best_val_loss"])
        restore_rng_state(checkpoint["rng_state"])
        if start_step >= args.max_steps:
            raise ValueError("--max-steps must exceed checkpoint step")
        print(f"resumed_from={args.resume} completed_step={completed_step}")
        experiment_config["output_dir"] = previous_experiment["output_dir"]
        experiment_config["resumed_from_step"] = completed_step
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    save_json(
        run_dir / "config.json",
        {
            **experiment_config,
            "model_config": model_config.to_dict(),
            "characters": tokenizer.characters,
            "corpus_sha256": corpus_hash,
            "corpus_characters": len(text),
            "parameter_count": parameter_count,
        },
    )
    print(
        f"device={device} train_tokens={len(train_data):,} val_tokens={len(val_data):,} "
        f"vocab={tokenizer.vocab_size} parameters={parameter_count:,}"
    )
    started = time.perf_counter()
    for step in range(start_step, args.max_steps + 1):
        already_logged_resume_step = args.resume is not None and step == start_step
        if (step % args.eval_interval == 0 or step == args.max_steps) and not already_logged_resume_step:
            losses = estimate_loss(model, train_data, val_data, args, device)
            elapsed = time.perf_counter() - started
            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": elapsed,
            }
            append_history(history_path, row)
            improved = losses["val"] < best_val_loss
            best_val_loss = min(best_val_loss, losses["val"])
            state_args = (
                model,
                optimizer,
                step,
                best_val_loss,
                experiment_config,
                model_config.to_dict(),
                tokenizer.characters,
                corpus_hash,
            )
            save_checkpoint(run_dir / "last.pt", *state_args)
            if improved:
                save_checkpoint(run_dir / "best.pt", *state_args)
            print(f"step={step:05d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} time={elapsed:.1f}s")
        if step == args.max_steps:
            break
        inputs, targets = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (step + 1) % args.log_interval == 0:
            print(f"train_step={step + 1:05d} batch_loss={loss.item():.4f}")
    plot_history(history_path, run_dir / "curves.png")
    print(f"done: best_val_loss={best_val_loss:.4f} artifacts={run_dir}")


if __name__ == "__main__":
    main()
