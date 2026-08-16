"""Parameter-efficiently fine-tune the W5 character GPT with LoRA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import torch

from project2_cifar10.engine import resolve_device
from project2_cifar10.utils import capture_rng_state, restore_rng_state, set_seed
from project4_nanogpt.data import CharTokenizer, get_batch, load_corpus
from project4_nanogpt.models import GPTConfig, CharGPT
from project4_nanogpt.utils import text_sha256

from .lora import LoRAConfig, adapter_state_dict, inject_lora, parameter_counts
from .plot_history import plot_history


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune the W5 character GPT")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/romeo_juliet_speeches.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora_rank8"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target", choices=["attention", "all-linear"], default="attention")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def last_logged_step(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return int(rows[-1]["step"])


@torch.inference_mode()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    model.eval()
    results = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for index in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, block_size, device)
            _, loss = model(inputs, targets)
            losses[index] = loss.detach().cpu()
        results[split] = losses.mean().item()
    model.train()
    return results


def save_state(path, model, optimizer, step, best_val_loss, metadata):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    torch.save(
        {
            **metadata,
            "model_state": model.state_dict(),
            "adapter_state": adapter_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "selection_metric": "validation_loss",
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.eval_interval, args.eval_iters, args.log_interval, args.batch_size) <= 0:
        raise ValueError("step, interval, and batch arguments must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    base = torch.load(args.base_checkpoint, map_location=device, weights_only=True)
    tokenizer = CharTokenizer(base["characters"])
    domain_text = load_corpus(args.data)
    encoded = torch.tensor(tokenizer.encode(domain_text), dtype=torch.long)
    split = int(0.9 * len(encoded))
    train_data, val_data = encoded[:split], encoded[split:]
    model_config = GPTConfig(**base["model_config"])
    if len(val_data) <= model_config.block_size:
        raise ValueError("domain validation split must be longer than the base block_size")
    model = CharGPT(model_config).to(device)
    model.load_state_dict(base["model_state"])
    lora_config = LoRAConfig(args.rank, args.alpha, args.lora_dropout, args.target)
    replaced = inject_lora(model, lora_config)
    # Different ranks allocate differently sized A matrices and therefore consume
    # different amounts of RNG during adapter initialization. Reset only after
    # injection so rank ablations start data sampling/dropout from the same stream;
    # checkpoint resume restores the saved stream below and takes precedence.
    set_seed(args.seed, deterministic=args.deterministic)
    trainable, total = parameter_counts(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    run_dir = args.resume.parent if args.resume else args.output_dir
    generated = [run_dir / name for name in ("config.json", "history.csv", "best.pt", "last.pt", "adapter_best.pt", "curves.png")]
    existing = [path for path in generated if path.exists()]
    if args.resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    elif existing and not args.overwrite:
        raise FileExistsError(f"{run_dir} already contains a run")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for path in existing:
                path.unlink()
    metadata = {
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "model_config": model_config.to_dict(),
        "characters": tokenizer.characters,
        "domain_sha256": text_sha256(domain_text),
        "lora_config": lora_config.to_dict(),
        "replaced_modules": replaced,
        "training_config": {
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "deterministic": args.deterministic,
        },
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        **metadata,
        "resolved_device": str(device),
        "rng_reseed_after_injection": True,
        "trainable_parameters": trainable,
        "total_parameters_with_adapters": total,
        "trainable_fraction": trainable / total,
        "domain_characters": len(domain_text),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    history_path = run_dir / "history.csv"
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        for key in (
            "base_checkpoint_sha256",
            "domain_sha256",
            "lora_config",
            "model_config",
            "training_config",
        ):
            if checkpoint[key] != metadata[key]:
                raise ValueError(f"resume mismatch for {key}")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
        if last_logged_step(history_path) != start_step:
            raise ValueError("history/checkpoint step mismatch")
        if start_step >= args.max_steps:
            raise ValueError("--max-steps must exceed checkpoint step")
        best_val_loss = float(checkpoint["best_val_loss"])
        restore_rng_state(checkpoint["rng_state"])
        print(f"resumed_from={args.resume} completed_step={start_step}")
    print(
        f"device={device} rank={args.rank} target={args.target} modules={len(replaced)} "
        f"trainable={trainable:,}/{total:,} ({trainable / total:.2%})"
    )
    started = time.perf_counter()
    for step in range(start_step, args.max_steps + 1):
        already_logged = args.resume is not None and step == start_step
        if (step % args.eval_interval == 0 or step == args.max_steps) and not already_logged:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size, model_config.block_size, args.eval_iters, device
            )
            elapsed = time.perf_counter() - started
            append_history(
                history_path,
                {
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": elapsed,
                },
            )
            improved = losses["val"] < best_val_loss
            best_val_loss = min(best_val_loss, losses["val"])
            save_state(run_dir / "last.pt", model, optimizer, step, best_val_loss, metadata)
            if improved:
                save_state(run_dir / "best.pt", model, optimizer, step, best_val_loss, metadata)
                torch.save(
                    {**metadata, "adapter_state": adapter_state_dict(model), "step": step, "val_loss": losses["val"]},
                    run_dir / "adapter_best.pt",
                )
            print(f"step={step:05d} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} time={elapsed:.1f}s")
        if step == args.max_steps:
            break
        inputs, targets = get_batch(train_data, args.batch_size, model_config.block_size, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        if (step + 1) % args.log_interval == 0:
            print(f"train_step={step + 1:05d} batch_loss={loss.item():.4f}")
    plot_history(history_path, run_dir / "curves.png")
    print(f"done: best_val_loss={best_val_loss:.4f} artifacts={run_dir}")


if __name__ == "__main__":
    main()
