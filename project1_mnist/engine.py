"""Reusable training and evaluation loops."""

from __future__ import annotations

import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def train_one_epoch(model, dataloader, loss_fn, optimizer, device) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("training dataloader is empty")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }


def evaluate(model, dataloader, loss_fn, device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = labels.shape[0]
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("evaluation dataloader is empty")
    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
    }
