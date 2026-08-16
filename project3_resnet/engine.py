"""Reuse device-safe train/evaluate loops established in W3."""

from project2_cifar10.engine import evaluate, resolve_device, train_one_epoch

__all__ = ["evaluate", "resolve_device", "train_one_epoch"]
