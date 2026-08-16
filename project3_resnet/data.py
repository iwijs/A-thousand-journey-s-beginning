"""Reuse the validated CIFAR-10 protocol from the W3 engineering project."""

from project2_cifar10.data import (
    AUGMENTATION_CHOICES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    SyntheticCifar10,
    build_transforms,
    make_test_loader,
    make_train_val_loaders,
    split_train_validation_indices,
)

__all__ = [
    "AUGMENTATION_CHOICES",
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "SyntheticCifar10",
    "build_transforms",
    "make_test_loader",
    "make_train_val_loaders",
    "split_train_validation_indices",
]
