"""CIFAR-10 data splits, augmentation, and offline synthetic data."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
AUGMENTATION_CHOICES = ("none", "basic", "strong")


def build_transforms(augmentation: str = "basic"):
    """Build a configurable train transform and a deterministic eval transform."""

    augmentation = augmentation.lower()
    if augmentation not in AUGMENTATION_CHOICES:
        raise ValueError(
            f"unknown augmentation {augmentation!r}; choose one of "
            f"{', '.join(AUGMENTATION_CHOICES)}"
        )

    train_steps: list = []
    if augmentation in {"basic", "strong"}:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if augmentation == "strong":
        train_steps.append(
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if augmentation == "strong":
        train_steps.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), value="random")
        )

    train_transform = transforms.Compose(train_steps)
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, evaluation_transform


class SyntheticCifar10(Dataset):
    """Easy 3x32x32 patterns for offline pipeline tests.

    This dataset is deliberately small and learnable, but it is not a proxy for
    real CIFAR-10 accuracy. Independent seeds are used for train, validation,
    and test instances.
    """

    def __init__(self, size: int, seed: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")

        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size, dtype=torch.int64) % 10
        self.labels = labels[torch.randperm(size, generator=generator)]
        images = torch.zeros(size, 3, 32, 32)

        for index, label_tensor in enumerate(self.labels):
            label = int(label_tensor)
            channel = label % 3
            row = 2 + (label // 5) * 16
            column = 2 + (label % 5) * 6
            images[index, channel, row : row + 10, column : column + 5] = 1.0
            images[index, (channel + 1) % 3, row + 3 : row + 7, :] = 0.45

        noise = 0.04 * torch.randn(images.shape, generator=generator)
        images = (images + noise).clamp(0.0, 1.0)
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
        self.images = (images - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def split_train_validation_indices(
    dataset_size: int,
    validation_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic, disjoint split of the official training set."""

    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than one")
    if not 0 < validation_size < dataset_size:
        raise ValueError("validation_size must be between 1 and dataset_size - 1")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator).tolist()
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]
    return train_indices, validation_indices


def _validate_limit(name: str, limit: int | None) -> None:
    if limit is not None and limit <= 0:
        raise ValueError(f"{name} must be positive")


def _limit_indices(indices: list[int], limit: int | None) -> list[int]:
    if limit is None:
        return indices
    return indices[: min(limit, len(indices))]


def _loader_options(batch_size: int, num_workers: int) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }


def make_train_val_loaders(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int,
    validation_size: int = 5000,
    num_workers: int = 0,
    seed: int = 42,
    limit_train: int | None = None,
    limit_val: int | None = None,
    augmentation: str = "basic",
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train/validation loaders without touching the test split."""

    options = _loader_options(batch_size, num_workers)
    options["pin_memory"] = pin_memory
    _validate_limit("limit_train", limit_train)
    _validate_limit("limit_val", limit_val)
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        train_transform, evaluation_transform = build_transforms(augmentation)
        data_dir = Path(data_dir)
        augmented_dataset: Dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        evaluation_dataset: Dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=evaluation_transform,
        )
        train_indices, validation_indices = split_train_validation_indices(
            len(augmented_dataset), validation_size, seed
        )
        train_dataset: Dataset = Subset(
            augmented_dataset,
            _limit_indices(train_indices, limit_train),
        )
        validation_dataset: Dataset = Subset(
            evaluation_dataset,
            _limit_indices(validation_indices, limit_val),
        )
    elif dataset_name == "synthetic":
        train_dataset = SyntheticCifar10(limit_train or 200, seed=seed)
        validation_dataset = SyntheticCifar10(limit_val or 80, seed=seed + 1)
    else:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; choose 'cifar10' or 'synthetic'"
        )

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **options,
    )
    return train_loader, validation_loader


def make_test_loader(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    limit_test: int | None = None,
    pin_memory: bool = False,
) -> DataLoader:
    """Build the held-out test loader for the independent evaluation entry point."""

    options = _loader_options(batch_size, num_workers)
    options["pin_memory"] = pin_memory
    _validate_limit("limit_test", limit_test)
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        _, evaluation_transform = build_transforms("none")
        test_dataset: Dataset = datasets.CIFAR10(
            root=Path(data_dir),
            train=False,
            download=True,
            transform=evaluation_transform,
        )
        if limit_test is not None:
            test_dataset = Subset(
                test_dataset,
                range(min(limit_test, len(test_dataset))),
            )
    elif dataset_name == "synthetic":
        test_dataset = SyntheticCifar10(limit_test or 80, seed=seed + 2)
    else:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; choose 'cifar10' or 'synthetic'"
        )

    return DataLoader(test_dataset, shuffle=False, **options)
