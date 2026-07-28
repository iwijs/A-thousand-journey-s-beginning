"""Real MNIST and deterministic synthetic smoke-test data."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


class SyntheticDigits(Dataset):
    """Easy 28x28 patterns used to test the pipeline without downloading data.

    This is not a substitute for MNIST. Each class owns a bright block at a
    class-specific location, so a tiny model can learn it quickly.
    """

    def __init__(self, size: int, seed: int = 0):
        if size <= 0:
            raise ValueError("size must be positive")

        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(size, dtype=torch.int64) % 10
        permutation = torch.randperm(size, generator=generator)
        self.labels = labels[permutation]
        self.images = torch.zeros(size, 1, 28, 28)

        for index, label_tensor in enumerate(self.labels):
            label = int(label_tensor)
            row = 4 + (label // 5) * 14
            column = 2 + (label % 5) * 5
            self.images[index, 0, row : row + 5, column : column + 4] = 1.0

        noise = 0.08 * torch.randn(
            self.images.shape,
            generator=generator,
            dtype=self.images.dtype,
        )
        self.images = (self.images + noise).clamp(0.0, 1.0)
        self.images = (self.images - MNIST_MEAN[0]) / MNIST_STD[0]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("dataset limit must be positive")
    return Subset(dataset, range(min(limit, len(dataset))))


def make_dataloaders(
    dataset_name: str,
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/test loaders for real MNIST or local synthetic patterns."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(MNIST_MEAN, MNIST_STD),
            ]
        )
        data_dir = Path(data_dir)
        train_dataset: Dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset: Dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "synthetic":
        train_dataset = SyntheticDigits(limit_train or 1000, seed=seed)
        test_dataset = SyntheticDigits(limit_test or 300, seed=seed + 1)
        # Limits were already applied while constructing the synthetic datasets.
        limit_train = None
        limit_test = None
    else:
        raise ValueError(
            f"unknown dataset {dataset_name!r}; choose 'mnist' or 'synthetic'"
        )

    train_dataset = _limit_dataset(train_dataset, limit_train)
    test_dataset = _limit_dataset(test_dataset, limit_test)
    loader_generator = torch.Generator().manual_seed(seed)

    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_options,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )
    return train_loader, test_loader
