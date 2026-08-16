import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from project2_cifar10.data import (
    SyntheticCifar10,
    build_transforms,
    make_test_loader,
    make_train_val_loaders,
    split_train_validation_indices,
)
from project2_cifar10.engine import evaluate, train_one_epoch
from project2_cifar10.models import build_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class Cifar10ProjectSmokeTests(unittest.TestCase):
    def test_model_output_shape(self):
        images = torch.randn(4, 3, 32, 32)
        self.assertEqual(tuple(build_model("cnn")(images).shape), (4, 10))

    def test_train_validation_split_is_deterministic_and_disjoint(self):
        train_a, val_a = split_train_validation_indices(100, 20, seed=7)
        train_b, val_b = split_train_validation_indices(100, 20, seed=7)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a), 80)
        self.assertEqual(len(val_a), 20)
        self.assertTrue(set(train_a).isdisjoint(val_a))
        self.assertEqual(set(train_a) | set(val_a), set(range(100)))

    def test_augmentation_is_train_only(self):
        train_transform, evaluation_transform = build_transforms("basic")
        self.assertTrue(
            any(isinstance(item, transforms.RandomCrop) for item in train_transform.transforms)
        )
        self.assertTrue(
            any(
                isinstance(item, transforms.RandomHorizontalFlip)
                for item in train_transform.transforms
            )
        )
        self.assertFalse(
            any(
                isinstance(item, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
                for item in evaluation_transform.transforms
            )
        )
        no_augmentation, _ = build_transforms("none")
        self.assertFalse(
            any(
                isinstance(item, (transforms.RandomCrop, transforms.RandomHorizontalFlip))
                for item in no_augmentation.transforms
            )
        )
        strong, _ = build_transforms("strong")
        self.assertTrue(
            any(isinstance(item, transforms.AutoAugment) for item in strong.transforms)
        )
        self.assertTrue(
            any(isinstance(item, transforms.RandomErasing) for item in strong.transforms)
        )

    def test_synthetic_dataset_and_one_epoch(self):
        dataset = SyntheticCifar10(size=23, seed=7)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.int64)

        train_loader, validation_loader = make_train_val_loaders(
            dataset_name="synthetic",
            data_dir="unused",
            batch_size=10,
            seed=5,
            limit_train=40,
            limit_val=20,
        )
        torch.manual_seed(5)
        model = build_model("cnn")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, torch.device("cpu")
        )
        val_metrics = evaluate(
            model, validation_loader, loss_fn, torch.device("cpu")
        )
        self.assertTrue(
            any(
                not torch.allclose(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )
        self.assertGreater(train_metrics["loss"], 0.0)
        self.assertGreaterEqual(val_metrics["accuracy"], 0.0)
        self.assertLessEqual(val_metrics["accuracy"], 1.0)

    def test_cifar10_loader_branch_without_network(self):
        class FakeCifar10(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.size = 100 if train else 30
                self.transform = transform

            def __len__(self):
                return self.size

            def __getitem__(self, index):
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                return self.transform(Image.fromarray(image)), index % 10

        with patch("project2_cifar10.data.datasets.CIFAR10", FakeCifar10):
            train_loader, validation_loader = make_train_val_loaders(
                dataset_name="cifar10",
                data_dir="unused",
                batch_size=8,
                validation_size=20,
                seed=9,
                limit_train=30,
                limit_val=10,
            )
            test_loader = make_test_loader(
                dataset_name="cifar10",
                data_dir="unused",
                batch_size=8,
                limit_test=12,
            )

        self.assertEqual(len(train_loader.dataset), 30)
        self.assertEqual(len(validation_loader.dataset), 10)
        self.assertEqual(len(test_loader.dataset), 12)
        self.assertTrue(
            set(train_loader.dataset.indices).isdisjoint(
                validation_loader.dataset.indices
            )
        )
        train_images, train_labels = next(iter(train_loader))
        validation_images, validation_labels = next(iter(validation_loader))
        self.assertEqual(tuple(train_images.shape), (8, 3, 32, 32))
        self.assertEqual(tuple(validation_images.shape), (8, 3, 32, 32))
        self.assertEqual(train_labels.dtype, torch.int64)
        self.assertEqual(validation_labels.dtype, torch.int64)

    def test_offline_cli_creates_validation_selected_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "runs"
            command = [
                sys.executable,
                "-m",
                "project2_cifar10.train",
                "--dataset",
                "synthetic",
                "--epochs",
                "2",
                "--batch-size",
                "20",
                "--limit-train",
                "80",
                "--limit-val",
                "40",
                "--device",
                "cpu",
                "--output-dir",
                str(output_root),
            ]
            completed = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            run_dir = output_root / "cnn"
            for name in ("history.csv", "config.json", "best.pt", "last.pt", "curves.png"):
                self.assertTrue((run_dir / name).is_file(), name)
            self.assertGreater((run_dir / "curves.png").stat().st_size, 0)

            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertIn("val_accuracy", rows[0])
            self.assertNotIn("test_accuracy", rows[0])

            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["selection_metric"], "validation_accuracy")
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            max_val_accuracy = max(float(row["val_accuracy"]) for row in rows)
            first_best_epoch = next(
                int(row["epoch"])
                for row in rows
                if float(row["val_accuracy"]) == max_val_accuracy
            )
            self.assertEqual(best["selection_metric"], "validation_accuracy")
            self.assertAlmostEqual(best["best_val_accuracy"], max_val_accuracy)
            self.assertEqual(best["epoch"], first_best_epoch)
            self.assertEqual(last["epoch"], 2)
            self.assertAlmostEqual(last["best_val_accuracy"], max_val_accuracy)
            self.assertIn("optimizer_state", last)
            self.assertIn("scheduler_state", last)
            self.assertIn("rng_state", last)
            self.assertIn("train_loader_generator_state", last)

            resumed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "project2_cifar10.train",
                    "--dataset",
                    "synthetic",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "20",
                    "--limit-train",
                    "80",
                    "--limit-val",
                    "40",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_root),
                    "--resume",
                    str(run_dir / "last.pt"),
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("completed_epoch=2", resumed.stdout)
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
                resumed_rows = list(csv.DictReader(file))
            self.assertEqual([int(row["epoch"]) for row in resumed_rows], [1, 2, 3])
            resumed_last = torch.load(
                run_dir / "last.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(resumed_last["epoch"], 3)

            evaluation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "project2_cifar10.evaluate",
                    str(run_dir / "best.pt"),
                    "--dataset",
                    "synthetic",
                    "--limit-test",
                    "30",
                    "--device",
                    "cpu",
                ],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
            self.assertIn("selected_by=validation_accuracy", evaluation.stdout)
            self.assertIn("test_accuracy=", evaluation.stdout)


if __name__ == "__main__":
    unittest.main()
