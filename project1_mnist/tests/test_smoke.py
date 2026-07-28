import unittest

import torch
from torch import nn

from project1_mnist.data import SyntheticDigits, make_dataloaders
from project1_mnist.engine import evaluate, train_one_epoch
from project1_mnist.models import build_model


class MnistProjectSmokeTests(unittest.TestCase):
    def test_model_output_shapes(self):
        images = torch.randn(4, 1, 28, 28)
        for name in ("mlp", "cnn"):
            with self.subTest(model=name):
                self.assertEqual(tuple(build_model(name)(images).shape), (4, 10))

    def test_synthetic_dataset_contract(self):
        dataset = SyntheticDigits(size=23, seed=7)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.int64)
        self.assertEqual(len(dataset), 23)

    def test_one_epoch_updates_and_evaluates(self):
        train_loader, test_loader = make_dataloaders(
            dataset_name="synthetic",
            data_dir="unused",
            batch_size=32,
            seed=5,
            limit_train=120,
            limit_test=40,
        )
        torch.manual_seed(5)
        model = build_model("mlp")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = [parameter.detach().clone() for parameter in model.parameters()]

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, torch.device("cpu")
        )
        test_metrics = evaluate(
            model, test_loader, loss_fn, torch.device("cpu")
        )

        self.assertTrue(
            any(
                not torch.allclose(old, new.detach())
                for old, new in zip(before, model.parameters())
            )
        )
        self.assertGreater(train_metrics["loss"], 0.0)
        self.assertGreaterEqual(test_metrics["accuracy"], 0.0)
        self.assertLessEqual(test_metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
