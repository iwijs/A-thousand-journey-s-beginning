import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from project3_resnet.models import BasicBlock, build_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ResNetSmokeTests(unittest.TestCase):
    def test_model_output_shapes(self):
        images = torch.randn(2, 3, 32, 32)
        for name in ("resnet18", "plain18"):
            with self.subTest(name=name):
                logits = build_model(name, base_channels=8)(images)
                self.assertEqual(tuple(logits.shape), (2, 10))

    def test_residual_block_preserves_gradient_path(self):
        block = BasicBlock(8, 8, use_residual=True)
        block.eval()
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
        inputs = torch.rand(2, 8, 6, 6, requires_grad=True)
        block(inputs).sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_projection_changes_shape(self):
        block = BasicBlock(8, 16, stride=2, use_residual=True)
        self.assertEqual(tuple(block(torch.randn(2, 8, 16, 16)).shape), (2, 16, 8, 8))

    def test_offline_cli_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            command = [
                sys.executable,
                "-m",
                "project3_resnet.train",
                "--dataset",
                "synthetic",
                "--epochs",
                "1",
                "--batch-size",
                "16",
                "--base-channels",
                "4",
                "--limit-train",
                "32",
                "--limit-val",
                "16",
                "--device",
                "cpu",
                "--output-dir",
                str(output),
            ]
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = output / "resnet18"
            for name in ("config.json", "history.csv", "best.pt", "last.pt", "curves.png"):
                self.assertTrue((run_dir / name).is_file(), name)
            with (run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 1)
            evaluated = subprocess.run(
                [sys.executable, "-m", "project3_resnet.evaluate", str(run_dir / "best.pt"), "--dataset", "synthetic", "--limit-test", "16", "--device", "cpu"],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertIn("test_accuracy=", evaluated.stdout)


if __name__ == "__main__":
    unittest.main()
