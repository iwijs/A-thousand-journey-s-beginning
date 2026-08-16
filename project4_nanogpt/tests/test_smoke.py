import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from project4_nanogpt.data import CharTokenizer, encode_and_split, get_batch
from project4_nanogpt.models import GPTConfig, CharGPT


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class NanoGPTSmokeTests(unittest.TestCase):
    def test_tokenizer_round_trip_and_batches(self):
        text = ("abc cab\n" * 30)
        train, _, tokenizer = encode_and_split(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode("cab")), "cab")
        inputs, targets = get_batch(train, 4, 8, torch.device("cpu"))
        self.assertEqual(tuple(inputs.shape), (4, 8))
        self.assertTrue(torch.equal(inputs[:, 1:], targets[:, :-1]))

    def test_forward_shape_and_loss(self):
        config = GPTConfig(vocab_size=7, block_size=12, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
        model = CharGPT(config)
        tokens = torch.randint(0, 7, (3, 12))
        logits, loss = model(tokens, tokens)
        self.assertEqual(tuple(logits.shape), (3, 12, 7))
        self.assertTrue(torch.isfinite(loss))

    def test_causal_mask_blocks_future_information(self):
        torch.manual_seed(1)
        config = GPTConfig(vocab_size=9, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
        model = CharGPT(config).eval()
        first = torch.randint(0, 9, (1, 8))
        second = first.clone()
        second[:, 5:] = (second[:, 5:] + 1) % 9
        logits_a, _ = model(first)
        logits_b, _ = model(second)
        self.assertTrue(torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6))

    def test_offline_train_resume_and_generate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            corpus = temporary / "corpus.txt"
            corpus.write_text(("First Citizen:\nWe learn small models.\n\n" * 30), encoding="utf-8")
            output = temporary / "run"
            base = [
                sys.executable, "-m", "project4_nanogpt.train",
                "--data", str(corpus), "--output-dir", str(output),
                "--max-steps", "4", "--eval-interval", "2", "--eval-iters", "1",
                "--log-interval", "2", "--batch-size", "4", "--block-size", "16",
                "--n-layer", "1", "--n-head", "2", "--n-embd", "16",
                "--dropout", "0", "--device", "cpu",
            ]
            completed = subprocess.run(base, cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in ("config.json", "history.csv", "best.pt", "last.pt", "curves.png"):
                self.assertTrue((output / name).is_file(), name)
            resumed_command = base.copy()
            resumed_command[resumed_command.index("4")] = "6"
            resumed_command.extend(["--resume", str(output / "last.pt")])
            resumed = subprocess.run(resumed_command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            with (output / "history.csv").open(encoding="utf-8", newline="") as file:
                self.assertEqual([int(row["step"]) for row in csv.DictReader(file)], [0, 2, 4, 6])
            generated = subprocess.run(
                [sys.executable, "-m", "project4_nanogpt.generate", str(output / "best.pt"), "--prompt", "F", "--max-new-tokens", "8", "--device", "cpu"],
                cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)


if __name__ == "__main__":
    unittest.main()
