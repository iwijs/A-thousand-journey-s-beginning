import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from project4_nanogpt.models import GPTConfig, CharGPT
from project5_lora.lora import LoRAConfig, LoRALinear, inject_lora, merge_lora, parameter_counts
from project5_lora.prepare_domain import extract_speeches


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class LoRATests(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_output(self):
        torch.manual_seed(2)
        config = GPTConfig(vocab_size=8, block_size=12, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
        model = CharGPT(config).eval()
        inputs = torch.randint(0, 8, (2, 12))
        before, _ = model(inputs)
        replaced = inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=0, target="attention"))
        after, _ = model(inputs)
        self.assertEqual(len(replaced), 2)
        self.assertTrue(torch.equal(before, after))
        trainable, total = parameter_counts(model)
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)
        self.assertTrue(all(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" in name))

    def test_adapter_inherits_base_device_and_dtype(self):
        base = torch.nn.Linear(7, 5, device="meta", dtype=torch.float64)
        wrapped = LoRALinear(
            base,
            LoRAConfig(rank=2, alpha=4, dropout=0, target="attention"),
        )
        self.assertEqual(wrapped.lora_a.weight.device, base.weight.device)
        self.assertEqual(wrapped.lora_b.weight.device, base.weight.device)
        self.assertEqual(wrapped.lora_a.weight.dtype, base.weight.dtype)
        self.assertEqual(wrapped.lora_b.weight.dtype, base.weight.dtype)

    def test_merge_preserves_eval_output(self):
        torch.manual_seed(3)
        model = CharGPT(GPTConfig(vocab_size=8, block_size=8, n_layer=1, n_head=2, n_embd=16, dropout=0.0))
        inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=0, target="all-linear"))
        for module in model.modules():
            if isinstance(module, LoRALinear):
                torch.nn.init.normal_(module.lora_b.weight, std=0.02)
        model.eval()
        inputs = torch.randint(0, 8, (2, 8))
        before, _ = model(inputs)
        merged = merge_lora(model)
        after, _ = model(inputs)
        self.assertGreater(len(merged), 0)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_domain_extraction(self):
        text = "ROMEO:\nHello.\n\nOTHER:\nNo.\n\nJULIET:\nWorld.\n"
        result = extract_speeches(text, {"ROMEO", "JULIET"})
        self.assertIn("ROMEO", result)
        self.assertIn("JULIET", result)
        self.assertNotIn("OTHER", result)

    def test_offline_fine_tune_and_generate(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary = Path(temporary_directory)
            characters = sorted(set("ROMEO:\nWe learn tiny adapters. "))
            model_config = GPTConfig(vocab_size=len(characters), block_size=12, n_layer=1, n_head=2, n_embd=16, dropout=0.0)
            base_model = CharGPT(model_config)
            base_checkpoint = temporary / "base.pt"
            torch.save(
                {"model_state": base_model.state_dict(), "model_config": model_config.to_dict(), "characters": characters},
                base_checkpoint,
            )
            corpus = temporary / "domain.txt"
            corpus.write_text("ROMEO:\nWe learn tiny adapters.\n" * 30, encoding="utf-8")
            output = temporary / "lora"
            command = [
                sys.executable, "-m", "project5_lora.train",
                "--base-checkpoint", str(base_checkpoint), "--data", str(corpus), "--output-dir", str(output),
                "--max-steps", "4", "--eval-interval", "2", "--eval-iters", "1", "--log-interval", "2",
                "--batch-size", "4", "--rank", "2", "--alpha", "4", "--lora-dropout", "0", "--device", "cpu",
            ]
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            for name in ("config.json", "history.csv", "best.pt", "last.pt", "adapter_best.pt", "curves.png"):
                self.assertTrue((output / name).is_file(), name)
            with (output / "history.csv").open(encoding="utf-8", newline="") as file:
                rank2_rows = list(csv.DictReader(file))
            self.assertEqual([int(row["step"]) for row in rank2_rows], [0, 2, 4])

            # A different rank consumes a different number of random values while
            # initializing LoRA A. The training entry point must realign RNG after
            # injection so the frozen-base step-0 evaluation uses identical batches.
            rank4_output = temporary / "lora_rank4"
            rank4_command = command.copy()
            rank4_command[rank4_command.index(str(output))] = str(rank4_output)
            rank4_command[rank4_command.index("2", rank4_command.index("--rank"))] = "4"
            rank4_command[rank4_command.index("4", rank4_command.index("--alpha"))] = "8"
            rank4_completed = subprocess.run(
                rank4_command,
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(rank4_completed.returncode, 0, rank4_completed.stderr)
            with (rank4_output / "history.csv").open(encoding="utf-8", newline="") as file:
                rank4_rows = list(csv.DictReader(file))
            self.assertEqual(rank2_rows[0]["train_loss"], rank4_rows[0]["train_loss"])
            self.assertEqual(rank2_rows[0]["val_loss"], rank4_rows[0]["val_loss"])
            generated = subprocess.run(
                [sys.executable, "-m", "project5_lora.generate", str(output / "best.pt"), "--prompt", "ROMEO:\n", "--max-new-tokens", "4", "--device", "cpu"],
                cwd=REPOSITORY_ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)


if __name__ == "__main__":
    unittest.main()
