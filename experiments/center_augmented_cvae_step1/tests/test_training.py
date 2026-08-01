from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.center_augmented_cvae_step1.data import SplitIndices
from experiments.center_augmented_cvae_step1.embedders import normalize_rows
from experiments.center_augmented_cvae_step1.model import CenterAugmentedCVAE
from experiments.center_augmented_cvae_step1.training import train_model, unsupervised_loss


class TrainingTests(unittest.TestCase):
    def test_loss_contains_only_reconstruction_and_kl(self) -> None:
        target = torch.zeros(2, 4)
        output = {"x_recon": torch.ones(2, 4), "z_mu": torch.zeros(2, 2), "z_logvar": torch.zeros(2, 2)}
        losses = unsupervised_loss(output, target, {"reconstruction": 1.0, "kl": 0.1})
        self.assertEqual(set(losses), {"loss", "recon_mse", "kl_loss", "recon_cosine", "recon_mse_per_sample", "kl_per_sample"})
        self.assertAlmostEqual(float(losses["loss"]), 1.0)

    def test_tiny_training_writes_checkpoint(self) -> None:
        rng = np.random.default_rng(8); x = normalize_rows(rng.normal(size=(24, 8)).astype(np.float32))
        decode = rng.normal(size=(14, 8)).astype(np.float32); gate = normalize_rows(decode)
        split = SplitIndices(np.arange(16), np.arange(16, 20), np.arange(20, 24))
        model = CenterAugmentedCVAE(8, 3, 14, 8, [12], [12], gate_temperature=0.2)
        training = {"batch_size": 4, "max_epochs": 2, "learning_rate": 1e-3,
                    "weight_decay": 0.0, "early_stopping_patience": 2, "num_workers": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "checkpoints").mkdir(); (root / "metrics").mkdir()
            checkpoint = root / "checkpoints" / "tiny.pt"
            result = train_model(model, x, split, training, {"reconstruction": 1.0, "kl": 0.01},
                                 torch.device("cpu"), checkpoint, 42,
                                 decode_matrix=decode, gate_matrix=gate)
            self.assertTrue(checkpoint.is_file()); self.assertGreaterEqual(result.best_epoch, 1)


if __name__ == "__main__": unittest.main()
