from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.ucd_cvae_v2_1.data import SplitIndices
from experiments.ucd_cvae_v2_1.model import UCDCVAE
from experiments.ucd_cvae_v2_1.tests.helpers import small_basis
from experiments.ucd_cvae_v2_1.training import (
    gradnorm_reconstruction_scale, schedule_state, train_model,
)


class TrainingTests(unittest.TestCase):
    def test_schedule_boundaries(self) -> None:
        config = {"phase1_end": 5, "phase2_end": 15}
        self.assertEqual(schedule_state(5, config).phase, 1)
        self.assertEqual(schedule_state(6, config).multiplier, 0.0)
        self.assertEqual(schedule_state(15, config).multiplier, 1.0)
        self.assertTrue(schedule_state(16, config).gradnorm_enabled)

    def test_gradient_cap_respects_both_ratios(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(4))
        reconstruction = (100.0 * parameter).sum(); sparse = parameter.sum(); align = (2.0 * parameter).sum()
        scale, diagnostics = gradnorm_reconstruction_scale(reconstruction, sparse, align, [parameter], 10.0)
        self.assertLess(scale, 1.0); self.assertLessEqual(diagnostics["ratio_rec_sparse"], 10.0 + 1e-6)
        self.assertLessEqual(diagnostics["ratio_rec_align"], 10.0 + 1e-6)

    def test_tiny_three_phase_training_writes_checkpoints(self) -> None:
        rng = np.random.default_rng(9); x = rng.normal(size=(36, 32)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        split = SplitIndices(np.arange(24), np.arange(24, 30), np.arange(30, 36)); common, tactics = small_basis()
        model = UCDCVAE(common, tactics, input_dim=32, gate_hidden_dims=[20], residual_hidden_dims=[20],
                        residual_up_hidden_dims=[20], decoder_hidden_dims=[20], concept_projector_hidden_dim=16, dropout=0)
        training = {"batch_size": 6, "max_epochs": 3, "learning_rate": 1e-3, "weight_decay": 0.0,
                    "early_stopping_patience": 2, "num_workers": 0, "phase1_end": 1, "phase2_end": 2,
                    "gradnorm_max_ratio": 10.0, "gradnorm_epsilon": 1e-12}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "checkpoints").mkdir(); (root / "metrics").mkdir()
            checkpoint = root / "checkpoints" / "model.pt"
            result = train_model(model, x, split, training,
                                 {"reconstruction": 1.0, "kl": 0.01, "sparse": 0.001, "align": 1.0},
                                 torch.device("cpu"), checkpoint, 42)
            self.assertTrue(checkpoint.is_file()); self.assertTrue(checkpoint.with_name("model_phase1.pt").is_file())
            self.assertTrue(checkpoint.with_name("model_phase2.pt").is_file()); self.assertEqual(result.best_epoch, 3)
            gradient_history = root / "metrics" / "gradient_history_full_orthogonal.csv"
            self.assertTrue(gradient_history.is_file())
            self.assertIn("ratio_rec_sparse", gradient_history.read_text(encoding="utf-8"))


if __name__ == "__main__": unittest.main()
