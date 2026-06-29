from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.ae_cvae_tactic.contrastive_pipeline import ContrastiveExperimentRunner
from experiments.ae_cvae_tactic.models.contrastive_cvae import ContrastiveConditionalVAE
from experiments.ae_cvae_tactic.utils.io import make_run_dir
from experiments.ae_cvae_tactic.utils.logging import configure_logging


class ContrastiveModelTests(unittest.TestCase):
    def _model(self) -> ContrastiveConditionalVAE:
        return ContrastiveConditionalVAE(
            input_dim=8,
            condition_dim=4,
            latent_dim=3,
            hidden_dims=[7, 5],
            batch_norm=False,
            projection_dim=4,
            projection_hidden_dims=[6],
            temperature=0.2,
            contrastive_weight=3.0,
        )

    def test_payload_logits_do_not_receive_oracle_condition(self) -> None:
        model = self._model().eval()
        x = torch.randn(5, 8)
        candidates = torch.randn(3, 4)
        first = model(x, torch.randn(5, 4), candidates, sample=False)
        second = model(x, torch.randn(5, 4), candidates, sample=False)
        self.assertTrue(torch.equal(first["contrastive_logits"], second["contrastive_logits"]))
        self.assertEqual(first["payload_projection"].shape, (5, 4))
        self.assertTrue(
            torch.allclose(first["payload_projection"].norm(dim=1), torch.ones(5), atol=1e-6)
        )

    def test_total_loss_adds_weighted_contrastive_term(self) -> None:
        model = self._model()
        x = torch.randn(5, 8)
        c = torch.randn(5, 4)
        candidates = torch.randn(3, 4)
        targets = torch.tensor([0, 1, 2, 0, 1])
        losses = model.loss(model(x, c, candidates, sample=False), x, targets)
        expected = losses["negative_elbo"] + 3.0 * losses["contrastive_loss"]
        self.assertTrue(torch.allclose(losses["loss"], expected))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreaterEqual(float(losses["contrastive_accuracy"]), 0.0)
        self.assertLessEqual(float(losses["contrastive_accuracy"]), 1.0)


class ContrastiveIntegrationTests(unittest.TestCase):
    def test_opt_in_pipeline_writes_only_contrastive_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            labels = np.repeat(np.asarray(["A", "B", "C"]), 12)
            x = rng.normal(size=(36, 8)).astype(np.float32)
            x += np.repeat(np.arange(3), 12)[:, None]
            np.save(root / "x.npy", x)
            pd.DataFrame({"id": np.arange(36).astype(str), "label": labels}).to_csv(
                root / "meta.csv", index=False
            )
            run_dir = make_run_dir(root / "contrastive_outputs")
            configure_logging(run_dir / "logs" / "test.log")
            config = {
                "seed": 42,
                "data": {
                    "input_path": str(root / "x.npy"),
                    "input_format": "npy",
                    "metadata_path": str(root / "meta.csv"),
                    "array_key": "x",
                    "sample_id_col": "id",
                    "label_col": "label",
                    "condition_key_col": "label",
                    "metadata_cols": [],
                    "exclude_from_cvae": [],
                    "split": {
                        "strategy": "stratified",
                        "train_ratio": 0.7,
                        "val_ratio": 0.15,
                        "test_ratio": 0.15,
                        "time_col": None,
                    },
                },
                "conditions": {
                    "path": None,
                    "format": None,
                    "condition_mode": "random",
                    "random_dim": 5,
                },
                "preprocessing": {"normalization": "standard"},
                "model": {
                    "ae": {
                        "input_dim": None,
                        "latent_dim": 4,
                        "hidden_dims": [12, 8],
                        "dropout": 0.0,
                        "batch_norm": False,
                        "activation": "relu",
                        "reconstruction_loss": "mse",
                    },
                    "cvae": {
                        "input_dim": None,
                        "condition_dim": None,
                        "latent_dim": 4,
                        "hidden_dims": [12, 8],
                        "dropout": 0.0,
                        "batch_norm": False,
                        "activation": "relu",
                        "reconstruction_loss": "mse",
                        "objective": "elbo",
                        "likelihood": "gaussian",
                        "observation_variance": 1.0,
                        "latent_representation": "mu",
                    },
                    "contrastive_cvae": {
                        "input_dim": None,
                        "condition_dim": None,
                        "latent_dim": 4,
                        "hidden_dims": [12, 8],
                        "dropout": 0.0,
                        "batch_norm": False,
                        "activation": "relu",
                        "reconstruction_loss": "mse",
                        "objective": "elbo",
                        "likelihood": "gaussian",
                        "observation_variance": 1.0,
                        "latent_representation": "mu",
                        "projection_dim": None,
                        "projection_hidden_dims": [8],
                        "condition_projection": "identity",
                        "temperature": 0.2,
                        "contrastive_weight": 2.0,
                    },
                },
                "training": {
                    "batch_size": 8,
                    "max_epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "early_stopping_patience": 2,
                    "num_workers": 0,
                    "device": "cpu",
                },
                "classifier": {
                    "type": "logistic_regression",
                    "class_weight": "balanced",
                    "mlp_hidden_dims": [8],
                    "max_iter": 100,
                    "random_state": 42,
                },
                "evaluation": {
                    "run_classification": True,
                    "run_clustering": True,
                    "run_visualization": False,
                    "run_compatibility_test": True,
                    "visualization_methods": ["pca"],
                    "visualization_max_samples": 100,
                    "compatibility_score": "reconstruction",
                },
                "ablation": {"modes": ["random"]},
                "output": {"base_dir": str(root / "contrastive_outputs"), "latent_format": "npz"},
                "_meta": {"project_root": str(root), "config_path": "synthetic"},
            }
            runner = ContrastiveExperimentRunner(config, run_dir, torch.device("cpu"))
            runner.run_contrastive()
            runner.finalize_contrastive_report()
            self.assertTrue((run_dir / "checkpoints" / "contrastive_cvae.pt").is_file())
            self.assertFalse((run_dir / "checkpoints" / "cvae.pt").exists())
            self.assertFalse((run_dir / "checkpoints" / "ae.pt").exists())
            self.assertTrue((run_dir / "metrics" / "contrastive_score_matrix.npz").is_file())
            self.assertTrue((run_dir / "reports" / "contrastive_report.md").is_file())
            logging.shutdown()


if __name__ == "__main__":
    unittest.main()
