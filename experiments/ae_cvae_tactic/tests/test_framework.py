from __future__ import annotations

import tempfile
import unittest
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.ae_cvae_tactic.data.adapters import load_raw_data, payload_to_text
from experiments.ae_cvae_tactic.data.condition_loader import load_condition_set
from experiments.ae_cvae_tactic.data.dataset import DatasetBundle
from experiments.ae_cvae_tactic.data.embedders import SentenceTransformerEmbedder
from experiments.ae_cvae_tactic.data.preprocessing import fit_transform_splits
from experiments.ae_cvae_tactic.data.split import make_split
from experiments.ae_cvae_tactic.evaluation.compatibility_test import compatibility_scores
from experiments.ae_cvae_tactic.models.ae import AutoEncoder
from experiments.ae_cvae_tactic.models.cvae import ConditionalVAE
from experiments.ae_cvae_tactic.pipeline import ExperimentRunner
from experiments.ae_cvae_tactic.training.common import load_best_state, save_checkpoint
from experiments.ae_cvae_tactic.utils.io import make_run_dir
from experiments.ae_cvae_tactic.utils.logging import configure_logging


class DummyEmbedder:
    model_name = "dummy"
    model_revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[len(text), index + 1, sum(map(ord, text)) % 17] for index, text in enumerate(texts)], dtype=np.float32)


class FakeTokenizer:
    def __call__(self, texts: list[str], **_: object) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0, *range(len(text.split())), 1] for text in texts]}


class FakeSentenceModel:
    tokenizer = FakeTokenizer()


class ModelTests(unittest.TestCase):
    def test_ae_shapes_and_loss(self) -> None:
        model = AutoEncoder(8, 3, [6, 4], batch_norm=False)
        x = torch.randn(5, 8)
        output = model(x)
        self.assertEqual(output["latent"].shape, (5, 3))
        self.assertEqual(output["x_recon"].shape, (5, 8))
        self.assertTrue(torch.isfinite(model.loss(output, x)["loss"]))

    def test_cvae_shapes_deterministic_mu_and_loss(self) -> None:
        model = ConditionalVAE(8, 4, 3, [7, 5], batch_norm=False)
        x, c = torch.randn(5, 8), torch.randn(5, 4)
        output = model(x, c, sample=False)
        self.assertTrue(torch.equal(output["z"], output["mu"]))
        self.assertEqual(output["logvar"].shape, (5, 3))
        losses = model.loss(output, x)
        expected_recon_nll = 0.5 * (
            (output["x_recon"] - x).pow(2) + float(np.log(2.0 * np.pi))
        ).sum(dim=1).mean()
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.allclose(losses["recon_nll"], expected_recon_nll))
        self.assertTrue(torch.allclose(losses["loss"], losses["negative_elbo"]))
        self.assertTrue(torch.allclose(losses["elbo"], -losses["negative_elbo"]))
        self.assertTrue(torch.allclose(losses["negative_elbo"], losses["recon_nll"] + losses["kl_loss"]))

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model = AutoEncoder(4, 2, [3])
            optimizer = torch.optim.Adam(model.parameters())
            save_checkpoint(path, model, optimizer, {"input_dim": 4}, 1, 0.5)
            expected = {key: value.clone() for key, value in model.state_dict().items()}
            for parameter in model.parameters():
                parameter.data.zero_()
            load_best_state(model, path, torch.device("cpu"))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))


class DataTests(unittest.TestCase):
    def test_payload_parser(self) -> None:
        self.assertEqual(payload_to_text("['one', 'two']", "python_literal_list"), "one\n[PACKET]\ntwo")

    def test_long_context_overflow_is_not_silent(self) -> None:
        embedder = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
        embedder.model = FakeSentenceModel()
        embedder.max_length = 5
        embedder.overflow_strategy = "error"
        embedder.max_observed_tokens = 0
        with self.assertRaisesRegex(ValueError, "exceed max_length=5"):
            embedder._validate_lengths(["one two three four"])
        embedder.overflow_strategy = "truncate"
        embedder._validate_lengths(["one two three four"])
        self.assertEqual(embedder.max_observed_tokens, 6)

    def test_csv_embedding_column_and_npy_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "input.csv"
            pd.DataFrame({"id": ["a", "b"], "label": ["x", "y"], "vector": ["[1,2]", "[3,4]"]}).to_csv(csv_path, index=False)
            loaded = load_raw_data({
                "input_path": str(csv_path), "input_format": "csv", "sample_id_col": "id", "label_col": "label",
                "condition_key_col": "label", "embedding_col": "vector", "embedding_prefix": None,
                "payload_text_col": None, "metadata_cols": []
            })
            self.assertEqual(loaded.features.shape, (2, 2))
            npy_path = root / "x.npy"
            np.save(npy_path, np.ones((2, 3), dtype=np.float32))
            meta_path = root / "meta.csv"
            pd.DataFrame({"id": ["a", "b"], "label": ["x", "y"]}).to_csv(meta_path, index=False)
            array = load_raw_data({
                "input_path": str(npy_path), "input_format": "npy", "metadata_path": str(meta_path),
                "sample_id_col": "id", "label_col": "label", "condition_key_col": "label", "metadata_cols": []
            })
            self.assertEqual(array.features.shape, (2, 3))

    def test_stratified_rare_class_and_scaler(self) -> None:
        x = np.arange(40, dtype=np.float32).reshape(10, 4)
        labels = np.asarray(["rare", "rare"] + ["common"] * 8)
        bundle = DatasetBundle(x, np.arange(10).astype(str), labels, labels)
        split = make_split(bundle, {"strategy": "stratified", "train_ratio": .7, "val_ratio": .15, "test_ratio": .15}, 42)
        self.assertEqual(int(np.sum(labels[split.train] == "rare")), 1)
        self.assertEqual(int(np.sum(labels[split.val] == "rare")), 0)
        self.assertEqual(int(np.sum(labels[split.test] == "rare")), 1)
        with tempfile.TemporaryDirectory() as folder:
            transformed, scaler = fit_transform_splits(x, split, "standard", str(Path(folder) / "scaler.pkl"))
            self.assertTrue(np.allclose(transformed[split.train].mean(axis=0), 0, atol=1e-6))
            self.assertTrue(np.allclose(scaler.mean_, x[split.train].mean(axis=0)))

    def test_conditions_and_missing_label(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "conditions.yaml"
            data = {"tactics": {
                "A (TA0001)": {"label": "A (TA0001)", "tactic_id": "TA0001", "description_full": "alpha"},
                "B (TA0002)": {"label": "B (TA0002)", "tactic_id": "TA0002", "description_full": "beta"},
                "C (TA0003)": {"label": "C (TA0003)", "tactic_id": "TA0003", "description_full": "gamma"},
            }}
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            config = {
                "path": str(path), "format": "yaml", "label_field": "label", "id_field": "tactic_id",
                "text_field_full": "description_full", "wrong_base_mode": "full", "random_dim": 3
            }
            conditions = load_condition_set(config, list(data["tactics"]), "wrong", 42, DummyEmbedder())
            self.assertEqual(conditions.matrix.shape, (3, 3))
            self.assertTrue(all(label != donor for label, donor in conditions.mapping.items()))
            with self.assertRaisesRegex(KeyError, "TA9999"):
                load_condition_set(config, ["Missing (TA9999)"], "full", 42, DummyEmbedder())


class EvaluationTests(unittest.TestCase):
    def test_compatibility_matrix_shape(self) -> None:
        model = ConditionalVAE(5, 3, 2, [4], batch_norm=False)
        x = np.random.default_rng(1).normal(size=(7, 5)).astype(np.float32)
        c = np.random.default_rng(2).normal(size=(4, 3)).astype(np.float32)
        scores = compatibility_scores(model, x, c, torch.device("cpu"), 3)
        self.assertEqual(scores.shape, (7, 4))
        self.assertTrue(np.isfinite(scores).all())


class IntegrationTests(unittest.TestCase):
    def test_synthetic_all_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            labels = np.repeat(np.asarray(["A", "B", "C"]), 12)
            x = rng.normal(size=(36, 8)).astype(np.float32) + np.repeat(np.arange(3), 12)[:, None]
            np.save(root / "x.npy", x)
            pd.DataFrame({"id": np.arange(36).astype(str), "label": labels}).to_csv(root / "meta.csv", index=False)
            run_dir = make_run_dir(root / "outputs")
            configure_logging(run_dir / "logs" / "test.log")
            config = {
                "seed": 42,
                "data": {
                    "input_path": str(root / "x.npy"), "input_format": "npy", "metadata_path": str(root / "meta.csv"),
                    "array_key": "x", "sample_id_col": "id", "label_col": "label", "condition_key_col": "label",
                    "metadata_cols": [], "exclude_from_cvae": [], "split": {
                        "strategy": "stratified", "train_ratio": .7, "val_ratio": .15, "test_ratio": .15, "time_col": None
                    }
                },
                "conditions": {"path": None, "format": None, "condition_mode": "none", "random_dim": 5},
                "preprocessing": {"normalization": "standard"},
                "model": {
                    "ae": {"input_dim": None, "latent_dim": 4, "hidden_dims": [12, 8], "dropout": 0.0,
                           "batch_norm": False, "activation": "relu", "reconstruction_loss": "mse"},
                    "cvae": {"input_dim": None, "condition_dim": None, "latent_dim": 4, "hidden_dims": [12, 8],
                             "dropout": 0.0, "batch_norm": False, "activation": "relu", "reconstruction_loss": "mse",
                             "objective": "elbo", "likelihood": "gaussian", "observation_variance": 1.0,
                             "latent_representation": "mu"}
                },
                "training": {"batch_size": 8, "max_epochs": 2, "learning_rate": .001, "weight_decay": 0.0,
                             "early_stopping_patience": 2, "num_workers": 0, "device": "cpu"},
                "classifier": {"type": "logistic_regression", "class_weight": "balanced", "mlp_hidden_dims": [8],
                               "max_iter": 100, "random_state": 42},
                "evaluation": {"run_classification": True, "run_clustering": True, "run_visualization": True,
                               "run_compatibility_test": True, "visualization_methods": ["pca"],
                               "visualization_max_samples": 100, "compatibility_score": "reconstruction"},
                "ablation": {"modes": ["none"]},
                "output": {"base_dir": str(root / "outputs"), "latent_format": "npz"},
                "_meta": {"project_root": str(root), "config_path": "synthetic"},
            }
            runner = ExperimentRunner(config, run_dir, torch.device("cpu"))
            runner.run_ae()
            runner.run_cvae()
            runner.run_compatibility()
            runner.run_ablation()
            runner.finalize_report()
            self.assertTrue((run_dir / "checkpoints" / "ae.pt").is_file())
            self.assertTrue((run_dir / "checkpoints" / "cvae.pt").is_file())
            self.assertTrue((run_dir / "metrics" / "compatibility_score_matrix.npz").is_file())
            self.assertTrue((run_dir / "metrics" / "condition_ablation_summary.csv").is_file())
            self.assertTrue((run_dir / "reports" / "report.md").is_file())
            logging.shutdown()


if __name__ == "__main__":
    unittest.main()
