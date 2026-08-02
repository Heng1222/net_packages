from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from experiments.disentangled_cvae_uwf_zeekdata24.download import download_dataset
from experiments.disentangled_cvae_uwf_zeekdata24.run_experiment import main
from experiments.disentangled_cvae_uwf_zeekdata24.tests.helpers import csv_bytes, uwf_row


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CATEGORIES = (
    "Benign",
    "Credential_Access",
    "Defense_Evasion",
    "Exfiltration",
    "Initial_Access",
    "Persistence",
    "Privilege_Escalation",
    "Reconnaissance",
)


def synthetic_category_rows() -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {category: [] for category in CATEGORIES}
    for index in range(15):
        rows["Benign"].append(uwf_row(f"benign-{index}", index=index))
        rows["Credential_Access"].append(
            uwf_row(f"t1110-{index}", "Credential Access", "T1110 - Brute Force", index)
        )
        rows["Reconnaissance"].append(
            uwf_row(f"t1595-{index}", "Reconnaissance", "T1595", index)
        )
        rows["Exfiltration"].append(
            uwf_row(f"t1048-{index}", "Exfiltration", "T1048", index)
        )
        rows["Initial_Access"].append(
            uwf_row(f"t1190-{index}", "Initial Access", "T1190", index)
        )
        rows["Initial_Access"].append(
            uwf_row(f"t1078-{index}", "Initial Access", "T1078", index)
        )
        for category, tactic in (
            ("Defense_Evasion", "Defense Evasion"),
            ("Persistence", "Persistence"),
            ("Privilege_Escalation", "Privilege Escalation"),
        ):
            rows[category].append(uwf_row(f"t1078-{index}", tactic, "T1078", index))
    return rows


class IntegrationTests(unittest.TestCase):
    def test_mock_download_prepare_two_epoch_train_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw_dir = root / "raw"
            prepared_dir = root / "prepared"
            run_dir = root / "run"
            config_path = root / "config.yaml"
            category_rows = synthetic_category_rows()
            data_config = {
                "raw_dir": str(raw_dir),
                "prepared_dir": str(prepared_dir),
                "source_base_url": "https://fixture.invalid/csv",
                "source_categories": list(CATEGORIES),
                "class_caps": {},
                "split": {
                    "strategy": "stratified_technique",
                    "train_ratio": 0.7,
                    "val_ratio": 0.15,
                    "test_ratio": 0.15,
                },
                "embedder": {"backend": "hashing", "output_dim": 768, "normalize": True},
            }

            def fetch(url: str) -> bytes:
                category = next(category for category in CATEGORIES if f"/{category}/" in url)
                if url.endswith("/"):
                    return b'<a href="part-fixture.csv">fixture</a>'
                return csv_bytes(category_rows[category])

            manifest = download_dataset(data_config, fetch_bytes=fetch)
            self.assertTrue(manifest.is_file())
            config = {
                "seed": 7,
                "data": data_config,
                "conditions": {
                    "path": str(
                        PROJECT_ROOT
                        / "experiments/disentangled_cvae_step1/conditions/mitre_attack_v11_3_step1.yaml"
                    ),
                    "format": "yaml",
                    "label_field": "label",
                    "text_fields": ["keywords", "techniques"],
                    "embedder_backend": "hashing",
                    "output_dim": 768,
                    "normalize": True,
                    "geometry": {
                        "method": "common_component_removal",
                        "center": True,
                        "remove_top_components": 0,
                        "normalize": True,
                        "strength": 1.0,
                        "append_common_condition": True,
                        "common_label": "Common Tactic Component",
                    },
                    "exclude_labels": ["Normal (TA9000)"],
                },
                "preprocessing": {"normalization": "standard", "batch_size": 128},
                "model": {
                    "input_dim": 768,
                    "residual_dim": 8,
                    "condition_dim": 768,
                    "encoder_hidden_dims": [24],
                    "decoder_hidden_dims": [24],
                    "behavior_projector_hidden_dims": [16],
                    "dropout": 0.0,
                    "batch_norm": False,
                    "activation": "relu",
                    "observation_variance": 1.0,
                    "temperature": 0.1,
                    "behavior_temperature": 0.1,
                    "residual_adversary_strength": 1.0,
                    "utility_margin": 0.1,
                    "residual_margin": 0.1,
                    "weights": {
                        "reconstruction": 1.0,
                        "kl": 0.01,
                        "decorrelation": 0.0,
                        "sparse": 0.001,
                        "gate_entropy": 0.01,
                        "utility": 0.0,
                        "residual_constraint": 0.1,
                        "behavior_infonce": 1.0,
                        "residual_adversary": 0.1,
                    },
                },
                "training": {
                    "batch_size": 32,
                    "max_epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "early_stopping_patience": 2,
                    "max_pos_weight": 50.0,
                    "num_workers": 0,
                    "device": "cpu",
                },
                "evaluation": {
                    "random_state": 7,
                    "threshold_grid": [0.25, 0.5, 0.75],
                    "probe_c_grid": [0.1, 1.0],
                    "shuffle_repeats": 5,
                    "bootstrap_repeats": 20,
                    "run_visualization": False,
                    "visualization_backend": "pca",
                    "visualization_max_samples": 100,
                },
                "output": {"base_dir": str(root / "outputs")},
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            with patch.object(sys, "argv", ["run_experiment.py", "--config", str(config_path), "--stage", "prepare"]):
                main()
            with patch.object(
                sys,
                "argv",
                [
                    "run_experiment.py", "--config", str(config_path), "--stage", "train",
                    "--run-dir", str(run_dir),
                ],
            ):
                main()

            self.assertTrue((run_dir / "checkpoints/disentangled_cvae.pt").is_file())
            self.assertTrue((run_dir / "reports/report.md").is_file())
            self.assertTrue((run_dir / "metrics/source_manifest.json").is_file())
            self.assertTrue((run_dir / "metrics/reconstruction_metrics.json").is_file())
            self.assertEqual(len(pd.read_csv(run_dir / "metrics/condition_gate_summary.csv")), 14)
            with np.load(run_dir / "embeddings/condition_embeddings.npz") as condition_archive:
                matrix = condition_archive["matrix"]
                raw_matrix = condition_archive["raw_matrix"]
                self.assertEqual(matrix.shape, (14, 768))
                np.testing.assert_allclose(matrix[-1], raw_matrix[:-1].mean(axis=0), atol=1e-6)
                self.assertNotAlmostEqual(float(np.linalg.norm(matrix[-1])), 1.0, places=4)
            predictions = pd.read_csv(run_dir / "metrics/testset_predictions.csv")
            self.assertIn("condition_prob__Common Tactic Component", predictions)
            probe_metrics = json.loads((run_dir / "metrics/probe_metrics.json").read_text(encoding="utf-8"))
            self.assertTrue({"x", "gates", "c", "h", "hc"}.issubset(probe_metrics))
            with np.load(prepared_dir / "split.npz") as split:
                self.assertEqual(sum(len(split[name]) for name in ("train", "val", "test")), 90)


if __name__ == "__main__":
    unittest.main()
