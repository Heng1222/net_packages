from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class IntegrationTests(unittest.TestCase):
    def test_smoke_prepare_train_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            labels = [
                "{'Reconnaissance (TA0043)'}",
                "{'Execution (TA0002)'}",
                "{'Normal (TA9000)'}",
            ]
            for index in range(120):
                rows.append(
                    {
                        "Session_ID": f"s{index}",
                        "Datetime": f"2022-10-{1 + index // 12:02d} 00:{index % 60:02d}:00",
                        "clean_payload_list": f"GET /payload/{index % 9} HTTP/VERSION[SEP]Host: IP",
                        "Sess_Tactic_predict": labels[index % len(labels)],
                        "Src_ISP": "cht",
                        "Protocol": "http",
                    }
                )
            csv_path = root / "step1.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            golden_rows = [
                {
                    "Session_ID": f"s{index}",
                    "Tactic": "Reconnaissance (TA0043)" if index % 3 == 0 else "Execution (TA0002)",
                }
                for index in range(0, 90)
                if index % 3 in {0, 1}
            ]
            golden_path = root / "step2_golden_review_2_with_Tactic.csv"
            pd.DataFrame(golden_rows).to_csv(golden_path, index=False)
            config = {
                "seed": 42,
                "data": {
                    "input_path": str(csv_path),
                    "prepared_dir": str(root / "prepared"),
                    "sample_id_col": "Session_ID",
                    "payload_text_col": "clean_payload_list",
                    "payload_parser": "auto",
                    "label_col": "Sess_Tactic_predict",
                    "time_col": "Datetime",
                    "metadata_cols": ["Src_ISP", "Protocol"],
                    "read_chunksize": 25,
                    "max_rows": 120,
                    "label_multi_policy": "error",
                    "skip_empty_labels": True,
                    "normal_label": "Normal (TA9000)",
                    "split": {"strategy": "time", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
                    "embedder": {"backend": "hashing", "output_dim": 768, "normalize": True},
                },
                "conditions": {
                    "path": str(Path("experiments/disentangled_cvae_step1/conditions/mitre_attack_v11_3_step1.yaml").resolve()),
                    "format": "yaml",
                    "label_field": "label",
                    "id_field": "tactic_id",
                    "text_fields": ["keywords", "techniques"],
                    "embedder_backend": "hashing",
                    "output_dim": 768,
                    "normalize": True,
                    "exclude_labels": ["Normal (TA9000)"],
                },
                "supervision": {
                    "enabled": True,
                    "path": str(golden_path),
                    "sample_id_col": "Session_ID",
                    "label_col": "Tactic",
                },
                "preprocessing": {"normalization": "standard", "batch_size": 50},
                "model": {
                    "input_dim": 768,
                    "residual_dim": 8,
                    "condition_dim": 768,
                    "encoder_hidden_dims": [32],
                    "decoder_hidden_dims": [32],
                    "behavior_projector_hidden_dims": [16],
                    "dropout": 0.0,
                    "batch_norm": False,
                    "activation": "relu",
                    "observation_variance": 1.0,
                    "temperature": 0.2,
                    "behavior_temperature": 0.2,
                    "residual_adversary_strength": 1.0,
                    "utility_margin": 0.1,
                    "residual_margin": 0.1,
                    "weights": {
                        "reconstruction": 1.0,
                        "kl": 1.0,
                        "decorrelation": 0.1,
                        "sparse": 0.001,
                        "gate_entropy": 0.01,
                        "utility": 0.1,
                        "residual_constraint": 0.1,
                        "behavior_infonce": 0.5,
                        "residual_adversary": 0.1,
                    },
                },
                "training": {
                    "batch_size": 16,
                    "max_epochs": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "early_stopping_patience": 2,
                    "num_workers": 0,
                    "device": "cpu",
                },
                "evaluation": {
                    "random_state": 42,
                    "condition_threshold": 0.5,
                    "run_visualization": True,
                    "visualization_backend": "pca",
                    "visualization_max_samples": 60,
                    "umap_n_neighbors": 5,
                    "umap_min_dist": 0.1,
                },
                "output": {"base_dir": str(root / "outputs")},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "experiments/disentangled_cvae_step1/run_experiment.py",
                    "--config",
                    str(config_path),
                    "--stage",
                    "all",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            run_dir = Path(result.stdout.strip().splitlines()[-1])
            self.assertTrue((run_dir / "checkpoints" / "disentangled_cvae.pt").is_file())
            self.assertFalse((run_dir / "metrics" / "probe_metrics.json").exists())
            self.assertTrue((run_dir / "metrics" / "behavior_supervision_summary.json").is_file())
            self.assertTrue((run_dir / "metrics" / "behavior_alignment_metrics.json").is_file())
            self.assertTrue((run_dir / "metrics" / "condition_gate_summary.csv").is_file())
            self.assertTrue((run_dir / "metrics" / "condition_ablation_delta_mse_summary.csv").is_file())
            self.assertTrue((run_dir / "metrics" / "testset_condition_predictions.csv").is_file())
            self.assertTrue((run_dir / "metrics" / "testset_subset_100.csv").is_file())
            self.assertFalse((run_dir / "metrics" / "component_activation_summary.csv").exists())
            self.assertFalse((run_dir / "metrics" / "component_activation_by_tactic.csv").exists())
            self.assertTrue((run_dir / "plots" / "condition_cosine_similarity.png").is_file())
            self.assertTrue((run_dir / "plots" / "training_reconstruction_losses.png").is_file())
            self.assertTrue((run_dir / "plots" / "umap_original_space.png").is_file())
            self.assertTrue((run_dir / "plots" / "umap_h_space.png").is_file())
            self.assertTrue((run_dir / "plots" / "umap_gated_c_space.png").is_file())
            self.assertTrue((run_dir / "reports" / "report.md").is_file())
            log_text = (run_dir / "logs" / "experiment.log").read_text(encoding="utf-8")
            self.assertIn("Training started", log_text)
            self.assertIn("Epoch 1/2", log_text)
            self.assertIn("train_loss=", log_text)
            self.assertIn("val_loss=", log_text)

            history = pd.read_csv(run_dir / "metrics" / "training_history.csv")
            self.assertIn("val_h_only_mse", history.columns)
            self.assertIn("val_c_only_mse", history.columns)
            self.assertIn("val_gate_entropy_loss", history.columns)
            self.assertIn("val_behavior_infonce_loss", history.columns)
            self.assertIn("val_residual_adversary_loss", history.columns)
            self.assertGreater(history["train_behavior_labeled_count"].max(), 0)
            predictions = pd.read_csv(run_dir / "metrics" / "testset_condition_predictions.csv")
            self.assertIn("gold_tactic", predictions.columns)
            prob_cols = [column for column in predictions.columns if column.startswith("condition_prob__")]
            self.assertGreater(len(prob_cols), 0)
            self.assertTrue(((predictions[prob_cols] >= 0.0) & (predictions[prob_cols] <= 1.0)).all().all())
            self.assertIn("active_condition_count", predictions.columns)
            self.assertIn("predicted_conditions", predictions.columns)
            allowed = {column.replace("condition_prob__", "", 1) for column in prob_cols}
            allowed.add("ambiguous")
            self.assertTrue(set(predictions["predicted_condition"]).issubset(allowed))
            subset = pd.read_csv(run_dir / "metrics" / "testset_subset_100.csv")
            self.assertLessEqual(int(subset.groupby("predicted_condition").size().max()), 100)


if __name__ == "__main__":
    unittest.main()
