from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


class GoldenIntegrationTests(unittest.TestCase):
    def test_prepare_train_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            labels = [
                "Discovery (TA0007)",
                "Credential Access (TA0006)",
                "Execution (TA0002)",
                "Normal (TA9000)",
            ]
            rows = []
            for label_index, label in enumerate(labels):
                for index in range(15):
                    rows.append(
                        {
                            "Session_ID": f"s-{label_index}-{index}",
                            "clean_payload_list": f"payload {label} evidence {index}",
                            "Tactic": label,
                        }
                    )
            source = root / "golden.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            config = {
                "seed": 42,
                "data": {
                    "input_path": str(source),
                    "prepared_dir": str(root / "prepared"),
                    "sample_id_col": "Session_ID",
                    "payload_text_col": "clean_payload_list",
                    "label_col": "Tactic",
                    "normal_label": "Normal (TA9000)",
                    "min_class_count": 5,
                    "conflicting_payload_policy": "exclude",
                    "deduplicate_payloads": True,
                    "split": {"strategy": "stratified_group", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
                    "embedder": {"backend": "hashing", "output_dim": 16, "normalize": True},
                },
                "conditions": {
                    "path": str(Path("experiments/disentangled_cvae_step1/conditions/mitre_attack_v11_3_step1.yaml").resolve()),
                    "format": "yaml",
                    "label_field": "label",
                    "text_fields": ["keywords", "techniques"],
                    "embedder_backend": "hashing",
                    "output_dim": 16,
                    "normalize": True,
                    "geometry": {"method": "common_component_removal", "center": True, "remove_top_components": 0, "normalize": True, "strength": 1.0},
                    "exclude_labels": ["Normal (TA9000)"],
                },
                "preprocessing": {"normalization": "standard"},
                "model": {
                    "input_dim": 16,
                    "condition_dim": 16,
                    "residual_dim": 4,
                    "encoder_hidden_dims": [16],
                    "decoder_hidden_dims": [16],
                    "projector_hidden_dims": [16],
                    "classifier_hidden_dims": [16],
                    "dropout": 0.0,
                    "activation": "gelu",
                    "temperature": 0.2,
                },
                "loss": {"reconstruction": 1.0, "kl": 0.01, "gate_supervision": 1.0, "condition_use": 0.1, "condition_use_margin": 0.01},
                "training": {"batch_size": 16, "max_epochs": 2, "learning_rate": 0.001, "weight_decay": 0.0, "early_stopping_patience": 2, "device": "cpu"},
                "evaluation": {"condition_threshold": 0.5},
                "output": {"base_dir": str(root / "outputs")},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "experiments/golden_oracle_cvae_step2/run_experiment.py", "--config", str(config_path), "--stage", "all"],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            run_dir = Path(result.stdout.strip().splitlines()[-1])
            expected = [
                "checkpoints/oracle_cvae.pt",
                "checkpoints/predicted_gate_cvae.pt",
                "checkpoints/payload_classifier.pt",
                "metrics/training_history.csv",
                "metrics/model_comparison.json",
                "metrics/loss_summary.json",
                "metrics/behavior_alignment_metrics.json",
                "metrics/testset_condition_predictions.csv",
                "metrics/testset_subset_100.csv",
                "metrics/condition_gate_summary.csv",
                "metrics/condition_ablation_delta_mse_summary.csv",
                "plots/training_reconstruction_losses.png",
                "plots/confusion_matrices.png",
                "reports/report.md",
            ]
            for relative in expected:
                self.assertTrue((run_dir / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
