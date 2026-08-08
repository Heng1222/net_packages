from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml


class IntegrationTests(unittest.TestCase):
    def test_package_has_no_old_experiment_imports(self) -> None:
        package = Path(__file__).resolve().parents[1]
        text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
        self.assertNotIn("experiments.disentangled_cvae_step1", text)
        self.assertNotIn("experiments.center_augmented_cvae_step1", text)

    def test_hashing_smoke_all_stages_and_artifacts(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "payloads.csv"; golden = root / "golden.csv"
            rows = [{"Session_ID": f"s{i}", "Datetime": f"2022-01-{1 + i // 5:02d} 00:00:00",
                     "clean_payload_list": f"GET /item/{i % 7} HTTP/1.1", "Sess_Tactic_predict": ""}
                    for i in range(60)]
            pd.DataFrame(rows).to_csv(source, index=False)
            pd.DataFrame({"Session_ID": [f"s{i}" for i in range(51, 60)],
                          "Tactic": ["Normal (TA9000)" if i % 2 else "Execution (TA0002)" for i in range(51, 60)]}).to_csv(golden, index=False)
            condition_path = project_root / "experiments" / "ucd_cvae_v2_1" / "conditions" / "mitre_attack_enterprise_v11_3.yaml"
            config = {
                "seed": 42,
                "data": {"input_path": str(source), "prepared_dir": str(root / "prepared"),
                    "sample_id_col": "Session_ID", "payload_text_col": "clean_payload_list", "time_col": "Datetime",
                    "metadata_cols": ["Sess_Tactic_predict"], "payload_parser": "auto", "read_chunksize": 13, "max_rows": None,
                    "split": {"strategy": "time", "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
                    "embedder": {"backend": "hashing", "output_dim": 32, "normalize": True}},
                "conditions": {"path": str(condition_path), "cache_dir": str(root / "condition_cache"),
                    "text_field": "description_full", "common_label": "Common Malicious Component",
                    "embedder": {"backend": "hashing", "output_dim": 32, "normalize": True}},
                "model": {"input_dim": 32, "residual_dim": 16, "gate_hidden_dims": [20],
                    "residual_hidden_dims": [20], "residual_up_hidden_dims": [20], "decoder_hidden_dims": [20],
                    "concept_projector_hidden_dim": 16, "activation": "gelu", "dropout": 0.0,
                    "geometry_variant": "full_orthogonal", "geometry_epsilon": 1e-6, "alignment_temperature": 0.1},
                "loss": {"reconstruction": 1.0, "kl": 0.01, "sparse": 0.001, "align": 1.0},
                "training": {"batch_size": 10, "max_epochs": 3, "learning_rate": 1e-3, "weight_decay": 0.0,
                    "early_stopping_patience": 2, "num_workers": 0, "device": "cpu", "phase1_end": 1,
                    "phase2_end": 2, "gradnorm_max_ratio": 10.0, "gradnorm_epsilon": 1e-12},
                "evaluation": {"golden_path": str(golden), "golden_sample_id_col": "Session_ID",
                    "golden_label_col": "Tactic", "normal_label": "Normal (TA9000)", "benign_threshold": 0.1,
                    "block_threshold": 0.5, "top_k": 3},
                "variants": ["full_orthogonal", "common_removal_only"], "output": {"base_dir": str(root / "outputs")}}
            config_path = root / "config.yaml"; config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            result = subprocess.run([sys.executable, "experiments/ucd_cvae_v2_1/run_experiment.py",
                                     "--config", str(config_path), "--stage", "all"], cwd=project_root,
                                    text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            run_dir = Path(result.stdout.strip().splitlines()[-1])
            for variant in ("full_orthogonal", "common_removal_only"):
                self.assertTrue((run_dir / "checkpoints" / f"ucd_cvae_{variant}.pt").is_file())
                self.assertTrue((run_dir / "checkpoints" / f"gate_only_{variant}.pt").is_file())
                self.assertTrue((run_dir / "metrics" / f"evaluation_{variant}.json").is_file())
            self.assertTrue((run_dir / "reports" / "report.md").is_file())


if __name__ == "__main__": unittest.main()
