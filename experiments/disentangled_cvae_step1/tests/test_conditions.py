from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.disentangled_cvae_step1.conditions import (
    apply_condition_geometry,
    cosine_similarity_matrix,
    load_condition_embeddings,
)
from experiments.disentangled_cvae_step1.embedders import HashingTextEmbedder


CONDITION_FILE = Path("experiments/disentangled_cvae_step1/conditions/mitre_attack_v11_3_step1.yaml")


class ConditionEmbeddingTests(unittest.TestCase):
    def test_default_geometry_centers_and_normalizes_without_component_removal(self) -> None:
        config_path = Path("experiments/disentangled_cvae_step1/configs/default.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        geometry = config["conditions"]["geometry"]
        self.assertEqual(geometry["method"], "common_component_removal")
        self.assertTrue(geometry["center"])
        self.assertEqual(geometry["remove_top_components"], 0)
        self.assertTrue(geometry["normalize"])
        self.assertTrue(geometry["append_common_condition"])
        self.assertEqual(geometry["common_label"], "Common Tactic Component")

    def test_text_fields_join_keywords_and_techniques(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            condition_path = root / "conditions.yaml"
            condition_path.write_text(
                yaml.safe_dump(
                    {
                        "tactics": {
                            "B": {
                                "label": "B (TA0002)",
                                "keywords": ["gamma"],
                                "techniques": ["Technique Two (T1002)", "Technique Three (T1003)"],
                            },
                            "A": {
                                "label": "A (TA0001)",
                                "keywords": ["alpha", "beta"],
                                "techniques": ["Technique One (T1001)"],
                            },
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = {
                "path": str(condition_path),
                "format": "yaml",
                "label_field": "label",
                "text_fields": ["keywords", "techniques"],
                "embedder_backend": "hashing",
                "output_dim": 8,
                "normalize": True,
            }

            result = load_condition_embeddings(config, None, torch.device("cpu"), root / "cache")

            self.assertEqual(result.labels, ["A (TA0001)", "B (TA0002)"])
            expected = HashingTextEmbedder(8, True).encode(
                [
                    "alpha, beta, Technique One (T1001)",
                    "gamma, Technique Two (T1002), Technique Three (T1003)",
                ]
            )
            np.testing.assert_allclose(result.matrix, expected)
            self.assertEqual(result.metadata["text_fields"], ["keywords", "techniques"])

            meta_path = Path(str(result.metadata["cache_path"])).with_suffix(".json")
            cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertEqual(cache_meta["text_preview"]["A (TA0001)"], "alpha, beta, Technique One (T1001)")

            second = load_condition_embeddings(config, None, torch.device("cpu"), root / "cache")
            self.assertTrue(second.metadata["cache_hit"])
            np.testing.assert_allclose(second.matrix, expected)

    def test_common_component_geometry_reduces_shared_offset_similarity(self) -> None:
        raw = np.asarray(
            [
                [10.0, 1.0, 0.0, 0.0],
                [10.0, 0.0, 1.0, 0.0],
                [10.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        transformed, metadata = apply_condition_geometry(
            raw,
            {
                "method": "common_component_removal",
                "center": True,
                "remove_top_components": 0,
                "normalize": True,
                "strength": 1.0,
            },
        )

        raw_similarity = cosine_similarity_matrix(raw)
        transformed_similarity = cosine_similarity_matrix(transformed)
        raw_offdiag = raw_similarity[~np.eye(raw_similarity.shape[0], dtype=bool)]
        transformed_offdiag = transformed_similarity[~np.eye(transformed_similarity.shape[0], dtype=bool)]
        self.assertGreater(float(raw_offdiag.mean()), 0.95)
        self.assertLess(float(transformed_offdiag.mean()), 0.0)
        self.assertEqual(transformed.shape, raw.shape)
        self.assertEqual(metadata["condition_geometry"]["method"], "common_component_removal")
        self.assertGreater(metadata["raw_condition_cosine"]["offdiag_mean"], 0.95)
        self.assertLess(metadata["transformed_condition_cosine"]["offdiag_mean"], 0.0)

    def test_deducted_common_vector_is_appended_as_last_condition(self) -> None:
        raw = np.asarray(
            [
                [3.0, 1.0, 0.0, 0.0],
                [3.0, 0.0, 1.0, 0.0],
                [3.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        transformed, metadata = apply_condition_geometry(
            raw,
            {
                "method": "common_component_removal",
                "center": True,
                "remove_top_components": 0,
                "normalize": True,
                "append_common_condition": True,
                "common_label": "Shared",
            },
        )

        self.assertEqual(transformed.shape, (4, 4))
        expected_common = raw.mean(axis=0)
        np.testing.assert_allclose(transformed[-1], expected_common, atol=1e-6)
        common = metadata["condition_geometry"]["common_condition"]
        self.assertTrue(common["appended"])
        self.assertEqual(common["index"], 3)
        self.assertEqual(common["label"], "Shared")

    def test_loader_appends_common_after_all_tactic_labels(self) -> None:
        config = yaml.safe_load(
            Path("experiments/disentangled_cvae_step1/configs/default.yaml").read_text(
                encoding="utf-8"
            )
        )["conditions"]
        config["embedder_backend"] = "hashing"
        config["output_dim"] = 8
        with tempfile.TemporaryDirectory() as folder:
            result = load_condition_embeddings(
                config,
                None,
                torch.device("cpu"),
                Path(folder),
            )

        self.assertEqual(len(result.tactic_labels), 13)
        self.assertEqual(len(result.labels), 14)
        self.assertEqual(result.labels[-1], "Common Tactic Component")
        self.assertEqual(result.matrix.shape, (14, 8))
        self.assertEqual(result.raw_matrix.shape, (14, 8))
        np.testing.assert_allclose(
            result.matrix[-1],
            result.raw_matrix[:-1].mean(axis=0),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.raw_matrix[-1],
            result.raw_matrix[:-1].mean(axis=0),
            atol=1e-6,
        )

    def test_missing_text_field_raises(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            condition_path = root / "conditions.yaml"
            condition_path.write_text(
                yaml.safe_dump(
                    {"tactics": {"A": {"label": "A (TA0001)", "keywords": ["alpha"]}}},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            config = {
                "path": str(condition_path),
                "format": "yaml",
                "label_field": "label",
                "text_fields": ["keywords", "techniques"],
                "embedder_backend": "hashing",
                "output_dim": 8,
                "normalize": True,
            }

            with self.assertRaisesRegex(KeyError, "techniques"):
                load_condition_embeddings(config, None, torch.device("cpu"), root / "cache")

    def test_default_condition_file_uses_complete_technique_names_without_ids(self) -> None:
        data = yaml.safe_load(CONDITION_FILE.read_text(encoding="utf-8"))
        expected_counts = {
            "Initial Access (TA0001)": 9,
            "Execution (TA0002)": 12,
            "Persistence (TA0003)": 19,
            "Privilege Escalation (TA0004)": 13,
            "Defense Evasion (TA0005)": 42,
            "Credential Access (TA0006)": 16,
            "Discovery (TA0007)": 30,
            "Lateral Movement (TA0008)": 9,
            "Collection (TA0009)": 17,
            "Exfiltration (TA0010)": 9,
            "Command and Control (TA0011)": 16,
            "Resource Development (TA0042)": 7,
            "Reconnaissance (TA0043)": 10,
        }

        for label, expected_count in expected_counts.items():
            techniques = data["tactics"][label]["techniques"]
            self.assertEqual(len(techniques), expected_count, msg=label)
            self.assertEqual(len(techniques), len(set(techniques)), msg=label)
            for technique in techniques:
                self.assertIsNone(re.search(r"\(T\d", technique), msg=f"{label}: {technique}")


if __name__ == "__main__":
    unittest.main()
