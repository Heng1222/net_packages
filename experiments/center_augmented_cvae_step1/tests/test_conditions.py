from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.center_augmented_cvae_step1.conditions import centroid_decomposition, load_condition_set


class ConditionTests(unittest.TestCase):
    def test_centroid_decomposition_recomposes_raw_vectors(self) -> None:
        raw = np.random.default_rng(4).normal(size=(13, 8)).astype(np.float32)
        centroid, centered, decode, gate = centroid_decomposition(raw)
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(raw, centroid[None, :] + centered, atol=1e-6)
        self.assertEqual(decode.shape, (14, 8)); self.assertEqual(gate.shape, (14, 8))
        np.testing.assert_allclose(np.linalg.norm(gate, axis=1), 1.0, atol=1e-6)

    def test_local_yaml_produces_fourteen_conditions_without_network(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            conditions = load_condition_set({
                "path": str(root / "conditions" / "mitre_attack_v11_3_step1.yaml"),
                "cache_dir": directory, "text_fields": ["keywords", "techniques"],
                "common_label": "Common Tactic Component",
                "embedder": {"backend": "hashing", "output_dim": 16, "normalize": True},
            }, torch.device("cpu"))
        self.assertEqual(len(conditions.labels), 14)
        self.assertEqual(conditions.dimension, 16)
        self.assertLess(conditions.metadata["recomposition_error"], 1e-6)


if __name__ == "__main__": unittest.main()

