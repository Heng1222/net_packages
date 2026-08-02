from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.disentangled_cvae_uwf_zeekdata24.controls import (
    ControlResult,
    aggregate_control_decision,
    build_condition_control_matrices,
    paired_control_comparison,
    run_control_variant,
    shuffled_supervision_targets,
)
from experiments.disentangled_cvae_uwf_zeekdata24.data import SplitIndices


class ControlTests(unittest.TestCase):
    def test_random_controls_match_norms_and_orthogonal_geometry(self) -> None:
        rng = np.random.default_rng(4)
        semantic = rng.normal(size=(14, 32)).astype(np.float32)
        semantic[:13] /= np.linalg.norm(semantic[:13], axis=1, keepdims=True)
        semantic[13] *= 0.25 / np.linalg.norm(semantic[13])
        first = build_condition_control_matrices(semantic, 42)
        second = build_condition_control_matrices(semantic, 42)
        for name in first:
            np.testing.assert_allclose(first[name], second[name])
            np.testing.assert_allclose(
                np.linalg.norm(first[name], axis=1), np.linalg.norm(semantic, axis=1), atol=1e-6
            )
        normalized = first["random_orthogonal"] / np.linalg.norm(
            first["random_orthogonal"], axis=1, keepdims=True
        )
        np.testing.assert_allclose(normalized @ normalized.T, np.eye(14), atol=1e-6)

    def test_label_shuffle_only_changes_train_and_validation(self) -> None:
        targets = np.arange(60, dtype=np.float32).reshape(20, 3)
        split = SplitIndices(np.arange(0, 10), np.arange(10, 15), np.arange(15, 20))
        shuffled = shuffled_supervision_targets(targets, split, 7)
        np.testing.assert_array_equal(shuffled[split.test], targets[split.test])
        self.assertFalse(np.array_equal(shuffled[split.train], targets[split.train]))
        self.assertCountEqual(map(tuple, shuffled[split.train]), map(tuple, targets[split.train]))
        self.assertCountEqual(map(tuple, shuffled[split.val]), map(tuple, targets[split.val]))

    def test_paired_comparison_detects_semantic_advantage(self) -> None:
        tactic_truth = np.asarray(
            [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.int8
        )
        semantic_scores = tactic_truth * 0.8 + 0.1
        control_scores = 1.0 - semantic_scores
        probe_gold = np.asarray(["A", "A", "B", "B", "C", "C"])

        def result(name: str, scores: np.ndarray, prediction: np.ndarray) -> ControlResult:
            return ControlResult(
                name,
                42,
                {},
                scores.astype(np.float32),
                np.asarray([0.5, 0.5], dtype=np.float32),
                probe_gold,
                {"c": prediction, "gates": prediction, "h": prediction},
            )

        semantic = result("semantic", semantic_scores, probe_gold.copy())
        control = result("random_gaussian", control_scores, np.asarray(["A"] * 6))
        comparison = paired_control_comparison(
            semantic,
            control,
            tactic_truth,
            np.asarray([True, True]),
            probe_gold,
            probe_gold,
            50,
            3,
        )
        self.assertTrue(comparison["supported"])
        decision = aggregate_control_decision([comparison], ["random_gaussian"], [42], 1.0)
        self.assertTrue(decision["semantic_geometry_supported"])

    def test_one_epoch_control_variant_smoke(self) -> None:
        rng = np.random.default_rng(12)
        classes = ["Benign", "T1048", "T1078", "T1110", "T1190", "T1595"]
        techniques = np.asarray([label for label in classes for _ in range(6)])
        train = np.asarray([base + offset for base in range(0, 36, 6) for offset in range(3)])
        val = np.asarray([base + 3 for base in range(0, 36, 6)])
        test = np.asarray([base + offset for base in range(0, 36, 6) for offset in (4, 5)])
        split = SplitIndices(train, val, test)
        x = rng.normal(size=(36, 16)).astype(np.float32)
        targets = np.zeros((36, 13), dtype=np.float32)
        mapping = {
            "T1048": [6],
            "T1078": [3, 7, 9, 10],
            "T1110": [2],
            "T1190": [7],
            "T1595": [11],
        }
        for row, technique in enumerate(techniques):
            targets[row, mapping.get(technique, [])] = 1.0
        metadata = pd.DataFrame(
            {
                "technique": techniques,
                "probe_eligible": techniques != "Benign",
            }
        )
        conditions = rng.normal(size=(14, 16)).astype(np.float32)
        conditions /= np.linalg.norm(conditions, axis=1, keepdims=True)
        config = {
            "model": {
                "input_dim": 16,
                "residual_dim": 4,
                "condition_dim": 16,
                "encoder_hidden_dims": [12],
                "decoder_hidden_dims": [12],
                "behavior_projector_hidden_dims": [8],
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
                    "sparse": 0.0,
                    "gate_entropy": 0.0,
                    "utility": 0.0,
                    "residual_constraint": 0.1,
                    "behavior_infonce": 1.0,
                    "residual_adversary": 0.1,
                },
            },
            "training": {
                "batch_size": 8,
                "max_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "early_stopping_patience": 1,
                "max_pos_weight": 10.0,
                "num_workers": 0,
            },
            "evaluation": {
                "threshold_grid": [0.25, 0.5, 0.75],
                "probe_c_grid": [0.1],
            },
            "controls": {"deterministic": True, "keep_checkpoints": False},
        }
        with tempfile.TemporaryDirectory() as folder:
            result = run_control_variant(
                "semantic",
                9,
                x,
                metadata,
                targets,
                split,
                conditions,
                config,
                torch.device("cpu"),
                Path(folder),
            )
            self.assertIn("macro_auprc", result.summary["tactic"])
            self.assertIn("gates", result.summary["probes"])
            self.assertTrue((Path(folder) / "metrics/control_result.npz").is_file())
            self.assertFalse((Path(folder) / "checkpoints/disentangled_cvae.pt").exists())


if __name__ == "__main__":
    unittest.main()
