from __future__ import annotations

import unittest

import numpy as np

from experiments.disentangled_cvae_uwf_zeekdata24.evaluate import (
    bootstrap_macro_f1_difference,
    calibrate_thresholds,
    multilabel_metrics,
    run_technique_probes,
)


class EvaluationTests(unittest.TestCase):
    def test_thresholds_and_multilabel_metrics(self) -> None:
        targets = np.asarray([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)
        probabilities = np.asarray([[0.9, 0.1], [0.8, 0.7], [0.2, 0.8], [0.1, 0.2]], dtype=np.float32)
        thresholds = calibrate_thresholds(targets, probabilities, [0.3, 0.5, 0.7])
        metrics = multilabel_metrics(targets, probabilities, thresholds, ["A", "B"])
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["exact_match"], 1.0)

    def test_probe_selects_c_using_validation_only(self) -> None:
        rng = np.random.default_rng(42)
        train_labels = np.asarray(["T1"] * 20 + ["T2"] * 20)
        val_labels = np.asarray(["T1"] * 8 + ["T2"] * 8)
        test_labels = np.asarray(["T1"] * 8 + ["T2"] * 8)

        def features(labels: np.ndarray) -> np.ndarray:
            signal = (labels == "T2").astype(np.float32)[:, None]
            return np.hstack((signal, rng.normal(scale=0.05, size=(len(labels), 2)))).astype(np.float32)

        values = (features(train_labels), features(val_labels), features(test_labels))
        result = run_technique_probes({"c": values, "h": values}, train_labels, val_labels, test_labels, [0.1, 1.0], 42)
        self.assertIn(result.best_c_by_representation["c"], {0.1, 1.0})
        self.assertEqual(result.metrics["c"]["test_macro_f1"], 1.0)

    def test_bootstrap_uses_macro_f1_difference(self) -> None:
        targets = np.asarray(["A", "A", "B", "B"])
        perfect = targets.copy()
        poor = np.asarray(["A", "A", "A", "A"])
        result = bootstrap_macro_f1_difference(targets, perfect, poor, 50, 9)
        self.assertEqual(result["metric"], "macro_f1_difference")
        self.assertGreater(result["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
