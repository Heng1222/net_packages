from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.disentangled_cvae_step1.evaluate import (
    behavior_alignment_metrics,
    build_test_condition_predictions,
)


class ConditionPredictionTests(unittest.TestCase):
    def test_independent_threshold_assigns_multilabel_conditions(self) -> None:
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "label": ["metadata-a", "metadata-b", "metadata-c"],
            }
        )
        frame = build_test_condition_predictions(
            metadata,
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([[0.8, 0.2], [0.6, 0.7], [0.49, 0.2]], dtype=np.float32),
            ["Condition A", "Condition B"],
        )

        self.assertEqual(frame.loc[0, "predicted_condition"], "Condition A")
        self.assertEqual(frame.loc[0, "predicted_conditions"], "Condition A")
        self.assertEqual(frame.loc[0, "active_condition_count"], 1)
        self.assertEqual(frame.loc[1, "predicted_condition"], "Condition B")
        self.assertEqual(frame.loc[1, "predicted_conditions"], "Condition A|Condition B")
        self.assertEqual(frame.loc[1, "active_condition_count"], 2)
        self.assertEqual(frame.loc[2, "predicted_condition"], "ambiguous")
        self.assertEqual(frame.loc[2, "predicted_conditions"], "ambiguous")
        self.assertEqual(frame.loc[2, "active_condition_count"], 0)
        prob_cols = [column for column in frame.columns if column.startswith("condition_prob__")]
        np.testing.assert_allclose(
            frame[prob_cols].to_numpy(),
            np.asarray([[0.8, 0.2], [0.6, 0.7], [0.49, 0.2]], dtype=np.float32),
        )

    def test_behavior_alignment_metrics_uses_gold_tactic(self) -> None:
        predictions = pd.DataFrame(
            {
                "gold_tactic": ["Condition A", "Condition B", ""],
                "predicted_condition": ["Condition A", "ambiguous", "Condition B"],
            }
        )
        metrics = behavior_alignment_metrics(predictions, ["Condition A", "Condition B"])

        self.assertTrue(metrics["enabled"])
        self.assertEqual(metrics["labeled_rows"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertIn("classification_report", metrics)


if __name__ == "__main__":
    unittest.main()
