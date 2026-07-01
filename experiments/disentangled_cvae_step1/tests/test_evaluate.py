from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.disentangled_cvae_step1.evaluate import build_test_condition_predictions


class ConditionPredictionTests(unittest.TestCase):
    def test_softmax_threshold_assigns_condition_or_ambiguous(self) -> None:
        metadata = pd.DataFrame(
            {
                "sample_id": ["s1", "s2"],
                "label": ["metadata-a", "metadata-b"],
            }
        )
        frame = build_test_condition_predictions(
            metadata,
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([[2.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            ["Condition A", "Condition B"],
        )

        self.assertEqual(frame.loc[0, "predicted_condition"], "Condition A")
        self.assertGreater(frame.loc[0, "max_condition_probability"], 0.5)
        self.assertEqual(frame.loc[1, "predicted_condition"], "ambiguous")
        self.assertEqual(frame.loc[1, "max_condition_probability"], 0.5)
        prob_cols = [column for column in frame.columns if column.startswith("condition_prob__")]
        np.testing.assert_allclose(frame[prob_cols].sum(axis=1).to_numpy(), np.ones(2))


if __name__ == "__main__":
    unittest.main()
