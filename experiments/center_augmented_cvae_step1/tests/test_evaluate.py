from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.center_augmented_cvae_step1.evaluate import golden_alignment


class EvaluationTests(unittest.TestCase):
    def test_golden_labels_are_evaluation_only_and_normal_is_reported(self) -> None:
        metadata = pd.DataFrame({"sample_id": ["a", "b"]})
        x = np.asarray([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        gate = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); gold_path = root / "gold.csv"
            pd.DataFrame({"Session_ID": ["a", "b"], "Tactic": ["T1", "Normal (TA9000)"]}).to_csv(gold_path, index=False)
            summary = golden_alignment(metadata, x, np.asarray(["test", "test"]), gate,
                                       ["Common", "T1", "T2"],
                                       {"golden_path": str(gold_path), "golden_sample_id_col": "Session_ID",
                                        "golden_label_col": "Tactic"}, root, 0.2)
        self.assertFalse(summary["labels_used_for_training"])
        self.assertEqual(summary["matched_tactic_rows"], 1)
        self.assertEqual(summary["matched_normal_rows"], 1)


if __name__ == "__main__": unittest.main()
