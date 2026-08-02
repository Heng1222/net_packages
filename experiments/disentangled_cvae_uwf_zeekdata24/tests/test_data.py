from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.disentangled_cvae_uwf_zeekdata24.data import (
    TACTIC_LABELS,
    aggregate_flows,
    flow_to_text,
    sample_by_technique,
    stratified_technique_split,
    tactic_target_matrix,
)
from experiments.disentangled_cvae_uwf_zeekdata24.tests.helpers import uwf_row


class DataTests(unittest.TestCase):
    def test_flow_text_excludes_labels_identifiers_ips_and_time(self) -> None:
        row = uwf_row("secret-uid", "Credential Access", "T1110")
        text = flow_to_text(row)
        for forbidden in ("secret-uid", "Credential Access", "T1110", "143.88", "2024-03"):
            self.assertNotIn(forbidden, text)
        self.assertIn("protocol: tcp", text)
        self.assertIn("destination port: 445", text)

    def test_uid_rows_become_one_multilabel_target(self) -> None:
        rows = [
            uwf_row("shared", "Initial Access", "T1078"),
            uwf_row("shared", "Defense Evasion", "T1078"),
            uwf_row("shared", "Persistence", "T1078"),
            uwf_row("shared", "Privilege Escalation", "T1078"),
        ]
        frame = aggregate_flows([pd.DataFrame(rows)])
        targets = tactic_target_matrix(frame)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "technique"], "T1078")
        self.assertEqual(int(targets.sum()), 4)
        expected = {
            "Initial Access (TA0001)", "Defense Evasion (TA0005)",
            "Persistence (TA0003)", "Privilege Escalation (TA0004)",
        }
        actual = {TACTIC_LABELS[index] for index in np.flatnonzero(targets[0])}
        self.assertEqual(actual, expected)

    def test_official_duplicate_sentinel_maps_to_t1078_group(self) -> None:
        row = uwf_row("duplicate-row", "Persistence", "Duplicate")
        row["label_binary"] = "Duplicate"
        frame = aggregate_flows([pd.DataFrame([row])])
        targets = tactic_target_matrix(frame)
        self.assertEqual(frame.loc[0, "technique"], "T1078")
        self.assertEqual(int(frame.loc[0, "duplicate_sentinel_rows"]), 1)
        self.assertEqual(int(targets.sum()), 4)

    def test_duplicate_sentinel_is_rejected_outside_t1078_tactics(self) -> None:
        row = uwf_row("invalid-duplicate", "Credential Access", "Duplicate")
        with self.assertRaisesRegex(ValueError, "only valid for a UWF T1078 tactic row"):
            aggregate_flows([pd.DataFrame([row])])

    def test_stratified_split_is_reproducible_and_disjoint(self) -> None:
        labels = np.asarray([label for label in ("Benign", "T1110", "T1595") for _ in range(20)])
        config = {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}
        first = stratified_technique_split(labels, config, 42)
        second = stratified_technique_split(labels, config, 42)
        np.testing.assert_array_equal(first.train, second.train)
        self.assertFalse(set(first.train) & set(first.val))
        self.assertFalse(set(first.train) & set(first.test))
        self.assertEqual(set(np.concatenate((first.train, first.val, first.test))), set(range(len(labels))))

    def test_sampling_is_capped_and_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": [f"b-{index}" for index in range(10)] + [f"a-{index}" for index in range(4)],
                "technique": ["Benign"] * 10 + ["T1078"] * 4,
            }
        )
        first = sample_by_technique(frame, {"Benign": 3}, 17)
        second = sample_by_technique(frame, {"Benign": 3}, 17)
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        self.assertEqual(int((first["technique"] == "Benign").sum()), 3)
        self.assertEqual(int((first["technique"] == "T1078").sum()), 4)


if __name__ == "__main__":
    unittest.main()
