from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.golden_oracle_cvae_step2.data import (
    make_gate_targets,
    make_stratified_group_split,
    prepare_golden_dataset,
)


class GoldenDataTests(unittest.TestCase):
    def test_prepare_excludes_conflicts_and_deduplicates_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "golden.csv"
            pd.DataFrame(
                {
                    "Session_ID": ["s1", "s2", "s3", "s4", "s5"],
                    "clean_payload_list": ["same", "same", "dup", "dup", "unique"],
                    "Tactic": ["A", "B", "A", "A", "Normal"],
                }
            ).to_csv(source, index=False)
            prepared = prepare_golden_dataset(
                {
                    "input_path": str(source),
                    "prepared_dir": str(root / "prepared"),
                    "sample_id_col": "Session_ID",
                    "payload_text_col": "clean_payload_list",
                    "label_col": "Tactic",
                    "normal_label": "Normal",
                    "min_class_count": 1,
                    "conflicting_payload_policy": "exclude",
                    "deduplicate_payloads": True,
                    "embedder": {"backend": "hashing", "output_dim": 8, "normalize": True},
                },
                root,
                torch.device("cpu"),
            )
            self.assertEqual(prepared.summary["excluded_conflicting_payload_hashes"], 1)
            self.assertEqual(prepared.summary["excluded_conflicting_rows"], 2)
            self.assertEqual(prepared.summary["duplicate_rows_removed"], 1)
            self.assertEqual(prepared.summary["rows"], 2)

    def test_group_split_has_no_payload_overlap(self) -> None:
        rows = []
        for label in ("A", "B", "Normal"):
            for index in range(15):
                rows.append({"label": label, "payload_hash": f"{label}-{index}"})
        metadata = pd.DataFrame(rows)
        split = make_stratified_group_split(
            metadata,
            {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
            42,
        )
        hashes = [set(metadata.iloc[indices]["payload_hash"]) for indices in (split.train, split.val, split.test)]
        self.assertFalse(hashes[0] & hashes[1])
        self.assertFalse(hashes[0] & hashes[2])
        self.assertFalse(hashes[1] & hashes[2])
        self.assertEqual(sum(map(len, (split.train, split.val, split.test))), len(metadata))

    def test_normal_maps_to_all_zero_gate(self) -> None:
        targets = make_gate_targets(
            np.asarray(["A", "Normal", "B"]), ["A", "B"], "Normal"
        )
        np.testing.assert_array_equal(targets, np.asarray([[1, 0], [0, 0], [0, 1]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
