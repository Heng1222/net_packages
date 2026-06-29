from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.disentangled_cvae_step1.data import (
    expected_manifest,
    manifest_matches,
    parse_step1_label,
    prepare_dataset,
)


class LabelParserTests(unittest.TestCase):
    def test_single_label_set(self) -> None:
        self.assertEqual(
            parse_step1_label("{'Reconnaissance (TA0043)'}"),
            "Reconnaissance (TA0043)",
        )

    def test_empty_label(self) -> None:
        self.assertIsNone(parse_step1_label("set()"))
        self.assertIsNone(parse_step1_label(""))

    def test_multi_label_error_and_first(self) -> None:
        value = "{'A', 'B'}"
        with self.assertRaisesRegex(ValueError, "Expected one"):
            parse_step1_label(value)
        self.assertEqual(parse_step1_label(value, "first"), "A")

    def test_invalid_literal_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid"):
            parse_step1_label("{not valid")


class PrepareCacheTests(unittest.TestCase):
    def test_prepare_manifest_reuse_and_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            csv_path = root / "step1.csv"
            pd.DataFrame(
                {
                    "Session_ID": ["s1", "s2", "s3"],
                    "Datetime": ["2022-10-01", "2022-10-02", "2022-10-03"],
                    "clean_payload_list": ["GET /a", "GET /b", "GET /c"],
                    "Sess_Tactic_predict": [
                        "{'Reconnaissance (TA0043)'}",
                        "{'Execution (TA0002)'}",
                        "{'Normal (TA9000)'}",
                    ],
                    "Src_ISP": ["cht", "cht", "cht"],
                    "Protocol": ["http", "http", "http"],
                }
            ).to_csv(csv_path, index=False)
            config = {
                "data": {
                    "input_path": str(csv_path),
                    "prepared_dir": str(root / "prepared"),
                    "sample_id_col": "Session_ID",
                    "payload_text_col": "clean_payload_list",
                    "payload_parser": "auto",
                    "label_col": "Sess_Tactic_predict",
                    "time_col": "Datetime",
                    "metadata_cols": ["Src_ISP", "Protocol"],
                    "read_chunksize": 2,
                    "max_rows": None,
                    "label_multi_policy": "error",
                    "skip_empty_labels": True,
                    "embedder": {"backend": "hashing", "output_dim": 768, "normalize": True},
                }
            }
            first = prepare_dataset(config, root, torch.device("cpu"))
            self.assertFalse(first.reused)
            x = np.load(first.x_path)
            self.assertEqual(x.shape, (3, 768))
            second = prepare_dataset(config, root, torch.device("cpu"))
            self.assertTrue(second.reused)
            expected = expected_manifest(config["data"], csv_path)
            self.assertTrue(manifest_matches(first.manifest_path, expected))
            expected["payload_text_col"] = "payload_list"
            self.assertFalse(manifest_matches(first.manifest_path, expected))


if __name__ == "__main__":
    unittest.main()

