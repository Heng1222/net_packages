from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.ucd_cvae_v2_1.data import payload_to_text, prepare_dataset


class DataTests(unittest.TestCase):
    def test_payload_parser(self) -> None:
        self.assertEqual(payload_to_text("['GET /a', 'Host: x']"), "GET /a\n[PACKET]\nHost: x")

    def test_prepare_filters_only_empty_payload_and_reuses_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "input.csv"
            pd.DataFrame({"Session_ID": ["a", "b", "c"], "Datetime": ["2022-01-01", "2022-01-02", "2022-01-03"],
                          "clean_payload_list": ["GET /a", "", "POST /c"],
                          "Sess_Tactic_predict": ["", "Execution", None]}).to_csv(source, index=False)
            config = {"data": {"input_path": str(source), "prepared_dir": str(root / "prepared"),
                "sample_id_col": "Session_ID", "payload_text_col": "clean_payload_list", "time_col": "Datetime",
                "metadata_cols": ["Sess_Tactic_predict"], "payload_parser": "auto", "read_chunksize": 2,
                "max_rows": None, "embedder": {"backend": "hashing", "output_dim": 32, "normalize": True}}}
            first = prepare_dataset(config, root, torch.device("cpu")); second = prepare_dataset(config, root, torch.device("cpu"))
            self.assertFalse(first.reused); self.assertTrue(second.reused)
            self.assertEqual(np.load(first.x_path).shape, (2, 32))
            metadata = pd.read_csv(first.metadata_path, keep_default_na=False)
            self.assertEqual(metadata["sample_id"].tolist(), ["a", "c"])


if __name__ == "__main__": unittest.main()
