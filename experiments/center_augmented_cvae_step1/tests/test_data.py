from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from experiments.center_augmented_cvae_step1.data import make_time_split, payload_to_text


class DataTests(unittest.TestCase):
    def test_payload_list_parser(self) -> None:
        self.assertEqual(payload_to_text("['one', 'two']"), "one\n[PACKET]\ntwo")

    def test_time_split_is_chronological(self) -> None:
        metadata = pd.DataFrame({"datetime": ["2022-01-03", "2022-01-01", "2022-01-04", "2022-01-02"]})
        split = make_time_split(metadata, {"train_ratio": 0.5, "val_ratio": 0.25, "test_ratio": 0.25})
        np.testing.assert_array_equal(split.train, [1, 3])
        np.testing.assert_array_equal(split.val, [0])
        np.testing.assert_array_equal(split.test, [2])


if __name__ == "__main__": unittest.main()
