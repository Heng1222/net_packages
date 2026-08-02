from __future__ import annotations

import unittest

import numpy as np

from experiments.disentangled_cvae_uwf_zeekdata24.training import tactic_pos_weight


class TrainingTests(unittest.TestCase):
    def test_pos_weight_uses_training_rows_only(self) -> None:
        targets = np.asarray(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
            ],
            dtype=np.float32,
        )
        weights = tactic_pos_weight(targets, np.asarray([0, 1, 2, 3]), maximum=50.0)
        np.testing.assert_allclose(weights, np.asarray([3.0, 1.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
