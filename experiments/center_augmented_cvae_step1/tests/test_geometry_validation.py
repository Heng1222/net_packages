from __future__ import annotations

import unittest

import numpy as np

from experiments.center_augmented_cvae_step1.geometry_validation import (
    mean_reconstruction_metrics, streaming_mean, transform_vectors, variant_gate_matrix,
)


class GeometryValidationTests(unittest.TestCase):
    def test_mean_baseline_uses_train_rows_only(self) -> None:
        x = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        mean = streaming_mean(x, np.asarray([0, 1]), 1)
        np.testing.assert_allclose(mean, [0.5, 0.5])
        metrics = mean_reconstruction_metrics(x, np.asarray([2]), mean, 1)
        self.assertAlmostEqual(metrics["recon_mse"], 1.25)

    def test_mean_and_component_are_removed_before_normalization(self) -> None:
        values = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32)
        transformed = transform_vectors(values, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(transformed, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_variant_matrix_retains_fourteen_gate_directions(self) -> None:
        raw = np.random.default_rng(2).normal(size=(13, 8)).astype(np.float32)
        gate, metadata = variant_gate_matrix(raw, np.zeros(8, dtype=np.float32), None)
        self.assertEqual(gate.shape, (14, 8))
        self.assertLess(metadata["centered_mean_error"], 1e-6)


if __name__ == "__main__": unittest.main()
