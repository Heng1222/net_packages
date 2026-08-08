from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.ucd_cvae_v2_1.geometry import (
    project_tactic_basis, symmetric_orthogonalize_numpy, uncentered_common_component,
)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3); self.raw = rng.normal(size=(14, 32)).astype(np.float32)

    def test_svd_sign_is_aligned_with_centroid(self) -> None:
        common, residual, _ = uncentered_common_component(self.raw)
        self.assertGreaterEqual(float(common @ self.raw.mean(axis=0)), 0.0)
        self.assertLess(float(np.abs(residual @ common).max()), 1e-5)

    def test_full_and_literal_geometry_contracts(self) -> None:
        common, residual, _ = uncentered_common_component(self.raw)
        full = symmetric_orthogonalize_numpy(residual, forbidden=common)
        combined = torch.from_numpy(np.vstack((common, full)))
        torch.testing.assert_close(combined @ combined.T, torch.eye(15), atol=2e-4, rtol=2e-4)
        literal = project_tactic_basis(torch.from_numpy(residual), torch.from_numpy(common), "common_removal_only")
        self.assertLess(float(torch.abs(literal @ torch.from_numpy(common)).max()), 1e-5)
        self.assertGreater(float(torch.abs(literal @ literal.T - torch.eye(14)).max()), 1e-3)

    def test_rank_failure_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            symmetric_orthogonalize_numpy(np.ones((14, 32), dtype=np.float32))

    def test_condition_asset_has_fourteen_tactics_and_impact(self) -> None:
        path = Path(__file__).resolve().parents[1] / "conditions" / "mitre_attack_enterprise_v11_3.yaml"
        tactics = yaml.safe_load(path.read_text(encoding="utf-8"))["tactics"]
        self.assertEqual(len(tactics), 14); self.assertIn("Impact (TA0040)", tactics)


if __name__ == "__main__": unittest.main()
