from __future__ import annotations

import unittest
import torch
from torch.nn import functional as F

from experiments.center_augmented_cvae_step1.model import CenterAugmentedCVAE, PlainVAE


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = CenterAugmentedCVAE(8, 3, 14, 8, [16], [16], gate_temperature=0.2)
        self.decode = torch.randn(14, 8); self.gate = F.normalize(self.decode, dim=1)
        self.x = F.normalize(torch.randn(5, 8), dim=1)

    def test_shapes_and_additive_identity(self) -> None:
        output = self.model(self.x, self.decode, self.gate, sample=False)
        self.assertEqual(output["gates"].shape, (5, 14)); self.assertEqual(output["z_mu"].shape, (5, 3))
        torch.testing.assert_close(output["x_recon"], output["residual_component"] + output["condition_component"])

    def test_gate_is_fixed_cosine_without_projector(self) -> None:
        output = self.model(self.x, self.decode, self.gate, sample=False)
        expected = torch.sigmoid((F.normalize(self.x, dim=1) @ F.normalize(self.gate, dim=1).T) / 0.2)
        torch.testing.assert_close(output["gates"], expected)
        self.assertFalse(any("projector" in name for name, _ in self.model.named_parameters()))

    def test_plain_vae_shapes(self) -> None:
        self.assertEqual(PlainVAE(8, 3, [12], [12])(self.x, sample=False)["x_recon"].shape, (5, 8))


if __name__ == "__main__": unittest.main()
