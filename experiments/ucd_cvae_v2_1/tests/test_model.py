from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from experiments.ucd_cvae_v2_1.model import UCDCVAE, compute_losses
from experiments.ucd_cvae_v2_1.tests.helpers import small_basis


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        common, tactics = small_basis(); self.model = UCDCVAE(
            common, tactics, input_dim=32, residual_dim=16, gate_hidden_dims=[24],
            residual_hidden_dims=[24], residual_up_hidden_dims=[24], decoder_hidden_dims=[24],
            concept_projector_hidden_dim=16, dropout=0.0)
        self.x = F.normalize(torch.randn(5, 32), dim=1)

    def test_forward_shapes_additive_identity_and_orthogonality(self) -> None:
        output = self.model(self.x, sample=False)
        self.assertEqual(output["gates"].shape, (5, 15)); self.assertEqual(output["mu_r"].shape, (5, 16))
        torch.testing.assert_close(output["h_latent"], output["concept_component"] + output["h_res_perp"])
        self.assertLess(float(torch.abs(output["h_res_perp"] @ output["projected_basis"].T).max().detach()), 2e-5)
        torch.testing.assert_close(output["projected_basis"] @ output["projected_basis"].T,
                                   torch.eye(15), atol=2e-4, rtol=2e-4)

    def test_losses_have_expected_contract_and_detached_targets(self) -> None:
        output = self.model(self.x); losses = compute_losses(output, self.x, 0.1)
        self.assertEqual({"reconstruction_loss", "kl_loss", "sparse_loss", "align_loss",
                          "alignment_targets", "recon_cosine"}, set(losses))
        self.assertFalse(losses["alignment_targets"].requires_grad)
        torch.testing.assert_close(losses["sparse_loss"], output["gates"].sum(dim=1).mean())
        self.assertTrue(torch.isfinite(sum(losses[key] for key in
            ("reconstruction_loss", "kl_loss", "sparse_loss", "align_loss"))))


if __name__ == "__main__": unittest.main()
