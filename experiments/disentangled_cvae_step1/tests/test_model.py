from __future__ import annotations

import unittest

import torch

from experiments.disentangled_cvae_step1.model import DisentangledConditionalVAE


class ModelTests(unittest.TestCase):
    def _model(self) -> DisentangledConditionalVAE:
        return DisentangledConditionalVAE(
            input_dim=768,
            residual_dim=8,
            condition_count=3,
            condition_dim=768,
            encoder_hidden_dims=[32],
            decoder_hidden_dims=[32],
            batch_norm=False,
            observation_variance=1.0,
            weights={
                "reconstruction": 1.0,
                "kl": 1.0,
                "decorrelation": 0.1,
                "sparse": 0.001,
                "gate_entropy": 0.01,
                "utility": 0.5,
                "residual_constraint": 0.5,
            },
        )

    def test_forward_shapes_and_loss(self) -> None:
        model = self._model()
        x = torch.randn(5, 768)
        conditions = torch.randn(3, 768)
        output = model(x, conditions, sample=False)
        self.assertEqual(output["h"].shape, (5, 8))
        self.assertEqual(output["conditions"].shape, (5, 3, 768))
        self.assertEqual(output["gate_logits"].shape, (5, 3))
        self.assertEqual(output["gates"].shape, (5, 3))
        self.assertEqual(output["x_recon"].shape, (5, 768))
        losses = model.loss(output, x)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreaterEqual(float(losses["sparse_loss"].detach()), 0.0)
        self.assertGreaterEqual(float(losses["gate_entropy_loss"].detach()), 0.0)
        self.assertEqual(losses["ablation_delta_mse"].shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
