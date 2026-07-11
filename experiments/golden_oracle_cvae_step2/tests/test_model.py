from __future__ import annotations

import unittest

import torch

from experiments.golden_oracle_cvae_step2.model import GoldenConditionalVAE


class GoldenModelTests(unittest.TestCase):
    def test_oracle_and_predicted_forward_shapes(self) -> None:
        model = GoldenConditionalVAE(
            input_dim=16,
            condition_dim=16,
            condition_count=3,
            residual_dim=4,
            encoder_hidden_dims=[12],
            decoder_hidden_dims=[12],
            projector_hidden_dims=[12],
        )
        x = torch.randn(5, 16)
        conditions = torch.randn(3, 16)
        gold = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32)
        oracle = model(x, conditions, gates_override=gold, sample=False)
        predicted = model(x, conditions, sample=False)
        self.assertEqual(oracle["x_recon"].shape, (5, 16))
        self.assertEqual(predicted["predicted_gates"].shape, (5, 3))
        torch.testing.assert_close(oracle["used_gates"], gold)
        self.assertTrue(((predicted["predicted_gates"] >= 0) & (predicted["predicted_gates"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
