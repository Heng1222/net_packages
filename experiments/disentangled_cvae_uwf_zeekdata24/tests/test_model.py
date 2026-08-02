from __future__ import annotations

import unittest

import torch

from experiments.disentangled_cvae_uwf_zeekdata24.model import MultiLabelDisentangledConditionalVAE


class ModelTests(unittest.TestCase):
    def test_fourteen_gates_and_thirteen_supervised_logits(self) -> None:
        model = MultiLabelDisentangledConditionalVAE(
            input_dim=16,
            residual_dim=4,
            condition_count=14,
            condition_dim=16,
            supervised_condition_count=13,
            encoder_hidden_dims=[12],
            decoder_hidden_dims=[12],
            behavior_projector_hidden_dims=[8],
        )
        x = torch.randn(6, 16)
        output = model(x, torch.randn(14, 16), sample=False)
        targets = torch.zeros(6, 13)
        targets[0, 2] = 1
        targets[1, [3, 7, 9, 10]] = 1
        losses = model.loss(output, x, targets)
        self.assertEqual(output["gates"].shape, (6, 14))
        self.assertEqual(output["behavior_logits"].shape, (6, 13))
        self.assertEqual(output["residual_adversary_logits"].shape, (6, 13))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["behavior_infonce_loss"].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
