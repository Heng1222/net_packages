from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.disentangled_cvae_step1.model import DisentangledConditionalVAE


class MultiLabelDisentangledConditionalVAE(DisentangledConditionalVAE):
    """The original architecture with multi-label tactic supervision."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("tactic_pos_weight", torch.ones(self.supervised_condition_count))

    def set_tactic_pos_weight(self, value: torch.Tensor) -> None:
        weights = value.detach().to(device=self.tactic_pos_weight.device, dtype=torch.float32)
        if tuple(weights.shape) != (self.supervised_condition_count,):
            raise ValueError(
                f"pos_weight must have shape ({self.supervised_condition_count},); got {tuple(weights.shape)}"
            )
        self.tactic_pos_weight.copy_(weights)

    def loss(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        behavior_targets: torch.Tensor | None = None,
        compute_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        losses = super().loss(output, target, behavior_targets=None, compute_diagnostics=compute_diagnostics)
        zero = target.new_tensor(0.0)
        behavior_bce = zero
        residual_bce = zero
        behavior_accuracy = zero
        residual_accuracy = zero
        labeled_count = target.new_tensor(0.0)
        if behavior_targets is not None:
            labels = behavior_targets.to(device=target.device, dtype=torch.float32)
            expected = (len(target), self.supervised_condition_count)
            if tuple(labels.shape) != expected:
                raise ValueError(f"behavior_targets must have shape {expected}; got {tuple(labels.shape)}")
            valid = torch.isfinite(labels).all(dim=1)
            valid_count = int(valid.sum().item())
            labeled_count = target.new_tensor(float(valid_count))
            if valid_count:
                valid_labels = labels[valid]
                behavior_logits = output["behavior_logits"][valid]
                adversary_logits = output["residual_adversary_logits"][valid]
                behavior_bce = F.binary_cross_entropy_with_logits(
                    behavior_logits,
                    valid_labels,
                    pos_weight=self.tactic_pos_weight,
                )
                residual_bce = F.binary_cross_entropy_with_logits(
                    adversary_logits,
                    valid_labels,
                    pos_weight=self.tactic_pos_weight,
                )
                behavior_accuracy = (
                    (torch.sigmoid(behavior_logits) >= 0.5) == valid_labels.bool()
                ).float().mean()
                residual_accuracy = (
                    (torch.sigmoid(adversary_logits) >= 0.5) == valid_labels.bool()
                ).float().mean()
        total = (
            losses["loss"]
            + self.weights["behavior_infonce"] * behavior_bce
            + self.weights["residual_adversary"] * residual_bce
        )
        losses.update(
            {
                "loss": total,
                "total_loss": total,
                "behavior_infonce_loss": behavior_bce,
                "residual_adversary_loss": residual_bce,
                "behavior_infonce_accuracy": behavior_accuracy,
                "residual_adversary_accuracy": residual_accuracy,
                "behavior_labeled_count": labeled_count,
            }
        )
        return losses

