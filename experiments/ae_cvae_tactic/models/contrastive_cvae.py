from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .common import build_mlp
from .cvae import ConditionalVAE


class ContrastiveConditionalVAE(ConditionalVAE):
    """CVAE with an additional payload-only semantic alignment branch.

    The CVAE path remains q(z | x, c) / p(x | z, c).  The contrastive path is
    deliberately restricted to x so that it cannot copy the oracle condition.
    Candidate condition embeddings stay in their original frozen space.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        reconstruction_loss: str = "mse",
        objective: str = "elbo",
        likelihood: str = "gaussian",
        observation_variance: float = 1.0,
        latent_representation: str = "mu",
        projection_dim: int | None = None,
        projection_hidden_dims: list[int] | None = None,
        temperature: float = 0.1,
        contrastive_weight: float = 100.0,
        condition_projection: str = "identity",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            reconstruction_loss=reconstruction_loss,
            objective=objective,
            likelihood=likelihood,
            observation_variance=observation_variance,
            latent_representation=latent_representation,
        )
        self.projection_dim = int(projection_dim or condition_dim)
        self.temperature = float(temperature)
        self.contrastive_weight = float(contrastive_weight)
        self.condition_projection = condition_projection
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.contrastive_weight < 0:
            raise ValueError("contrastive_weight must be non-negative.")
        if condition_projection != "identity":
            raise ValueError("The first contrastive experiment supports condition_projection='identity' only.")
        if self.projection_dim != condition_dim:
            raise ValueError(
                "identity condition projection requires projection_dim to equal condition_dim. "
                f"Got projection_dim={self.projection_dim}, condition_dim={condition_dim}."
            )

        projector_hidden = list(projection_hidden_dims or [])
        self.payload_projector_body, projector_output_dim = build_mlp(
            input_dim, projector_hidden, activation, dropout, batch_norm
        )
        self.payload_projector_out = nn.Linear(projector_output_dim, self.projection_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ContrastiveConditionalVAE":
        keys = (
            "input_dim",
            "condition_dim",
            "latent_dim",
            "hidden_dims",
            "dropout",
            "batch_norm",
            "activation",
            "reconstruction_loss",
            "objective",
            "likelihood",
            "observation_variance",
            "latent_representation",
            "projection_dim",
            "projection_hidden_dims",
            "temperature",
            "contrastive_weight",
            "condition_projection",
        )
        return cls(**{key: config[key] for key in keys})

    def project_payload(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.payload_projector_body(x)
        return F.normalize(self.payload_projector_out(hidden), dim=1, eps=1e-8)

    @staticmethod
    def project_conditions(candidate_conditions: torch.Tensor) -> torch.Tensor:
        return F.normalize(candidate_conditions, dim=1, eps=1e-8)

    def contrastive_logits(
        self, x: torch.Tensor, candidate_conditions: torch.Tensor
    ) -> torch.Tensor:
        payload_projection = self.project_payload(x)
        condition_projection = self.project_conditions(candidate_conditions)
        return payload_projection @ condition_projection.T / self.temperature

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        candidate_conditions: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(x, condition, sample=sample)
        if candidate_conditions is not None:
            payload_projection = self.project_payload(x)
            condition_projection = self.project_conditions(candidate_conditions)
            output["payload_projection"] = payload_projection
            output["contrastive_logits"] = (
                payload_projection @ condition_projection.T / self.temperature
            )
        return output

    def loss(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        target_condition_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = super().loss(output, target)
        if target_condition_indices is None:
            return losses
        if "contrastive_logits" not in output:
            raise ValueError("candidate_conditions are required when computing contrastive loss.")

        targets = target_condition_indices.long()
        contrastive_per_sample = F.cross_entropy(
            output["contrastive_logits"], targets, reduction="none"
        )
        contrastive = contrastive_per_sample.mean()
        weighted_contrastive = self.contrastive_weight * contrastive
        total_per_sample = losses["negative_elbo_per_sample"] + (
            self.contrastive_weight * contrastive_per_sample
        )
        accuracy = (output["contrastive_logits"].argmax(dim=1) == targets).float().mean()
        total = total_per_sample.mean()
        losses.update(
            {
                "loss": total,
                "total_loss": total,
                "total_per_sample": total_per_sample,
                "contrastive_loss": contrastive,
                "weighted_contrastive_loss": weighted_contrastive,
                "contrastive_accuracy": accuracy,
                "contrastive_per_sample": contrastive_per_sample,
            }
        )
        return losses
