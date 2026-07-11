from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _activation(name: str) -> nn.Module:
    mapping = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}
    if name.lower() not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[name.lower()]()


def build_mlp(input_dim: int, hidden_dims: list[int], activation: str, dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(current, int(hidden)), _activation(activation)])
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        current = int(hidden)
    return nn.Sequential(*layers), current


class GoldenConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        condition_count: int,
        residual_dim: int,
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
        projector_hidden_dims: list[int],
        dropout: float = 0.0,
        activation: str = "gelu",
        temperature: float = 0.15,
        **_: Any,
    ) -> None:
        super().__init__()
        self.condition_count = int(condition_count)
        self.temperature = float(temperature)
        self.encoder, encoder_dim = build_mlp(input_dim, encoder_hidden_dims, activation, dropout)
        self.h_mu = nn.Linear(encoder_dim, residual_dim)
        self.h_logvar = nn.Linear(encoder_dim, residual_dim)
        self.projector, projector_dim = build_mlp(input_dim, projector_hidden_dims, activation, dropout)
        self.projector_out = nn.Linear(projector_dim, condition_dim)
        self.decoder, decoder_dim = build_mlp(
            residual_dim + condition_dim, decoder_hidden_dims, activation, dropout
        )
        self.decoder_out = nn.Linear(decoder_dim, input_dim)

    def encode(self, x: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.h_mu(hidden)
        logvar = self.h_logvar(hidden).clamp(-30.0, 20.0)
        h = mu
        if sample:
            h = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return h, mu, logvar

    def predict_gates(self, x: torch.Tensor, conditions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = F.normalize(self.projector_out(self.projector(x)), dim=1, eps=1e-8)
        normalized_conditions = F.normalize(conditions, dim=1, eps=1e-8)
        logits = query @ normalized_conditions.T / self.temperature
        return torch.sigmoid(logits), logits

    @staticmethod
    def condition_summary(gates: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return gates @ conditions

    def decode(self, h: torch.Tensor, gates: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        summary = self.condition_summary(gates, conditions)
        return self.decoder_out(self.decoder(torch.cat((h, summary), dim=1)))

    def forward(
        self,
        x: torch.Tensor,
        conditions: torch.Tensor,
        gates_override: torch.Tensor | None = None,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        h, mu, logvar = self.encode(x, sample)
        predicted_gates, gate_logits = self.predict_gates(x, conditions)
        used_gates = predicted_gates if gates_override is None else gates_override
        return {
            "h": h,
            "h_mu": mu,
            "h_logvar": logvar,
            "predicted_gates": predicted_gates,
            "gate_logits": gate_logits,
            "used_gates": used_gates,
            "x_recon": self.decode(h, used_gates, conditions),
        }


class PayloadClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        class_count: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.body, output_dim = build_mlp(input_dim, hidden_dims, activation, dropout)
        self.output = nn.Linear(output_dim, class_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(x))
