from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def activation_layer(name: str) -> nn.Module:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh,
    }
    if name.lower() not in mapping: raise ValueError(f"Unsupported activation: {name}")
    return mapping[name.lower()]()


def build_mlp(input_dim: int, hidden_dims: list[int], activation: str, dropout: float,
              batch_norm: bool = False) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []; current = int(input_dim)
    for hidden in map(int, hidden_dims):
        layers.append(nn.Linear(current, hidden))
        if batch_norm: layers.append(nn.BatchNorm1d(hidden))
        layers.append(activation_layer(activation))
        if dropout > 0: layers.append(nn.Dropout(float(dropout)))
        current = hidden
    return nn.Sequential(*layers), current


class CenterAugmentedCVAE(nn.Module):
    """CVAE with deterministic semantic gates and an additive condition decoder."""

    def __init__(self, input_dim: int, residual_dim: int, condition_count: int,
                 condition_dim: int, encoder_hidden_dims: list[int],
                 residual_decoder_hidden_dims: list[int], dropout: float = 0.0,
                 batch_norm: bool = False, activation: str = "relu",
                 gate_temperature: float = 0.1) -> None:
        super().__init__()
        if min(input_dim, residual_dim, condition_count, condition_dim) <= 0:
            raise ValueError("All dimensions must be positive.")
        if input_dim != condition_dim:
            raise ValueError("Additive reconstruction requires input_dim == condition_dim.")
        if gate_temperature <= 0: raise ValueError("gate_temperature must be positive.")
        self.input_dim = int(input_dim); self.residual_dim = int(residual_dim)
        self.condition_count = int(condition_count); self.condition_dim = int(condition_dim)
        self.gate_temperature = float(gate_temperature)
        encoder_input_dim = self.input_dim + self.condition_count * self.condition_dim
        self.encoder, encoded_dim = build_mlp(
            encoder_input_dim, encoder_hidden_dims, activation, dropout, batch_norm
        )
        self.z_mu = nn.Linear(encoded_dim, self.residual_dim)
        self.z_logvar = nn.Linear(encoded_dim, self.residual_dim)
        self.residual_decoder, decoded_dim = build_mlp(
            self.residual_dim, residual_decoder_hidden_dims, activation, dropout, batch_norm
        )
        self.residual_out = nn.Linear(decoded_dim, self.input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CenterAugmentedCVAE":
        return cls(**{key: config[key] for key in (
            "input_dim", "residual_dim", "condition_count", "condition_dim",
            "encoder_hidden_dims", "residual_decoder_hidden_dims", "dropout",
            "batch_norm", "activation", "gate_temperature",
        ) if key in config})

    def _validate_conditions(self, decode: torch.Tensor, gate: torch.Tensor) -> None:
        expected = (self.condition_count, self.condition_dim)
        if tuple(decode.shape) != expected or tuple(gate.shape) != expected:
            raise ValueError(f"Condition matrices must both have shape {expected}.")

    def semantic_gates(self, x: torch.Tensor, gate_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = F.normalize(x, dim=1, eps=1e-8)
        normalized_conditions = F.normalize(gate_matrix, dim=1, eps=1e-8)
        cosine = normalized_x @ normalized_conditions.T
        logits = cosine / self.gate_temperature
        return torch.sigmoid(logits), cosine

    def encode(self, x: torch.Tensor, decode_matrix: torch.Tensor,
               sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fixed = decode_matrix.flatten().unsqueeze(0).expand(len(x), -1)
        hidden = self.encoder(torch.cat((x, fixed), dim=1))
        mu = self.z_mu(hidden)
        logvar = self.z_logvar(hidden).clamp(min=-30.0, max=20.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return z, mu, logvar

    def residual_component(self, z: torch.Tensor) -> torch.Tensor:
        return self.residual_out(self.residual_decoder(z))

    @staticmethod
    def condition_component(gates: torch.Tensor, decode_matrix: torch.Tensor) -> torch.Tensor:
        return gates @ decode_matrix

    def decode(self, z: torch.Tensor, gates: torch.Tensor,
               decode_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.residual_component(z)
        condition = self.condition_component(gates, decode_matrix)
        return residual + condition, residual, condition

    def forward(self, x: torch.Tensor, decode_matrix: torch.Tensor,
                gate_matrix: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        self._validate_conditions(decode_matrix, gate_matrix)
        gates, cosine = self.semantic_gates(x, gate_matrix)
        z, mu, logvar = self.encode(x, decode_matrix, sample)
        reconstructed, residual, condition = self.decode(z, gates, decode_matrix)
        return {
            "x_recon": reconstructed, "residual_component": residual,
            "condition_component": condition, "gates": gates, "gate_cosine": cosine,
            "z": z, "z_mu": mu, "z_logvar": logvar,
        }


class PlainVAE(nn.Module):
    def __init__(self, input_dim: int, residual_dim: int, encoder_hidden_dims: list[int],
                 residual_decoder_hidden_dims: list[int], dropout: float = 0.0,
                 batch_norm: bool = False, activation: str = "relu", **_: Any) -> None:
        super().__init__(); self.input_dim = int(input_dim); self.residual_dim = int(residual_dim)
        self.encoder, encoded = build_mlp(input_dim, encoder_hidden_dims, activation, dropout, batch_norm)
        self.z_mu = nn.Linear(encoded, residual_dim); self.z_logvar = nn.Linear(encoded, residual_dim)
        self.decoder, decoded = build_mlp(residual_dim, residual_decoder_hidden_dims, activation, dropout, batch_norm)
        self.output = nn.Linear(decoded, input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PlainVAE":
        return cls(**config)

    def forward(self, x: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x); mu = self.z_mu(hidden)
        logvar = self.z_logvar(hidden).clamp(min=-30.0, max=20.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        reconstructed = self.output(self.decoder(z))
        return {"x_recon": reconstructed, "z": z, "z_mu": mu, "z_logvar": logvar}
