from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def activation_layer(name: str) -> nn.Module:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
    }
    if name.lower() not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(mapping)}")
    return mapping[name.lower()]()


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: str,
    dropout: float,
    batch_norm: bool,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current = input_dim
    for hidden in hidden_dims:
        if hidden <= 0:
            raise ValueError("All hidden dimensions must be positive.")
        layers.append(nn.Linear(current, hidden))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden
    return nn.Sequential(*layers), current


def reconstruction_per_sample(prediction: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(prediction, target, reduction="none").mean(dim=1)
    if loss_type == "l1":
        return F.l1_loss(prediction, target, reduction="none").mean(dim=1)
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(prediction, target, dim=1, eps=1e-8)
    raise ValueError("reconstruction_loss must be mse, l1, or cosine.")
