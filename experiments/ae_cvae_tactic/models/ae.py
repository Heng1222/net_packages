from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .common import build_mlp, reconstruction_per_sample


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        reconstruction_loss: str = "mse",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or latent_dim <= 0:
            raise ValueError("input_dim and latent_dim must be positive.")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        self.encoder_body, encoded_dim = build_mlp(input_dim, hidden_dims, activation, dropout, batch_norm)
        self.encoder_out = nn.Linear(encoded_dim, latent_dim)
        self.decoder_body, decoded_dim = build_mlp(
            latent_dim, list(reversed(hidden_dims)), activation, dropout, batch_norm
        )
        self.decoder_out = nn.Linear(decoded_dim, input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AutoEncoder":
        return cls(**{key: config[key] for key in (
            "input_dim", "latent_dim", "hidden_dims", "dropout", "batch_norm", "activation", "reconstruction_loss"
        )})

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_out(self.encoder_body(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder_out(self.decoder_body(latent))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(x)
        return {"latent": latent, "x_recon": self.decode(latent)}

    def loss(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        per_sample = reconstruction_per_sample(output["x_recon"], target, self.reconstruction_loss)
        value = per_sample.mean()
        return {"loss": value, "recon_loss": value, "recon_per_sample": per_sample}
