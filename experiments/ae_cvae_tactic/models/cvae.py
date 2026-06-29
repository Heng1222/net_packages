from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .common import build_mlp, reconstruction_per_sample


class ConditionalVAE(nn.Module):
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
    ) -> None:
        super().__init__()
        if min(input_dim, condition_dim, latent_dim) <= 0:
            raise ValueError("input_dim, condition_dim, and latent_dim must be positive.")
        if objective != "elbo":
            raise ValueError("objective must be 'elbo'.")
        if likelihood != "gaussian":
            raise ValueError("likelihood must be 'gaussian'.")
        if reconstruction_loss != "mse":
            raise ValueError("Gaussian ELBO requires reconstruction_loss='mse'.")
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive.")
        if latent_representation not in {"mu", "z"}:
            raise ValueError("latent_representation must be 'mu' or 'z'.")
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.reconstruction_loss = reconstruction_loss
        self.objective = objective
        self.likelihood = likelihood
        self.observation_variance = float(observation_variance)
        self.latent_representation = latent_representation
        self.encoder_body, encoded_dim = build_mlp(
            input_dim + condition_dim, hidden_dims, activation, dropout, batch_norm
        )
        self.mu = nn.Linear(encoded_dim, latent_dim)
        self.logvar = nn.Linear(encoded_dim, latent_dim)
        self.decoder_body, decoded_dim = build_mlp(
            latent_dim + condition_dim, list(reversed(hidden_dims)), activation, dropout, batch_norm
        )
        self.decoder_out = nn.Linear(decoded_dim, input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ConditionalVAE":
        keys = (
            "input_dim", "condition_dim", "latent_dim", "hidden_dims", "dropout", "batch_norm",
            "activation", "reconstruction_loss", "objective", "likelihood", "observation_variance",
            "latent_representation"
        )
        return cls(**{key: config[key] for key in keys})

    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder_body(torch.cat((x, condition), dim=1))
        return self.mu(hidden), self.logvar(hidden).clamp(min=-30.0, max=20.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_body(torch.cat((latent, condition), dim=1))
        return self.decoder_out(hidden)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor, sample: bool = True
    ) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar) if sample else mu
        return {"z": z, "mu": mu, "logvar": logvar, "x_recon": self.decode(z, condition)}

    def loss(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> dict[str, torch.Tensor]:
        recon_mse_per_sample = reconstruction_per_sample(output["x_recon"], target, "mse")
        squared_error = (output["x_recon"] - target).pow(2)
        recon_nll_per_sample = 0.5 * (
            squared_error / self.observation_variance
            + math.log(2.0 * math.pi * self.observation_variance)
        ).sum(dim=1)
        kl_per_sample = -0.5 * torch.sum(
            1.0 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp(), dim=1
        )
        negative_elbo_per_sample = recon_nll_per_sample + kl_per_sample
        elbo_per_sample = -negative_elbo_per_sample
        recon_nll = recon_nll_per_sample.mean()
        recon_mse = recon_mse_per_sample.mean()
        kl = kl_per_sample.mean()
        negative_elbo = negative_elbo_per_sample.mean()
        elbo = elbo_per_sample.mean()
        return {
            "loss": negative_elbo,
            "total_loss": negative_elbo,
            "negative_elbo": negative_elbo,
            "elbo": elbo,
            "recon_loss": recon_nll,
            "recon_nll": recon_nll,
            "recon_mse": recon_mse,
            "kl_loss": kl,
            "recon_per_sample": recon_nll_per_sample,
            "recon_nll_per_sample": recon_nll_per_sample,
            "recon_mse_per_sample": recon_mse_per_sample,
            "kl_per_sample": kl_per_sample,
            "negative_elbo_per_sample": negative_elbo_per_sample,
            "elbo_per_sample": elbo_per_sample,
        }

    def representation(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return output[self.latent_representation]
