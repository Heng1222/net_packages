from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .geometry import project_tactic_basis


def activation_layer(name: str) -> nn.Module:
    mapping: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU,
                                           "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
    if name.lower() not in mapping: raise ValueError(f"Unsupported activation: {name}")
    return mapping[name.lower()]()


def build_mlp(input_dim: int, hidden_dims: list[int], activation: str, dropout: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []; current = int(input_dim)
    for hidden in map(int, hidden_dims):
        layers.extend((nn.Linear(current, hidden), activation_layer(activation)))
        if dropout > 0: layers.append(nn.Dropout(float(dropout)))
        current = hidden
    return nn.Sequential(*layers), current


class GateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], condition_count: int = 15,
                 activation: str = "gelu", dropout: float = 0.1) -> None:
        super().__init__(); self.hidden_dims = list(map(int, hidden_dims)); self.activation_name = str(activation)
        self.dropout_rate = float(dropout); self.body, output_dim = build_mlp(input_dim, self.hidden_dims, activation, dropout)
        self.output = nn.Linear(output_dim, condition_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.output(self.body(x))


class ConceptProjector(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, activation: str = "gelu") -> None:
        super().__init__(); self.input = nn.Linear(dimension, hidden_dim); self.activation = activation_layer(activation)
        self.output = nn.Linear(hidden_dim, dimension); nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.output(self.activation(self.input(values)))


class UCDCVAE(nn.Module):
    condition_count = 15

    def __init__(self, common: torch.Tensor, tactic_basis: torch.Tensor, input_dim: int = 768,
                 residual_dim: int = 16, gate_hidden_dims: list[int] | None = None,
                 residual_hidden_dims: list[int] | None = None,
                 residual_up_hidden_dims: list[int] | None = None,
                 decoder_hidden_dims: list[int] | None = None,
                 concept_projector_hidden_dim: int = 256, activation: str = "gelu",
                 dropout: float = 0.1, geometry_variant: str = "full_orthogonal",
                 geometry_epsilon: float = 1e-6, alignment_temperature: float = 0.1) -> None:
        super().__init__()
        if tuple(common.shape) != (input_dim,) or tuple(tactic_basis.shape) != (14, input_dim):
            raise ValueError("Expected common [D] and tactic_basis [14,D].")
        if residual_dim != 16: raise ValueError("UCD-CVAE v2.1 residual dimension must be 16.")
        self.input_dim = int(input_dim); self.residual_dim = int(residual_dim)
        self.geometry_variant = geometry_variant; self.geometry_epsilon = float(geometry_epsilon)
        self.alignment_temperature = float(alignment_temperature)
        self.register_buffer("common", F.normalize(common.float(), dim=0))
        self.register_buffer("base_tactics", tactic_basis.float())
        self.gate_encoder = GateEncoder(input_dim, gate_hidden_dims or [512, 128], 15, activation, dropout)
        self.residual_encoder, encoded = build_mlp(input_dim, residual_hidden_dims or [512, 128], activation, dropout)
        self.residual_mu = nn.Linear(encoded, residual_dim); self.residual_logvar = nn.Linear(encoded, residual_dim)
        self.residual_up, up_dim = build_mlp(residual_dim, residual_up_hidden_dims or [128, 512], activation, dropout)
        self.residual_out = nn.Linear(up_dim, input_dim)
        self.concept_projector = ConceptProjector(input_dim, concept_projector_hidden_dim, activation)
        self.decoder, decoded = build_mlp(input_dim, decoder_hidden_dims or [1024, 768], activation, dropout)
        self.decoder_out = nn.Linear(decoded, input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any], common: torch.Tensor, tactics: torch.Tensor) -> "UCDCVAE":
        keys = ("input_dim", "residual_dim", "gate_hidden_dims", "residual_hidden_dims",
                "residual_up_hidden_dims", "decoder_hidden_dims", "concept_projector_hidden_dim",
                "activation", "dropout", "geometry_variant", "geometry_epsilon", "alignment_temperature")
        return cls(common, tactics, **{key: config[key] for key in keys if key in config})

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def projected_basis(self) -> torch.Tensor:
        projected = self.concept_projector(self.base_tactics)
        tactics = project_tactic_basis(projected, self.common, self.geometry_variant, self.geometry_epsilon)
        return torch.cat((self.common.unsqueeze(0), tactics), dim=0)

    def encode_residual(self, x: torch.Tensor, sample: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.residual_encoder(x); mu = self.residual_mu(hidden)
        logvar = self.residual_logvar(hidden).clamp(min=-30.0, max=20.0)
        return (self.reparameterize(mu, logvar) if sample else mu), mu, logvar

    @staticmethod
    def orthogonalize_residual(h_res: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        q, _ = torch.linalg.qr(basis.T, mode="reduced")
        return h_res - (h_res @ q) @ q.T

    def forward(self, x: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        basis = self.projected_basis(); logits = self.gate_encoder(x); gates = torch.sigmoid(logits)
        z, mu, logvar = self.encode_residual(x, sample)
        h_res = self.residual_out(self.residual_up(z)); h_res_perp = self.orthogonalize_residual(h_res, basis)
        common_component = gates[:, :1] * basis[:1]
        tactic_component = gates[:, 1:] @ basis[1:]
        concept = common_component + tactic_component; latent = concept + h_res_perp
        reconstructed = self.decoder_out(self.decoder(latent))
        return {"gate_logits": logits, "gates": gates, "mu_r": mu, "logvar_r": logvar,
                "z_r": z, "projected_basis": basis, "common_component": common_component,
                "tactic_component": tactic_component, "concept_component": concept,
                "h_res": h_res, "h_res_perp": h_res_perp, "h_latent": latent, "x_hat": reconstructed}


def compute_losses(output: dict[str, torch.Tensor], target: torch.Tensor,
                   alignment_temperature: float = 0.1) -> dict[str, torch.Tensor]:
    reconstruction = (1.0 - F.cosine_similarity(target, output["x_hat"], dim=1, eps=1e-8)).mean()
    kl = (-0.5 * torch.sum(1.0 + output["logvar_r"] - output["mu_r"].pow(2)
                           - output["logvar_r"].exp(), dim=1)).mean()
    sparse = output["gates"].abs().sum(dim=1).mean()
    normalized_x = F.normalize(target, dim=1, eps=1e-8)
    normalized_basis = F.normalize(output["projected_basis"].detach(), dim=1, eps=1e-8)
    similarities = normalized_x @ normalized_basis.T
    pseudo_targets = torch.sigmoid(similarities / float(alignment_temperature)).detach()
    align = F.binary_cross_entropy_with_logits(output["gate_logits"], pseudo_targets)
    return {"reconstruction_loss": reconstruction, "kl_loss": kl, "sparse_loss": sparse,
            "align_loss": align, "alignment_targets": pseudo_targets,
            "recon_cosine": 1.0 - reconstruction}
