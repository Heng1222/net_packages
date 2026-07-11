from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class GradientReversal(nn.Module):
    def __init__(self, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = float(strength)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(value, self.strength)


def activation_layer(name: str) -> nn.Module:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported activation: {name}")
    return mapping[key]()


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    activation: str,
    dropout: float,
    batch_norm: bool,
) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.append(nn.Linear(current, hidden))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(activation_layer(activation))
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        current = hidden
    return nn.Sequential(*layers), current


class DisentangledConditionalVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        residual_dim: int,
        condition_count: int,
        condition_dim: int,
        encoder_hidden_dims: list[int],
        decoder_hidden_dims: list[int],
        behavior_projector_hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        activation: str = "relu",
        observation_variance: float = 1.0,
        temperature: float = 0.1,
        behavior_temperature: float = 0.1,
        residual_adversary_strength: float = 1.0,
        utility_margin: float = 0.5,
        residual_margin: float = 0.5,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if min(input_dim, residual_dim, condition_count, condition_dim) <= 0:
            raise ValueError("All core model dimensions must be positive.")
        if observation_variance <= 0:
            raise ValueError("observation_variance must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if behavior_temperature <= 0:
            raise ValueError("behavior_temperature must be positive.")
        self.input_dim = int(input_dim)
        self.residual_dim = int(residual_dim)
        self.condition_count = int(condition_count)
        self.condition_dim = int(condition_dim)
        self.observation_variance = float(observation_variance)
        self.temperature = float(temperature)
        self.behavior_temperature = float(behavior_temperature)
        self.residual_adversary_strength = float(residual_adversary_strength)
        self.utility_margin = float(utility_margin)
        self.residual_margin = float(residual_margin)
        self.behavior_projector_hidden_dims = list(behavior_projector_hidden_dims or [])
        self.weights = {
            "reconstruction": 1.0,
            "kl": 1.0,
            "decorrelation": 0.0,
            "sparse": 0.0,
            "gate_entropy": 0.0,
            "utility": 0.0,
            "residual_constraint": 0.1,
            "behavior_infonce": 1.0,
            "residual_adversary": 0.0,
        }
        if weights:
            self.weights.update({str(key): float(value) for key, value in weights.items()})

        condition_flat_dim = self.condition_count * self.condition_dim
        self.encoder_body, encoded_dim = build_mlp(
            self.input_dim + condition_flat_dim, encoder_hidden_dims, activation, dropout, batch_norm
        )
        self.h_mu = nn.Linear(encoded_dim, self.residual_dim)
        self.h_logvar = nn.Linear(encoded_dim, self.residual_dim)
        self.residual_reversal = GradientReversal(self.residual_adversary_strength)
        self.residual_adversary = nn.Linear(self.residual_dim, self.condition_count)
        self.behavior_projector_body, behavior_projector_dim = build_mlp(
            self.input_dim,
            self.behavior_projector_hidden_dims,
            activation,
            dropout,
            batch_norm,
        )
        self.behavior_projector_out = nn.Linear(behavior_projector_dim, self.condition_dim)

        decoder_input = self.residual_dim + condition_flat_dim
        self.decoder_body, decoded_dim = build_mlp(
            decoder_input, decoder_hidden_dims, activation, dropout, batch_norm
        )
        self.decoder_out = nn.Linear(decoded_dim, self.input_dim)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DisentangledConditionalVAE":
        keys = (
            "input_dim",
            "residual_dim",
            "condition_count",
            "condition_dim",
            "encoder_hidden_dims",
            "decoder_hidden_dims",
            "behavior_projector_hidden_dims",
            "dropout",
            "batch_norm",
            "activation",
            "observation_variance",
            "temperature",
            "behavior_temperature",
            "residual_adversary_strength",
            "utility_margin",
            "residual_margin",
            "weights",
        )
        return cls(**{key: config[key] for key in keys if key in config})

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _expand_conditions(self, condition_embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
        if condition_embeddings.ndim == 2:
            if condition_embeddings.shape != (self.condition_count, self.condition_dim):
                raise ValueError(
                    "condition_embeddings must have shape "
                    f"({self.condition_count}, {self.condition_dim}); got {tuple(condition_embeddings.shape)}"
                )
            return condition_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        if condition_embeddings.ndim == 3:
            if condition_embeddings.shape[1:] != (self.condition_count, self.condition_dim):
                raise ValueError(
                    "batched condition_embeddings must have trailing shape "
                    f"({self.condition_count}, {self.condition_dim}); got {tuple(condition_embeddings.shape)}"
                )
            if condition_embeddings.shape[0] != batch_size:
                raise ValueError("Batched condition_embeddings batch size does not match x.")
            return condition_embeddings
        raise ValueError("condition_embeddings must be 2D or 3D.")

    def project_payload_behavior(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.behavior_projector_body(x)
        return F.normalize(self.behavior_projector_out(hidden), dim=1, eps=1e-8)

    def behavior_condition_logits(
        self,
        behavior_query: torch.Tensor,
        condition_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        conditions = self._expand_conditions(condition_embeddings, len(behavior_query))
        normalized_conditions = F.normalize(conditions, dim=2, eps=1e-8)
        return torch.bmm(
            normalized_conditions,
            behavior_query.unsqueeze(2),
        ).squeeze(2)

    def encode(
        self,
        x: torch.Tensor,
        condition_embeddings: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        conditions = self._expand_conditions(condition_embeddings, len(x))
        encoder_input = torch.cat((x, conditions.flatten(start_dim=1)), dim=1)
        hidden = self.encoder_body(encoder_input)
        h_mu = self.h_mu(hidden)
        h_logvar = self.h_logvar(hidden).clamp(min=-30.0, max=20.0)
        h = self.reparameterize(h_mu, h_logvar) if sample else h_mu
        behavior_query = self.project_payload_behavior(x)
        gate_logits = self.behavior_condition_logits(behavior_query, conditions)
        gates = torch.sigmoid(gate_logits / self.temperature)
        return {
            "h": h,
            "h_mu": h_mu,
            "h_logvar": h_logvar,
            "conditions": conditions,
            "behavior_query": behavior_query,
            "behavior_logits": gate_logits / self.behavior_temperature,
            "gate_logits": gate_logits,
            "gates": gates,
        }

    def decode(self, h: torch.Tensor, condition_embeddings: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        conditions = self._expand_conditions(condition_embeddings, len(h))
        gated = conditions * gates.unsqueeze(-1)
        decoder_input = torch.cat((h, gated.flatten(start_dim=1)), dim=1)
        hidden = self.decoder_body(decoder_input)
        return self.decoder_out(hidden)

    def forward(
        self,
        x: torch.Tensor,
        condition_embeddings: torch.Tensor,
        sample: bool = True,
    ) -> dict[str, torch.Tensor]:
        output = self.encode(x, condition_embeddings, sample=sample)
        output["x_recon"] = self.decode(output["h"], output["conditions"], output["gates"])
        output["residual_adversary_logits"] = self.residual_adversary(
            self.residual_reversal(output["h_mu"])
        )
        return output

    def semantic_summary(self, condition_embeddings: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        conditions = self._expand_conditions(condition_embeddings, len(gates))
        weighted = conditions * gates.unsqueeze(-1)
        denom = gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return weighted.sum(dim=1) / denom

    def ablation_deltas(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        full_mse = F.mse_loss(output["x_recon"], target, reduction="none").mean(dim=1)
        deltas: list[torch.Tensor] = []
        for component_index in range(self.condition_count):
            ablated_gates = output["gates"].clone()
            ablated_gates[:, component_index] = 0.0
            ablated = self.decode(output["h"], output["conditions"], ablated_gates)
            ablated_mse = F.mse_loss(ablated, target, reduction="none").mean(dim=1)
            deltas.append(ablated_mse - full_mse)
        return torch.stack(deltas, dim=1)

    def auxiliary_reconstructions(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        zero_gates = torch.zeros_like(output["gates"])
        zero_h = torch.zeros_like(output["h"])
        return {
            "h_only": self.decode(output["h"], output["conditions"], zero_gates),
            "c_only": self.decode(zero_h, output["conditions"], output["gates"]),
        }

    def loss(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        behavior_targets: torch.Tensor | None = None,
        compute_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        squared_error = (output["x_recon"] - target).pow(2)
        recon_nll_per_sample = 0.5 * (
            squared_error / self.observation_variance
            + math.log(2.0 * math.pi * self.observation_variance)
        ).sum(dim=1)
        recon_nll = recon_nll_per_sample.mean()
        recon_mse_per_sample = squared_error.mean(dim=1)
        recon_mse = recon_mse_per_sample.mean()
        kl_per_sample = -0.5 * torch.sum(
            1.0 + output["h_logvar"] - output["h_mu"].pow(2) - output["h_logvar"].exp(), dim=1
        )
        kl = kl_per_sample.mean()

        normalized_components = F.normalize(output["conditions"], dim=2, eps=1e-8)
        cosine = torch.bmm(normalized_components, normalized_components.transpose(1, 2))
        offdiag = 1.0 - torch.eye(self.condition_count, device=target.device).unsqueeze(0)
        gate_pairs = output["gates"].unsqueeze(2) * output["gates"].unsqueeze(1)
        decor_num = (cosine.pow(2) * offdiag * gate_pairs).sum()
        decor_den = (offdiag * gate_pairs).sum().clamp_min(1e-6)
        decorrelation = decor_num / decor_den

        sparse = output["gates"].mean()
        gate_probs = output["gates"].clamp(min=1e-6, max=1.0 - 1e-6)
        gate_entropy = -(
            gate_probs * gate_probs.log()
            + (1.0 - gate_probs) * (1.0 - gate_probs).log()
        ).mean()

        zero = target.new_tensor(0.0)
        if self.weights["utility"] > 0.0 or compute_diagnostics:
            deltas = self.ablation_deltas(output, target)
        else:
            deltas = target.new_zeros((len(target), self.condition_count))
        if self.weights["utility"] > 0.0:
            utility_margin = F.relu(self.utility_margin - deltas)
            utility = (utility_margin * output["gates"]).sum() / output["gates"].sum().clamp_min(1e-6)
        else:
            utility = zero

        if self.weights["residual_constraint"] > 0.0:
            h_only = self.auxiliary_reconstructions(output)["h_only"]
            h_only_mse = F.mse_loss(h_only, target, reduction="none").mean(dim=1)
            residual_constraint = F.relu(
                self.residual_margin - (h_only_mse - recon_mse_per_sample)
            ).mean()
        else:
            residual_constraint = zero

        behavior_infonce = zero
        residual_adversary = zero
        behavior_accuracy = zero
        residual_adversary_accuracy = zero
        behavior_labeled_count = target.new_tensor(0.0)
        if behavior_targets is not None:
            labels = behavior_targets.to(device=target.device).long()
            valid = labels >= 0
            valid_count = int(valid.sum().item())
            behavior_labeled_count = target.new_tensor(float(valid_count))
            if valid_count > 0:
                logits = output["behavior_logits"][valid]
                behavior_infonce = F.cross_entropy(logits, labels[valid])
                behavior_accuracy = (logits.argmax(dim=1) == labels[valid]).float().mean()

                adversary_logits = output["residual_adversary_logits"][valid]
                residual_adversary = F.cross_entropy(adversary_logits, labels[valid])
                residual_adversary_accuracy = (
                    adversary_logits.argmax(dim=1) == labels[valid]
                ).float().mean()

        total = (
            self.weights["reconstruction"] * recon_nll
            + self.weights["kl"] * kl
            + self.weights["decorrelation"] * decorrelation
            + self.weights["sparse"] * sparse
            + self.weights["gate_entropy"] * gate_entropy
            + self.weights["utility"] * utility
            + self.weights["residual_constraint"] * residual_constraint
            + self.weights["behavior_infonce"] * behavior_infonce
            + self.weights["residual_adversary"] * residual_adversary
        )
        return {
            "loss": total,
            "total_loss": total,
            "recon_nll": recon_nll,
            "recon_mse": recon_mse,
            "kl_loss": kl,
            "decorrelation_loss": decorrelation,
            "sparse_loss": sparse,
            "gate_entropy_loss": gate_entropy,
            "utility_loss": utility,
            "residual_constraint_loss": residual_constraint,
            "behavior_infonce_loss": behavior_infonce,
            "residual_adversary_loss": residual_adversary,
            "behavior_infonce_accuracy": behavior_accuracy,
            "residual_adversary_accuracy": residual_adversary_accuracy,
            "behavior_labeled_count": behavior_labeled_count,
            "recon_nll_per_sample": recon_nll_per_sample,
            "recon_mse_per_sample": recon_mse_per_sample,
            "kl_per_sample": kl_per_sample,
            "ablation_delta_mse": deltas,
        }
