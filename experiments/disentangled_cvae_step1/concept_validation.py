from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from .model import DisentangledConditionalVAE


@dataclass(slots=True)
class ConceptValidationResult:
    passed: bool
    macro_f1: float
    gate_target_correlation: float
    full_reconstruction_mse: float
    h_only_reconstruction_mse: float
    condition_reconstruction_gain: float
    shuffled_target_macro_f1: float
    train_rows: int
    test_rows: int
    seed: int


def _make_synthetic_mixtures(
    rows: int,
    input_dim: int,
    condition_count: int,
    residual_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    gates = (rng.random((rows, condition_count)) < 0.35).astype(np.float32)
    empty = gates.sum(axis=1) == 0
    gates[empty, rng.integers(0, condition_count, size=int(empty.sum()))] = 1.0

    condition_effects = rng.normal(size=(condition_count, input_dim)).astype(np.float32)
    condition_effects /= np.linalg.norm(condition_effects, axis=1, keepdims=True)
    residual_effects = rng.normal(size=(residual_dim, input_dim)).astype(np.float32)
    residual_effects /= np.linalg.norm(residual_effects, axis=1, keepdims=True)
    residual = rng.normal(size=(rows, residual_dim)).astype(np.float32)
    noise = 0.04 * rng.normal(size=(rows, input_dim)).astype(np.float32)
    x = 2.5 * gates @ condition_effects + 0.45 * residual @ residual_effects + noise

    condition_geometry = rng.normal(size=(input_dim, condition_count)).astype(np.float32)
    condition_geometry, _ = np.linalg.qr(condition_geometry)
    conditions = condition_geometry[:, :condition_count].T.astype(np.float32)
    return x.astype(np.float32), gates, conditions


def run_concept_validation(
    seed: int = 42,
    epochs: int = 80,
    train_rows: int = 768,
    test_rows: int = 256,
    device: str = "cpu",
) -> ConceptValidationResult:
    """Check whether the gate/H architecture can recover known condition mixtures.

    This is a capability test on identifiable synthetic data, not evidence that the
    real payload embeddings contain identifiable MITRE tactic mixtures.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_dim = 16
    condition_count = 4
    residual_dim = 3
    x, targets, conditions = _make_synthetic_mixtures(
        train_rows + test_rows,
        input_dim,
        condition_count,
        residual_dim,
        seed,
    )
    mean = x[:train_rows].mean(axis=0, keepdims=True)
    scale = x[:train_rows].std(axis=0, keepdims=True).clip(min=1e-5)
    x = (x - mean) / scale

    torch_device = torch.device(device)
    x_tensor = torch.from_numpy(x).to(torch_device)
    target_tensor = torch.from_numpy(targets).to(torch_device)
    condition_tensor = torch.from_numpy(conditions).to(torch_device)
    model = DisentangledConditionalVAE(
        input_dim=input_dim,
        residual_dim=residual_dim,
        condition_count=condition_count,
        condition_dim=input_dim,
        encoder_hidden_dims=[64, 32],
        decoder_hidden_dims=[64, 32],
        behavior_projector_hidden_dims=[32],
        dropout=0.0,
        batch_norm=False,
        temperature=0.15,
        behavior_temperature=0.15,
        utility_margin=0.05,
        residual_margin=0.10,
        weights={
            "reconstruction": 0.0,
            "kl": 0.0,
            "decorrelation": 0.0,
            "sparse": 0.0,
            "gate_entropy": 0.0,
            "utility": 0.0,
            "residual_constraint": 0.0,
            "behavior_infonce": 0.0,
            "residual_adversary": 0.0,
        },
    ).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(train_rows, generator=generator)
        for indices in order.split(64):
            xb = x_tensor[indices]
            yb = target_tensor[indices]
            output = model(xb, condition_tensor, sample=True)
            reconstruction = F.mse_loss(output["x_recon"], xb)
            gate_supervision = F.binary_cross_entropy(output["gates"], yb)
            kl = -0.5 * (
                1.0
                + output["h_logvar"]
                - output["h_mu"].pow(2)
                - output["h_logvar"].exp()
            ).sum(dim=1).mean()
            h_only = model.decode(
                output["h"], output["conditions"], torch.zeros_like(output["gates"])
            )
            h_only_mse = F.mse_loss(h_only, xb)
            condition_use = F.relu(0.10 - (h_only_mse - reconstruction))
            loss = reconstruction + 2.0 * gate_supervision + 0.02 * kl + condition_use
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_x = x_tensor[train_rows:]
        output = model(test_x, condition_tensor, sample=False)
        probabilities = output["gates"].cpu().numpy()
        predictions = probabilities >= 0.5
        truth = targets[train_rows:].astype(bool)
        full_mse = float(F.mse_loss(output["x_recon"], test_x))
        h_only = model.decode(
            output["h"], output["conditions"], torch.zeros_like(output["gates"])
        )
        h_only_mse = float(F.mse_loss(h_only, test_x))

    macro_f1 = float(f1_score(truth, predictions, average="macro", zero_division=0))
    correlation = float(np.corrcoef(probabilities.ravel(), truth.astype(np.float32).ravel())[0, 1])
    shuffled = truth.copy()
    np.random.default_rng(seed + 1).shuffle(shuffled, axis=0)
    shuffled_f1 = float(f1_score(shuffled, predictions, average="macro", zero_division=0))
    gain = h_only_mse - full_mse
    passed = bool(macro_f1 >= 0.80 and correlation >= 0.75 and gain >= 0.05)
    return ConceptValidationResult(
        passed=passed,
        macro_f1=macro_f1,
        gate_target_correlation=correlation,
        full_reconstruction_mse=full_mse,
        h_only_reconstruction_mse=h_only_mse,
        condition_reconstruction_gain=gain,
        shuffled_target_macro_f1=shuffled_f1,
        train_rows=train_rows,
        test_rows=test_rows,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gate/H recovery on known synthetic mixtures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_concept_validation(seed=args.seed, epochs=args.epochs)
    payload = asdict(result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
