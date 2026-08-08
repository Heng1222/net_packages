from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from .embedders import build_text_embedder, normalize_rows


@dataclass(slots=True)
class ConditionGeometry:
    labels: list[str]
    tactic_labels: list[str]
    raw_tactics: np.ndarray
    common: np.ndarray
    common_removed: np.ndarray
    initial_tactics: np.ndarray
    variant: str
    metadata: dict[str, Any]

    @property
    def dimension(self) -> int: return int(self.raw_tactics.shape[1])


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    values = normalize_rows(matrix)
    return (values @ values.T).astype(np.float32)


def uncentered_common_component(raw_tactics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_tactics, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 14: raise ValueError("Exactly 14 tactic vectors are required.")
    _, singular_values, vh = np.linalg.svd(raw, full_matrices=False)
    common = vh[0]
    if float(common @ raw.mean(axis=0)) < 0: common = -common
    common /= np.linalg.norm(common)
    residual = raw - (raw @ common)[:, None] * common[None, :]
    if np.any(np.linalg.norm(residual, axis=1) < 1e-10):
        raise ValueError("A tactic collapses after common-component removal.")
    return common.astype(np.float32), residual.astype(np.float32), singular_values.astype(np.float32)


def _deterministic_complement(row_basis: np.ndarray, forbidden: np.ndarray | None) -> np.ndarray:
    constraints = row_basis
    if forbidden is not None:
        vector = np.asarray(forbidden, dtype=np.float64)
        vector = vector - constraints.T @ (constraints @ vector)
        if np.linalg.norm(vector) > 1e-10:
            constraints = np.vstack((constraints, vector / np.linalg.norm(vector)))
    dimension = constraints.shape[1]; best = None; best_norm = -1.0
    for index in range(dimension):
        candidate = np.zeros(dimension); candidate[index] = 1.0
        candidate -= constraints.T @ (constraints @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm > best_norm: best, best_norm = candidate, norm
    if best is None or best_norm < 1e-10: raise ValueError("No deterministic complement direction is available.")
    best /= best_norm
    first = np.flatnonzero(np.abs(best) > 1e-10)
    if len(first) and best[first[0]] < 0: best = -best
    return best


def symmetric_orthogonalize_numpy(matrix: np.ndarray, epsilon: float = 1e-6,
                                  forbidden: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64); gram = values @ values.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram); rank = int(np.sum(eigenvalues > epsilon))
    if rank == len(values) - 1:
        _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
        complement = _deterministic_complement(vh[:rank], forbidden)
        null_coefficients = eigenvectors[:, 0]
        positive = eigenvalues[eigenvalues > epsilon]
        completion_scale = float(np.sqrt(np.median(positive)))
        values = values + completion_scale * null_coefficients[:, None] * complement[None, :]
        gram = values @ values.T; eigenvalues, eigenvectors = np.linalg.eigh(gram)
    elif rank < len(values):
        raise ValueError(f"Tactic residual matrix is rank deficient by more than one (rank={rank}).")
    if float(eigenvalues.min()) <= epsilon:
        raise ValueError(f"Tactic residual rank completion failed (min eigenvalue={eigenvalues.min():.3g}).")
    inverse_sqrt = (eigenvectors * eigenvalues[None, :].clip(epsilon) ** -0.5) @ eigenvectors.T
    result = inverse_sqrt @ values
    signs = np.sign(np.sum(result * values, axis=1)); signs[signs == 0] = 1
    return (result * signs[:, None]).astype(np.float32)


def symmetric_orthogonalize_torch(matrix: torch.Tensor, epsilon: float = 1e-6,
                                  iterations: int = 12) -> torch.Tensor:
    """Differentiable polar/Löwdin factor via Newton-Schulz inverse square root.

    An eigendecomposition has undefined eigenvector gradients at the intentionally
    repeated eigenvalues of an orthonormal frame. Newton-Schulz is stable there.
    """
    gram = matrix @ matrix.T
    norm = torch.linalg.matrix_norm(gram, ord="fro").clamp_min(epsilon)
    y = gram / norm; identity = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    z = identity
    for _ in range(iterations):
        update = 0.5 * (3.0 * identity - z @ y)
        y = y @ update; z = update @ z
    result = (z / torch.sqrt(norm)) @ matrix
    signs = torch.sign((result * matrix).sum(dim=1, keepdim=True)).detach()
    return result * torch.where(signs == 0, torch.ones_like(signs), signs)


def project_tactic_basis(tactics: torch.Tensor, common: torch.Tensor, variant: str,
                         epsilon: float = 1e-6) -> torch.Tensor:
    residual = tactics - (tactics @ common).unsqueeze(1) * common.unsqueeze(0)
    if variant == "full_orthogonal": return symmetric_orthogonalize_torch(residual, epsilon)
    if variant == "common_removal_only": return F.normalize(residual, dim=1, eps=epsilon)
    raise ValueError(f"Unsupported geometry variant: {variant}")


def _condition_records(path: Path, field: str) -> tuple[list[str], list[str], dict[str, Any]]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    tactics = body.get("tactics") if isinstance(body, dict) else None
    if not isinstance(tactics, dict) or len(tactics) != 14:
        raise ValueError("Condition YAML must contain exactly 14 tactics.")
    labels, texts = [], []
    for label, record in tactics.items():
        text = str(record.get(field, "")).strip()
        if not text: raise ValueError(f"Condition {label} has no {field}.")
        labels.append(str(label)); texts.append(text)
    if not any("TA0040" in label for label in labels): raise ValueError("Impact (TA0040) is required.")
    return labels, texts, dict(body.get("metadata", {}))


def load_condition_geometry(config: dict[str, Any], device: torch.device, variant: str,
                            epsilon: float = 1e-6) -> ConditionGeometry:
    path = Path(config["path"]); labels, texts, source = _condition_records(path, str(config.get("text_field", "description_full")))
    cache_dir = Path(config["cache_dir"]); cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(path.read_bytes()); digest.update(json.dumps(config["embedder"], sort_keys=True).encode())
    cache_path = cache_dir / f"raw_tactics_{digest.hexdigest()[:24]}.npz"; hit = cache_path.is_file()
    if hit:
        with np.load(cache_path, allow_pickle=False) as archive:
            raw = archive["raw"].astype(np.float32); cached_labels = archive["labels"].astype(str).tolist()
        if cached_labels != labels: raise ValueError("Condition cache label order mismatch.")
    else:
        raw = normalize_rows(build_text_embedder(config["embedder"], device).encode(texts))
        np.savez_compressed(cache_path, raw=raw, labels=np.asarray(labels, dtype=str))
    common, residual, singular_values = uncentered_common_component(raw)
    residual_rank = int(np.linalg.matrix_rank(residual))
    initial = symmetric_orthogonalize_numpy(residual, epsilon, common) if variant == "full_orthogonal" else normalize_rows(residual)
    combined = np.vstack((common, initial)).astype(np.float32)
    gram = combined @ combined.T
    metadata = {
        "variant": variant, "cache_hit": hit, "cache_path": str(cache_path), "source": source,
        "singular_values": singular_values.tolist(),
        "first_component_energy_ratio": float(singular_values[0] ** 2 / np.sum(singular_values ** 2)),
        "residual_rank": residual_rank,
        "rank_completion_applied": bool(variant == "full_orthogonal" and residual_rank == 13),
        "residual_condition_number": float(np.linalg.cond(residual @ residual.T)),
        "max_common_tactic_dot": float(np.abs(initial @ common).max()),
        "max_orthogonality_error": float(np.abs(gram - np.eye(15)).max()),
    }
    common_label = str(config.get("common_label", "Common Malicious Component"))
    return ConditionGeometry([common_label, *labels], labels, raw, common, residual, initial, variant, metadata)


def save_condition_geometry(geometry: ConditionGeometry, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"condition_geometry_{geometry.variant}.npz",
        labels=np.asarray(geometry.labels, dtype=str), tactic_labels=np.asarray(geometry.tactic_labels, dtype=str),
        raw_tactics=geometry.raw_tactics, common=geometry.common,
        common_removed=geometry.common_removed, initial_tactics=geometry.initial_tactics)
    (directory / f"condition_geometry_{geometry.variant}.json").write_text(
        json.dumps(geometry.metadata, indent=2, ensure_ascii=False), encoding="utf-8")
