from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from .data import SplitIndices, TACTIC_LABELS
from .evaluate import calibrate_thresholds, multilabel_metrics, observed_tactics, run_technique_probes
from .model import MultiLabelDisentangledConditionalVAE
from .training import extract_batches, train_model
from .utils import write_json


CONTROL_NAMES = (
    "semantic",
    "random_gaussian",
    "random_orthogonal",
    "semantic_label_shuffle",
)


@dataclass(slots=True)
class ControlResult:
    name: str
    seed: int
    summary: dict[str, Any]
    tactic_probabilities: np.ndarray
    tactic_thresholds: np.ndarray
    probe_gold: np.ndarray
    probe_predictions: dict[str, np.ndarray]


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _row_norms(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(matrix, dtype=np.float32), axis=1, keepdims=True)


def build_condition_control_matrices(
    semantic_matrix: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    semantic = np.asarray(semantic_matrix, dtype=np.float32)
    if semantic.ndim != 2 or len(semantic) != 14:
        raise ValueError("Semantic control suite requires a [14, condition_dim] matrix.")
    count, dimension = semantic.shape
    if dimension < count:
        raise ValueError("condition_dim must be at least 14 for an orthogonal control.")
    norms = _row_norms(semantic)
    if np.any(norms < 1e-12):
        raise ValueError("Semantic condition rows must have non-zero norm.")

    gaussian_rng = np.random.default_rng(seed + 10_003)
    gaussian = gaussian_rng.normal(size=semantic.shape).astype(np.float32)
    gaussian /= np.maximum(_row_norms(gaussian), 1e-12)
    gaussian *= norms

    orthogonal_rng = np.random.default_rng(seed + 20_003)
    basis, _ = np.linalg.qr(orthogonal_rng.normal(size=(dimension, count)))
    orthogonal = basis.T.astype(np.float32) * norms
    return {
        "semantic": semantic.copy(),
        "random_gaussian": gaussian.astype(np.float32),
        "random_orthogonal": orthogonal.astype(np.float32),
        "semantic_label_shuffle": semantic.copy(),
    }


def condition_geometry_summary(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(len(values), dtype=bool)]
    return {
        "row_norm_min": float(np.linalg.norm(values, axis=1).min()),
        "row_norm_max": float(np.linalg.norm(values, axis=1).max()),
        "offdiag_cosine_min": float(off_diagonal.min()),
        "offdiag_cosine_mean": float(off_diagonal.mean()),
        "offdiag_cosine_max": float(off_diagonal.max()),
    }


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def shuffled_supervision_targets(
    targets: np.ndarray,
    split: SplitIndices,
    seed: int,
) -> np.ndarray:
    shuffled = np.asarray(targets, dtype=np.float32).copy()
    rng = np.random.default_rng(seed + 30_007)
    for indices in (split.train, split.val):
        permutation = rng.permutation(len(indices))
        shuffled[indices] = shuffled[indices[permutation]]
    return shuffled


def _malicious_representations(
    outputs: dict[str, dict[str, np.ndarray]],
    x: np.ndarray,
    metadata: pd.DataFrame,
    split: SplitIndices,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    split_indices = (split.train, split.val, split.test)
    names = ("train", "val", "test")
    masks = tuple(
        metadata.iloc[indices]["probe_eligible"].astype(str).str.casefold().eq("true").to_numpy()
        for indices in split_indices
    )
    labels = tuple(
        metadata.iloc[indices]["technique"].to_numpy(dtype=str)[mask]
        for indices, mask in zip(split_indices, masks, strict=True)
    )
    representations = {
        name: tuple(outputs[split_name][name][mask] for split_name, mask in zip(names, masks, strict=True))
        for name in ("h", "c", "hc")
    }
    representations["gates"] = tuple(
        outputs[split_name]["gates"][mask, : len(TACTIC_LABELS)]
        for split_name, mask in zip(names, masks, strict=True)
    )
    representations["x"] = tuple(
        np.asarray(x[indices], dtype=np.float32)[mask]
        for indices, mask in zip(split_indices, masks, strict=True)
    )
    return representations, labels


def _make_variant_directories(directory: Path) -> None:
    for name in ("checkpoints", "metrics", "embeddings"):
        (directory / name).mkdir(parents=True, exist_ok=True)


def run_control_variant(
    name: str,
    seed: int,
    x: np.ndarray,
    metadata: pd.DataFrame,
    true_targets: np.ndarray,
    split: SplitIndices,
    condition_matrix: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    directory: Path,
    logger: Any = None,
) -> ControlResult:
    if name not in CONTROL_NAMES:
        raise ValueError(f"Unknown semantic control: {name}")
    _make_variant_directories(directory)
    seed_everything(seed, bool(config.get("controls", {}).get("deterministic", True)))
    supervision_targets = (
        shuffled_supervision_targets(true_targets, split, seed)
        if name == "semantic_label_shuffle"
        else np.asarray(true_targets, dtype=np.float32)
    )
    model_config = dict(config["model"])
    model_config.update(
        {
            "condition_count": int(len(condition_matrix)),
            "supervised_condition_count": len(TACTIC_LABELS),
            "condition_dim": int(condition_matrix.shape[1]),
        }
    )
    model = MultiLabelDisentangledConditionalVAE.from_config(model_config)
    checkpoint = directory / "checkpoints" / "disentangled_cvae.pt"
    training_result = train_model(
        model,
        x,
        supervision_targets,
        condition_matrix,
        split,
        config["training"],
        model_config,
        device,
        checkpoint,
        seed,
        logger,
    )
    outputs = {
        split_name: extract_batches(
            model,
            x,
            true_targets,
            condition_matrix,
            indices,
            device,
            int(config["training"]["batch_size"]),
        )
        for split_name, indices in (("train", split.train), ("val", split.val), ("test", split.test))
    }
    evaluation = config["evaluation"]
    tactic_probabilities = outputs["test"]["gates"][:, : len(TACTIC_LABELS)]
    thresholds = calibrate_thresholds(
        true_targets[split.val],
        outputs["val"]["gates"][:, : len(TACTIC_LABELS)],
        evaluation["threshold_grid"],
    )
    tactic_metrics = multilabel_metrics(
        true_targets[split.test],
        tactic_probabilities,
        thresholds,
        TACTIC_LABELS,
        observed_tactics(true_targets[split.train]),
    )
    representations, technique_labels = _malicious_representations(outputs, x, metadata, split)
    probe = run_technique_probes(
        representations,
        *technique_labels,
        evaluation["probe_c_grid"],
        seed,
    )
    reconstruction = {
        "full_mse": float(outputs["test"]["recon_mse"].mean()),
        "h_only_mse": float(outputs["test"]["h_only_mse"].mean()),
        "c_only_mse": float(outputs["test"]["c_only_mse"].mean()),
    }
    summary = {
        "control": name,
        "seed": int(seed),
        "label_shuffle": name == "semantic_label_shuffle",
        "condition_matrix_sha256": _array_sha256(condition_matrix),
        "supervision": {
            "train_sha256": _array_sha256(supervision_targets[split.train]),
            "validation_sha256": _array_sha256(supervision_targets[split.val]),
            "test_sha256": _array_sha256(supervision_targets[split.test]),
            "test_matches_true_labels": bool(
                np.array_equal(supervision_targets[split.test], true_targets[split.test])
            ),
        },
        "condition_geometry": condition_geometry_summary(condition_matrix),
        "training": {
            "best_epoch": training_result.best_epoch,
            "best_val_loss": training_result.best_val_loss,
            "pos_weight": training_result.pos_weight,
        },
        "tactic": tactic_metrics,
        "probes": probe.metrics,
        "reconstruction": reconstruction,
    }
    write_json(summary, directory / "metrics" / "summary.json")
    np.save(directory / "embeddings" / "condition_matrix.npy", np.asarray(condition_matrix, dtype=np.float32))
    np.savez_compressed(
        directory / "metrics" / "control_result.npz",
        tactic_probabilities=tactic_probabilities,
        tactic_thresholds=thresholds,
        probe_gold=technique_labels[2],
        probe_c=probe.predictions["c"],
        probe_gates=probe.predictions["gates"],
        probe_h=probe.predictions["h"],
    )
    if not bool(config.get("controls", {}).get("keep_checkpoints", False)):
        checkpoint.unlink(missing_ok=True)
    return ControlResult(
        name,
        seed,
        summary,
        tactic_probabilities,
        thresholds,
        technique_labels[2],
        {key: probe.predictions[key] for key in ("c", "gates", "h")},
    )


def _stratified_bootstrap_indices(strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(strata, dtype=str)
    parts = []
    for label in sorted(set(values)):
        indices = np.flatnonzero(values == label)
        parts.append(rng.choice(indices, size=len(indices), replace=True))
    combined = np.concatenate(parts)
    rng.shuffle(combined)
    return combined


def paired_control_comparison(
    semantic: ControlResult,
    control: ControlResult,
    tactic_targets: np.ndarray,
    tactic_observed_mask: np.ndarray,
    tactic_strata: np.ndarray,
    probe_strata: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    if semantic.seed != control.seed:
        raise ValueError("Paired controls must use the same seed.")
    y = np.asarray(tactic_targets, dtype=np.int8)
    mask = np.asarray(tactic_observed_mask, dtype=bool)
    labels = np.unique(semantic.probe_gold)
    rng = np.random.default_rng(seed)
    distributions = {
        "tactic_macro_f1": np.empty(repeats, dtype=np.float64),
        "tactic_macro_auprc": np.empty(repeats, dtype=np.float64),
        "probe_c_macro_f1": np.empty(repeats, dtype=np.float64),
        "probe_gates_macro_f1": np.empty(repeats, dtype=np.float64),
    }

    def tactic_values(result: ControlResult, indices: np.ndarray) -> tuple[float, float]:
        truth = y[indices][:, mask]
        scores = result.tactic_probabilities[indices][:, mask]
        predictions = scores >= result.tactic_thresholds[mask][None, :]
        f1 = f1_score(truth, predictions, average="macro", zero_division=0)
        ap = np.mean(
            [average_precision_score(truth[:, index], scores[:, index]) for index in range(truth.shape[1])]
        )
        return float(f1), float(ap)

    def probe_value(result: ControlResult, representation: str, indices: np.ndarray) -> float:
        return float(
            f1_score(
                result.probe_gold[indices],
                result.probe_predictions[representation][indices],
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

    for repeat in range(repeats):
        tactic_indices = _stratified_bootstrap_indices(tactic_strata, rng)
        semantic_f1, semantic_ap = tactic_values(semantic, tactic_indices)
        control_f1, control_ap = tactic_values(control, tactic_indices)
        distributions["tactic_macro_f1"][repeat] = semantic_f1 - control_f1
        distributions["tactic_macro_auprc"][repeat] = semantic_ap - control_ap
        probe_indices = _stratified_bootstrap_indices(probe_strata, rng)
        for representation in ("c", "gates"):
            distributions[f"probe_{representation}_macro_f1"][repeat] = (
                probe_value(semantic, representation, probe_indices)
                - probe_value(control, representation, probe_indices)
            )

    result: dict[str, Any] = {"semantic": semantic.name, "control": control.name, "seed": semantic.seed}
    for metric, distribution in distributions.items():
        result[metric] = {
            "mean": float(distribution.mean()),
            "ci95_low": float(np.quantile(distribution, 0.025)),
            "ci95_high": float(np.quantile(distribution, 0.975)),
        }
    checks = {
        "tactic_macro_f1": result["tactic_macro_f1"]["ci95_low"] > 0.0,
        "tactic_macro_auprc": result["tactic_macro_auprc"]["ci95_low"] > 0.0,
        "probe_c_macro_f1": result["probe_c_macro_f1"]["ci95_low"] > 0.0,
        "probe_gates_macro_f1": result["probe_gates_macro_f1"]["ci95_low"] > 0.0,
    }
    result["checks"] = checks
    result["supported"] = bool(all(checks.values()))
    result["bootstrap_repeats"] = int(repeats)
    return result


def aggregate_control_decision(
    comparisons: list[dict[str, Any]],
    controls: list[str],
    seeds: list[int],
    minimum_seed_fraction: float,
) -> dict[str, Any]:
    by_control: dict[str, Any] = {}
    for control in controls:
        selected = [item for item in comparisons if item["control"] == control]
        supported = sum(bool(item["supported"]) for item in selected)
        fraction = supported / max(len(selected), 1)
        by_control[control] = {
            "supported_seeds": supported,
            "total_seeds": len(selected),
            "supported_seed_fraction": fraction,
            "mean_deltas": {
                metric: float(np.mean([item[metric]["mean"] for item in selected]))
                for metric in (
                    "tactic_macro_f1",
                    "tactic_macro_auprc",
                    "probe_c_macro_f1",
                    "probe_gates_macro_f1",
                )
            },
            "passes": fraction >= minimum_seed_fraction,
        }
    return {
        "semantic_geometry_supported": bool(by_control and all(item["passes"] for item in by_control.values())),
        "minimum_seed_fraction": float(minimum_seed_fraction),
        "seeds": list(map(int, seeds)),
        "comparators": by_control,
        "interpretation": (
            "ATT&CK semantic geometry is supported only when semantic conditions beat every random and "
            "label-shuffle control on paired tactic and condition-derived probe metrics."
        ),
    }


def load_control_result(directory: Path, summary: dict[str, Any]) -> ControlResult:
    with np.load(directory / "metrics" / "control_result.npz") as archive:
        return ControlResult(
            str(summary["control"]),
            int(summary["seed"]),
            summary,
            archive["tactic_probabilities"],
            archive["tactic_thresholds"],
            archive["probe_gold"].astype(str),
            {
                "c": archive["probe_c"].astype(str),
                "gates": archive["probe_gates"].astype(str),
                "h": archive["probe_h"].astype(str),
            },
        )
