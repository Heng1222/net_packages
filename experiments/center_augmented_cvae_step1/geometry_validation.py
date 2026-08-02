from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.center_augmented_cvae_step1.conditions import (  # noqa: E402
    ConditionSet, centroid_decomposition, load_condition_set,
)
from experiments.center_augmented_cvae_step1.data import make_time_split  # noqa: E402
from experiments.center_augmented_cvae_step1.embedders import normalize_rows  # noqa: E402
from experiments.center_augmented_cvae_step1.utils import (  # noqa: E402
    load_config, resolve_device, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate mean baseline and unsupervised gate geometry.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-dir", default=None, help="Compatible prepared x.npy + metadata.csv directory")
    parser.add_argument("--reference-run", default=None, help="Run containing metrics/experiment_summary.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pca-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def load_compatible_prepared(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    root = Path(path)
    x = np.load(root / "x.npy", mmap_mode="r")
    metadata = pd.read_csv(root / "metadata.csv", dtype=str)
    if len(x) != len(metadata):
        raise ValueError("Prepared x.npy and metadata.csv row counts differ.")
    if x.ndim != 2:
        raise ValueError("Prepared x.npy must be a 2D embedding matrix.")
    sample = np.asarray(x[: min(len(x), 1000)], dtype=np.float32)
    if not np.allclose(np.linalg.norm(sample, axis=1), 1.0, atol=1e-4):
        raise ValueError("Prepared embeddings must be L2-normalized.")
    required = {"sample_id", "datetime"}
    if not required.issubset(metadata.columns):
        raise KeyError(f"Prepared metadata is missing: {sorted(required - set(metadata.columns))}")
    return x, metadata


def streaming_mean(x: np.ndarray, indices: np.ndarray, batch_size: int) -> np.ndarray:
    total = np.zeros(x.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(x[indices[start:start + batch_size]], dtype=np.float32)
        total += batch.sum(axis=0, dtype=np.float64)
        count += len(batch)
    if count == 0:
        raise ValueError("Cannot compute a mean from an empty split.")
    return (total / count).astype(np.float32)


def mean_reconstruction_metrics(x: np.ndarray, indices: np.ndarray, mean: np.ndarray,
                                batch_size: int) -> dict[str, float]:
    mse_sum = 0.0
    cosine_sum = 0.0
    mean_norm = max(float(np.linalg.norm(mean)), 1e-12)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(x[indices[start:start + batch_size]], dtype=np.float32)
        mse_sum += float(np.square(batch - mean[None, :]).mean(axis=1).sum())
        cosine_sum += float((batch @ mean / (np.clip(np.linalg.norm(batch, axis=1), 1e-12, None) * mean_norm)).sum())
        count += len(batch)
    return {"recon_mse": mse_sum / max(count, 1), "recon_cosine": cosine_sum / max(count, 1)}


def fit_top_component(x: np.ndarray, train_indices: np.ndarray, train_mean: np.ndarray,
                      max_samples: int, seed: int) -> tuple[np.ndarray, int]:
    sample_count = min(len(train_indices), int(max_samples))
    positions = np.linspace(0, len(train_indices) - 1, sample_count, dtype=np.int64)
    sample = np.asarray(x[train_indices[positions]], dtype=np.float32) - train_mean[None, :]
    _, _, vh = randomized_svd(sample, n_components=1, n_iter=5, random_state=seed)
    component = vh[0].astype(np.float32)
    component /= max(float(np.linalg.norm(component)), 1e-12)
    return component, sample_count


def transform_vectors(values: np.ndarray, payload_mean: np.ndarray | None,
                      top_component: np.ndarray | None) -> np.ndarray:
    transformed = np.asarray(values, dtype=np.float32)
    if payload_mean is not None:
        transformed = transformed - payload_mean[None, :]
    if top_component is not None:
        transformed = transformed - (transformed @ top_component)[:, None] * top_component[None, :]
    return normalize_rows(transformed)


def variant_gate_matrix(raw_tactics: np.ndarray, payload_mean: np.ndarray | None,
                        top_component: np.ndarray | None) -> tuple[np.ndarray, dict[str, float]]:
    transformed = np.asarray(raw_tactics, dtype=np.float32)
    if payload_mean is not None:
        transformed = transformed - payload_mean[None, :]
    if top_component is not None:
        transformed = transformed - (transformed @ top_component)[:, None] * top_component[None, :]
    centroid, centered, _, gate = centroid_decomposition(transformed)
    return gate, {
        "transformed_centroid_norm": float(np.linalg.norm(centroid)),
        "centered_mean_error": float(np.max(np.abs(centered.mean(axis=0)))),
    }


def clean_label(value: Any) -> str:
    text = str(value).strip()
    if text.startswith(("{", "[", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (set, list, tuple)) and len(parsed) == 1:
                return str(next(iter(parsed)))
        except (SyntaxError, ValueError):
            pass
    return text


def matched_golden(metadata: pd.DataFrame, split_names: np.ndarray, config: dict[str, Any],
                   tactic_labels: list[str]) -> pd.DataFrame:
    gold = pd.read_csv(config["golden_path"], usecols=[config["golden_sample_id_col"], config["golden_label_col"]], dtype=str)
    gold = gold.rename(columns={config["golden_sample_id_col"]: "sample_id", config["golden_label_col"]: "gold_tactic"})
    gold["gold_tactic"] = gold["gold_tactic"].map(clean_label)
    lookup = metadata[["sample_id"]].copy()
    lookup["row_index"] = np.arange(len(metadata))
    lookup["split"] = split_names
    recognized = [*tactic_labels, "Normal (TA9000)"]
    return gold.merge(lookup, on="sample_id", how="inner").drop_duplicates(["sample_id", "gold_tactic"]).query(
        "gold_tactic in @recognized"
    ).reset_index(drop=True)


def sigmoid_cosine(values: np.ndarray, gate_matrix: np.ndarray, temperature: float) -> np.ndarray:
    cosine = normalize_rows(values) @ normalize_rows(gate_matrix).T
    logits = np.clip(cosine / float(temperature), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def alignment_metrics(gates: np.ndarray, matched: pd.DataFrame,
                      tactic_labels: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    label_to_index = {label: index for index, label in enumerate(tactic_labels)}
    tactic_gates = gates[:, 1:]
    predicted_all = tactic_gates.argmax(axis=1)
    predicted_labels = np.asarray(tactic_labels, dtype=object)[predicted_all]
    result = matched.copy()
    result["predicted_tactic"] = predicted_labels
    result["common_gate"] = gates[:, 0]
    result["gold_gate"] = np.nan
    result["gold_gate_rank"] = np.nan
    is_tactic = result["gold_tactic"].isin(label_to_index).to_numpy()
    rows = np.flatnonzero(is_tactic)
    targets = result.iloc[rows]["gold_tactic"].map(label_to_index).astype(int).to_numpy()
    predicted = predicted_all[rows]
    ranked = np.argsort(np.argsort(-tactic_gates[rows], axis=1), axis=1)
    result.loc[result.index[rows], "gold_gate"] = tactic_gates[rows, targets]
    result.loc[result.index[rows], "gold_gate_rank"] = ranked[np.arange(len(rows)), targets] + 1
    normal = ~is_tactic
    majority = float(pd.Series(targets).value_counts(normalize=True).max())
    top_counts = Counter(predicted_labels[rows])
    top_label, top_count = top_counts.most_common(1)[0]
    common_auc = float(roc_auc_score(is_tactic.astype(int), gates[:, 0])) if normal.any() and is_tactic.any() else float("nan")
    metrics = {
        "matched_rows": len(result), "matched_tactic_rows": int(is_tactic.sum()),
        "matched_normal_rows": int(normal.sum()), "accuracy": float(accuracy_score(targets, predicted)),
        "majority_accuracy": majority,
        "macro_f1_present_labels": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "macro_f1_all_13": float(f1_score(targets, predicted, labels=np.arange(len(tactic_labels)), average="macro", zero_division=0)),
        "mean_gold_gate": float(result.loc[is_tactic, "gold_gate"].mean()),
        "mean_non_gold_gate": float(((tactic_gates[rows].sum(axis=1) - tactic_gates[rows, targets]) / (len(tactic_labels) - 1)).mean()),
        "mean_gold_rank": float(result.loc[is_tactic, "gold_gate_rank"].mean()),
        "top_predicted_label": top_label, "top_prediction_rate": float(top_count / len(rows)),
        "mean_common_gate_malicious": float(gates[is_tactic, 0].mean()),
        "mean_common_gate_normal": float(gates[normal, 0].mean()) if normal.any() else float("nan"),
        "common_gate_gap": float(gates[is_tactic, 0].mean() - gates[normal, 0].mean()) if normal.any() else float("nan"),
        "common_gate_malicious_auc": common_auc,
        "matched_by_split": {str(k): int(v) for k, v in result["split"].value_counts().items()},
        "predicted_distribution": {str(k): int(v) for k, v in top_counts.most_common()},
    }
    metrics["semantic_alignment_pass"] = bool(
        metrics["accuracy"] > metrics["majority_accuracy"]
        and metrics["macro_f1_present_labels"] >= 0.20
        and metrics["mean_gold_rank"] < 5.0
        and metrics["top_prediction_rate"] < 0.70
    )
    metrics["common_direction_pass"] = bool(
        np.isfinite(common_auc) and common_auc >= 0.65 and metrics["common_gate_gap"] >= 0.02
    )
    return metrics, result


def test_gate_distribution(x: np.ndarray, indices: np.ndarray, gate_matrix: np.ndarray,
                           payload_mean: np.ndarray | None, top_component: np.ndarray | None,
                           temperature: float, threshold: float, batch_size: int) -> dict[str, Any]:
    gate_sum = np.zeros(len(gate_matrix), dtype=np.float64)
    gate_sq_sum = np.zeros(len(gate_matrix), dtype=np.float64)
    active_sum = 0
    predicted = Counter()
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(x[indices[start:start + batch_size]], dtype=np.float32)
        batch = transform_vectors(batch, payload_mean, top_component)
        gates = sigmoid_cosine(batch, gate_matrix, temperature)
        gate_sum += gates.sum(axis=0, dtype=np.float64)
        gate_sq_sum += np.square(gates).sum(axis=0, dtype=np.float64)
        active_sum += int((gates >= threshold).sum())
        predicted.update(gates[:, 1:].argmax(axis=1).tolist())
        count += len(gates)
    mean = gate_sum / max(count, 1)
    std = np.sqrt(np.maximum(gate_sq_sum / max(count, 1) - mean ** 2, 0.0))
    top_index, top_count = predicted.most_common(1)[0]
    return {
        "rows": count, "mean_active_conditions": active_sum / max(count, 1),
        "mean_gates": mean, "std_gates": std,
        "top_predicted_tactic_index": int(top_index), "top_prediction_rate": top_count / max(count, 1),
        "predicted_distribution_by_index": {str(k): int(v) for k, v in predicted.most_common()},
    }


def reference_metrics(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    summary_path = Path(path) / "metrics" / "experiment_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config, PROJECT_ROOT)
    prepared_dir = Path(args.prepared_dir or config["data"]["prepared_dir"])
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["output"]["base_dir"]) / "geometry_validation" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    x, metadata = load_compatible_prepared(prepared_dir)
    split = make_time_split(metadata, config["data"]["split"])
    train_mean = streaming_mean(x, split.train, args.batch_size)
    mean_metrics = mean_reconstruction_metrics(x, split.test, train_mean, args.batch_size)
    top_component, pca_samples = fit_top_component(x, split.train, train_mean, args.pca_samples, int(config.get("seed", 42)))
    conditions: ConditionSet = load_condition_set(config["conditions"], resolve_device(config["training"].get("device", "auto")))
    split_names = np.full(len(metadata), "", dtype=object)
    split_names[split.train] = "train"; split_names[split.val] = "val"; split_names[split.test] = "test"
    matched = matched_golden(metadata, split_names, config["evaluation"], conditions.tactic_labels)
    temperature = float(config["model"]["gate_temperature"])
    threshold = float(config["evaluation"].get("condition_threshold", 0.5))
    variants = {
        "full_yaml_raw": (None, None),
        "payload_mean_removed": (train_mean, None),
        "payload_mean_and_top_pc_removed": (train_mean, top_component),
    }
    variant_results: dict[str, Any] = {}
    for name, (payload_mean, component) in variants.items():
        gate_matrix, geometry = variant_gate_matrix(conditions.raw_tactics, payload_mean, component)
        gold_values = transform_vectors(np.asarray(x[matched["row_index"].astype(int).to_numpy()], dtype=np.float32), payload_mean, component)
        gates = sigmoid_cosine(gold_values, gate_matrix, temperature)
        alignment, predictions = alignment_metrics(gates, matched, conditions.tactic_labels)
        distribution = test_gate_distribution(x, split.test, gate_matrix, payload_mean, component,
                                              temperature, threshold, args.batch_size)
        distribution["top_predicted_tactic"] = conditions.tactic_labels[distribution.pop("top_predicted_tactic_index")]
        alignment["test_gate_distribution"] = distribution
        alignment["geometry"] = geometry
        variant_results[name] = alignment
        predictions.to_csv(output_dir / f"{name}_golden_predictions.csv", index=False)
        np.savez_compressed(output_dir / f"{name}_geometry.npz", gate_matrix=gate_matrix,
                            payload_mean=np.asarray(payload_mean if payload_mean is not None else [], dtype=np.float32),
                            top_component=np.asarray(component if component is not None else [], dtype=np.float32))
    reference = reference_metrics(args.reference_run)
    mean_comparison: dict[str, Any] = {"train_mean_norm": float(np.linalg.norm(train_mean)), "test": mean_metrics}
    if reference:
        for key, model_metrics in (("main_cvae", reference["main_cvae"]["full"]),
                                   ("random_condition_cvae", reference["random_condition_cvae"]["full"]),
                                   ("plain_vae", reference["plain_vae"])):
            model_mse = float(model_metrics["recon_mse"])
            mean_comparison[key] = {"recon_mse": model_mse,
                                    "mse_improvement_over_mean_pct": (mean_metrics["recon_mse"] - model_mse) / mean_metrics["recon_mse"] * 100.0}
        mean_comparison["main_beats_mean_by_10pct"] = bool(mean_comparison["main_cvae"]["mse_improvement_over_mean_pct"] >= 10.0)
    summary = {
        "rows": len(x), "split_rows": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "step1_full_condition_yaml": {"tactics": len(conditions.tactic_labels),
                                      "techniques": 209, "keywords": 65, "pass": len(conditions.tactic_labels) == 13},
        "step2_mean_reconstruction_baseline": mean_comparison,
        "step3_gate_geometry_variants": variant_results,
        "pca_fit_samples": pca_samples,
        "success_criteria": {
            "semantic_alignment": "accuracy > majority, macro F1 >= 0.20, mean gold rank < 5, top prediction rate < 0.70",
            "common_direction": "malicious-vs-Normal AUC >= 0.65 and mean gate gap >= 0.02",
            "model_vs_mean": "main CVAE test MSE at least 10% below train-mean baseline",
        },
    }
    write_json(summary, output_dir / "geometry_validation_summary.json")
    pd.DataFrame({"dimension": np.arange(len(train_mean)), "train_payload_mean": train_mean,
                  "top_component": top_component}).to_csv(output_dir / "payload_common_geometry.csv", index=False)
    print(output_dir)


if __name__ == "__main__":
    main()
