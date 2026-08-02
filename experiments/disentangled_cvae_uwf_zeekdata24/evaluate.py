from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def observed_tactics(targets: np.ndarray) -> np.ndarray:
    values = np.asarray(targets, dtype=np.float32)
    return values.sum(axis=0) > 0


def calibrate_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    grid: list[float],
) -> np.ndarray:
    y = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float32)
    if y.shape != scores.shape:
        raise ValueError("targets and probabilities must have the same shape.")
    thresholds = np.ones(y.shape[1], dtype=np.float32)
    for index in range(y.shape[1]):
        if y[:, index].sum() == 0:
            continue
        candidates = []
        for threshold in map(float, grid):
            prediction = scores[:, index] >= threshold
            value = f1_score(y[:, index], prediction, zero_division=0)
            candidates.append((float(value), -abs(threshold - 0.5), -threshold, threshold))
        thresholds[index] = max(candidates)[-1]
    return thresholds


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str] | tuple[str, ...],
    observed_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(targets, dtype=np.int8)
    scores = np.asarray(probabilities, dtype=np.float32)
    prediction = (scores >= np.asarray(thresholds)[None, :]).astype(np.int8)
    mask = observed_tactics(y) if observed_mask is None else np.asarray(observed_mask, dtype=bool)
    if not mask.any():
        raise ValueError("At least one observed tactic is required for metrics.")
    per_label: dict[str, Any] = {}
    ap_values = []
    auc_values = []
    for index, label in enumerate(labels):
        truth = y[:, index]
        item: dict[str, Any] = {
            "observed": bool(mask[index]),
            "support": int(truth.sum()),
            "predicted_positive": int(prediction[:, index].sum()),
            "threshold": float(thresholds[index]),
            "f1": float(f1_score(truth, prediction[:, index], zero_division=0)),
        }
        if truth.min() != truth.max():
            item["average_precision"] = float(average_precision_score(truth, scores[:, index]))
            item["roc_auc"] = float(roc_auc_score(truth, scores[:, index]))
            if mask[index]:
                ap_values.append(item["average_precision"])
                auc_values.append(item["roc_auc"])
        else:
            item["average_precision"] = None
            item["roc_auc"] = None
        per_label[str(label)] = item
    observed_y = y[:, mask]
    observed_prediction = prediction[:, mask]
    return {
        "observed_label_count": int(mask.sum()),
        "micro_f1": float(f1_score(observed_y, observed_prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(observed_y, observed_prediction, average="macro", zero_division=0)),
        "macro_auprc": float(np.mean(ap_values)) if ap_values else None,
        "macro_roc_auc": float(np.mean(auc_values)) if auc_values else None,
        "hamming_loss": float(hamming_loss(observed_y, observed_prediction)),
        "exact_match": float(accuracy_score(observed_y, observed_prediction)),
        "per_label": per_label,
    }


def shuffled_label_baseline(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    labels: list[str] | tuple[str, ...],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    observed = observed_tactics(targets)
    macro_f1 = []
    macro_auprc = []
    for _ in range(int(repeats)):
        shuffled = np.asarray(targets)[rng.permutation(len(targets))]
        metrics = multilabel_metrics(shuffled, probabilities, thresholds, labels, observed)
        macro_f1.append(float(metrics["macro_f1"]))
        macro_auprc.append(float(metrics["macro_auprc"]))
    return {
        "repeats": int(repeats),
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_p95": float(np.quantile(macro_f1, 0.95)),
        "macro_auprc_mean": float(np.mean(macro_auprc)),
        "macro_auprc_p95": float(np.quantile(macro_auprc, 0.95)),
    }


def majority_baseline(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    labels: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    prevalence = np.asarray(train_targets).mean(axis=0)
    predictions = np.broadcast_to(prevalence >= 0.5, np.asarray(test_targets).shape)
    return multilabel_metrics(
        test_targets,
        predictions.astype(np.float32),
        np.full(len(labels), 0.5, dtype=np.float32),
        labels,
        observed_tactics(train_targets),
    )


@dataclass(slots=True)
class ProbeResult:
    metrics: dict[str, Any]
    predictions: dict[str, np.ndarray]
    models: dict[str, Pipeline]
    best_c_by_representation: dict[str, float]
    validation_macro_f1: dict[str, float]


def run_technique_probes(
    representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    c_grid: list[float],
    seed: int,
) -> ProbeResult:
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    models: dict[str, Pipeline] = {}
    best_cs: dict[str, float] = {}
    val_scores: dict[str, float] = {}
    classes = np.unique(train_labels)
    if len(classes) < 2:
        raise ValueError("Technique probes require at least two training classes.")
    for name, (train_x, val_x, test_x) in representations.items():
        candidates = []
        for c_value in map(float, c_grid):
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c_value,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(train_x, train_labels)
            val_prediction = pipeline.predict(val_x)
            score = float(f1_score(val_labels, val_prediction, average="macro", zero_division=0))
            candidates.append((score, -c_value, c_value, pipeline))
        score, _, best_c, model = max(candidates, key=lambda item: (item[0], item[1]))
        prediction = model.predict(test_x)
        predictions[name] = prediction.astype(str)
        models[name] = model
        best_cs[name] = float(best_c)
        val_scores[name] = float(score)
        metrics[name] = {
            "best_c": float(best_c),
            "validation_macro_f1": float(score),
            "test_accuracy": float(accuracy_score(test_labels, prediction)),
            "test_macro_f1": float(f1_score(test_labels, prediction, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(test_labels, prediction, average="weighted", zero_division=0)),
        }
    counts = {label: int(np.count_nonzero(train_labels == label)) for label in sorted(set(train_labels))}
    majority_label = max(counts, key=counts.get)
    majority_prediction = np.full(len(test_labels), majority_label, dtype=object)
    metrics["majority_baseline"] = {
        "label": majority_label,
        "test_accuracy": float(accuracy_score(test_labels, majority_prediction)),
        "test_macro_f1": float(f1_score(test_labels, majority_prediction, average="macro", zero_division=0)),
    }
    return ProbeResult(metrics, predictions, models, best_cs, val_scores)


def bootstrap_mean_interval(values: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("bootstrap values must be a non-empty 1D array.")
    rng = np.random.default_rng(seed)
    means = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        means[index] = array[rng.integers(0, len(array), size=len(array))].mean()
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "repeats": int(repeats),
    }


def bootstrap_macro_f1_difference(
    targets: np.ndarray,
    first_predictions: np.ndarray,
    second_predictions: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float | str]:
    truth = np.asarray(targets, dtype=str)
    first = np.asarray(first_predictions, dtype=str)
    second = np.asarray(second_predictions, dtype=str)
    if truth.ndim != 1 or not (len(truth) == len(first) == len(second)) or len(truth) == 0:
        raise ValueError("Technique bootstrap inputs must be non-empty aligned 1D arrays.")
    labels = np.unique(truth)

    def difference(indices: np.ndarray) -> float:
        return float(
            f1_score(truth[indices], first[indices], labels=labels, average="macro", zero_division=0)
            - f1_score(truth[indices], second[indices], labels=labels, average="macro", zero_division=0)
        )

    all_indices = np.arange(len(truth))
    rng = np.random.default_rng(seed)
    differences = np.empty(int(repeats), dtype=np.float64)
    for index in range(int(repeats)):
        differences[index] = difference(rng.integers(0, len(truth), size=len(truth)))
    return {
        "metric": "macro_f1_difference",
        "mean": difference(all_indices),
        "ci95_low": float(np.quantile(differences, 0.025)),
        "ci95_high": float(np.quantile(differences, 0.975)),
        "repeats": int(repeats),
    }


def semantic_acceptance(
    tactic_metrics: dict[str, Any],
    shuffle: dict[str, Any],
    probe_metrics: dict[str, Any],
    best_condition_representation: str,
    technique_delta: dict[str, float],
    reconstruction_gain: dict[str, float],
) -> dict[str, Any]:
    checks = {
        "tactic_f1_above_shuffle_p95": tactic_metrics["macro_f1"] > shuffle["macro_f1_p95"],
        "tactic_auprc_above_shuffle_p95": tactic_metrics["macro_auprc"] > shuffle["macro_auprc_p95"],
        "condition_probe_beats_h": technique_delta["ci95_low"] > 0.0,
        "full_reconstruction_beats_h_only": reconstruction_gain["ci95_low"] > 0.0,
        "h_does_not_beat_x": probe_metrics["h"]["test_macro_f1"] <= probe_metrics["x"]["test_macro_f1"],
        "h_does_not_beat_best_condition": probe_metrics["h"]["test_macro_f1"] <= probe_metrics[best_condition_representation]["test_macro_f1"],
    }
    return {"supported": bool(all(checks.values())), "checks": checks}
