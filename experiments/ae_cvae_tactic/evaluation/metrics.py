from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from ..models.classifier import build_classifier


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None) -> dict[str, Any]:
    names = labels or sorted(set(map(str, y_true)).union(map(str, y_pred)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=names, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=names, average="weighted", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=names, target_names=names, output_dict=True, zero_division=0
        ),
        "labels": names,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=names).tolist(),
    }


def evaluate_classifier(
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    test_latent: np.ndarray,
    test_labels: np.ndarray,
    config: dict[str, Any],
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if len(np.unique(train_labels)) < 2:
        raise ValueError("Classifier evaluation requires at least two training classes.")
    model = build_classifier(config)
    model.fit(train_latent, train_labels)
    predictions = np.asarray(model.predict(test_latent), dtype=str)
    if model_path is not None:
        joblib.dump(model, model_path)
    labels = sorted(set(map(str, train_labels)).union(map(str, test_labels)))
    return classification_metrics(test_labels.astype(str), predictions, labels), predictions


def evaluate_clustering(
    train_latent: np.ndarray,
    train_labels: np.ndarray,
    test_latent: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    cluster_count = len(np.unique(train_labels))
    if cluster_count < 2 or len(test_latent) < 2:
        return {"silhouette": None, "nmi": None, "ari": None, "n_clusters": cluster_count}, np.zeros(len(test_latent), dtype=int)
    model = KMeans(n_clusters=cluster_count, n_init=20, random_state=seed)
    model.fit(train_latent)
    predicted = model.predict(test_latent)
    unique_clusters = len(np.unique(predicted))
    silhouette: float | None = None
    if 1 < unique_clusters < len(test_latent):
        silhouette = float(silhouette_score(test_latent, predicted))
    return {
        "silhouette": silhouette,
        "nmi": float(normalized_mutual_info_score(test_labels, predicted)),
        "ari": float(adjusted_rand_score(test_labels, predicted)),
        "n_clusters": cluster_count,
    }, predicted
