from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _sample(values: np.ndarray, labels: np.ndarray | None, maximum: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(values) <= maximum:
        return values, labels
    indices = np.random.default_rng(seed).choice(len(values), size=maximum, replace=False)
    return values[indices], labels[indices] if labels is not None else None


def _scatter(coordinates: np.ndarray, labels: np.ndarray | None, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 7))
    if labels is None:
        axis.scatter(coordinates[:, 0], coordinates[:, 1], s=14, alpha=0.7)
    else:
        unique = sorted(np.unique(labels).tolist())
        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        for index, label in enumerate(unique):
            mask = labels == label
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=14, alpha=0.7, label=label, color=cmap(index))
        axis.legend(loc="best", fontsize=7, markerscale=1.4)
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def visualize_latent(
    train_latent: np.ndarray,
    test_latent: np.ndarray,
    test_labels: np.ndarray | None,
    methods: list[str],
    output_prefix: str | Path,
    seed: int,
    max_samples: int,
) -> dict[str, str | None]:
    logger = logging.getLogger("ae_cvae_tactic")
    prefix = Path(output_prefix)
    values, labels = _sample(test_latent, test_labels, max_samples, seed)
    outputs: dict[str, str | None] = {}
    if len(values) < 2:
        logger.warning("Visualization skipped because fewer than two samples are available.")
        return outputs
    if "pca" in methods:
        components = min(2, train_latent.shape[1], len(train_latent))
        model = PCA(n_components=components, random_state=seed).fit(train_latent)
        coordinates = model.transform(values)
        if components == 1:
            coordinates = np.column_stack((coordinates[:, 0], np.zeros(len(coordinates))))
        path = prefix.with_name(prefix.name + "_pca.png")
        _scatter(coordinates, labels, "PCA latent projection", path)
        outputs["pca"] = str(path)
    if "tsne" in methods:
        if len(values) < 4:
            logger.warning("t-SNE skipped because fewer than four test samples are available.")
            outputs["tsne"] = None
        else:
            perplexity = min(30.0, max(2.0, (len(values) - 1) / 3.0))
            coordinates = TSNE(
                n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed
            ).fit_transform(values)
            path = prefix.with_name(prefix.name + "_tsne.png")
            _scatter(coordinates, labels, "t-SNE latent projection", path)
            outputs["tsne"] = str(path)
    if "umap" in methods:
        try:
            from umap import UMAP
        except ImportError:
            logger.warning("umap-learn is not installed; UMAP visualization was skipped.")
            outputs["umap"] = None
        else:
            neighbors = min(15, max(2, len(train_latent) - 1))
            model = UMAP(n_components=2, n_neighbors=neighbors, random_state=seed, n_jobs=1).fit(train_latent)
            coordinates = model.transform(values)
            path = prefix.with_name(prefix.name + "_umap.png")
            _scatter(coordinates, labels, "UMAP latent projection", path)
            outputs["umap"] = str(path)
    return outputs


def plot_confusion(metrics: dict[str, Any], path: str | Path, title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["labels"]
    size = max(7, min(14, len(labels) * 0.8))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046)
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7,
                      color="white" if matrix[row, column] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history: list[dict[str, float]], path: str | Path, title: str) -> None:
    if not history:
        return
    keys = [key for key in history[0] if key != "epoch" and ("loss" in key)]
    fig, axis = plt.subplots(figsize=(8, 5))
    epochs = [row["epoch"] for row in history]
    for key in keys:
        axis.plot(epochs, [row[key] for row in history], label=key)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(title)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
