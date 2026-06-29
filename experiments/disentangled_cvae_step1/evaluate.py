from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .conditions import cosine_similarity_matrix


def write_condition_gate_summary(
    gates: np.ndarray,
    condition_labels: list[str],
    path: Path,
) -> None:
    frame = pd.DataFrame(gates, columns=condition_labels)
    rows = []
    for component in frame.columns:
        values = frame[component]
        rows.append(
            {
                "condition": component,
                "mean_gate": float(values.mean()),
                "std_gate": float(values.std(ddof=0)),
                "p50_gate": float(values.quantile(0.50)),
                "p90_gate": float(values.quantile(0.90)),
                "p99_gate": float(values.quantile(0.99)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_condition_ablation_summary(
    deltas: np.ndarray,
    condition_labels: list[str],
    path: Path,
) -> None:
    frame = pd.DataFrame(deltas, columns=condition_labels)
    rows = []
    for component in frame.columns:
        values = frame[component]
        rows.append(
            {
                "condition": component,
                "mean_delta_mse": float(values.mean()),
                "std_delta_mse": float(values.std(ddof=0)),
                "p50_delta_mse": float(values.quantile(0.50)),
                "p90_delta_mse": float(values.quantile(0.90)),
                "p99_delta_mse": float(values.quantile(0.99)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_similarity_matrix(
    labels: list[str],
    matrix: np.ndarray,
    path: Path,
) -> None:
    pd.DataFrame(cosine_similarity_matrix(matrix), index=labels, columns=labels).to_csv(path)


def plot_condition_similarity_heatmap(
    labels: list[str],
    matrix: np.ndarray,
    path: Path,
    title: str,
) -> None:
    similarity = cosine_similarity_matrix(matrix)
    fig_width = max(8.0, min(18.0, 0.72 * len(labels) + 4.0))
    fig_height = max(7.0, min(18.0, 0.62 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(similarity, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = similarity[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if abs(value) > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _visualization_sample(size: int, max_samples: int, seed: int) -> np.ndarray:
    if size <= max_samples:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    result = rng.choice(np.arange(size), size=max_samples, replace=False)
    return np.sort(result).astype(np.int64)


def _project_2d(
    features: np.ndarray,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> tuple[np.ndarray, str]:
    values = np.asarray(features, dtype=np.float32)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float32), "none"
    if len(values) == 1:
        return np.zeros((1, 2), dtype=np.float32), "single-point"
    values = StandardScaler().fit_transform(values)
    try:
        import umap

        neighbors = max(2, min(int(n_neighbors), len(values) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(values).astype(np.float32), "UMAP"
    except Exception:
        coords = PCA(n_components=2, random_state=seed).fit_transform(values)
        return coords.astype(np.float32), "PCA fallback"


def plot_umap_projection(
    features: np.ndarray,
    path: Path,
    title: str,
    config: dict[str, Any],
) -> None:
    max_samples = int(config.get("visualization_max_samples", 5000))
    seed = int(config.get("random_state", 42))
    selected = _visualization_sample(len(features), max_samples, seed)
    selected_features = np.asarray(features[selected], dtype=np.float32)
    backend = str(config.get("visualization_backend", "umap")).lower()
    if backend == "pca":
        coords, method = _project_2d_with_pca(selected_features, seed)
    elif backend == "umap":
        coords, method = _project_2d(
            selected_features,
            seed,
            int(config.get("umap_n_neighbors", 30)),
            float(config.get("umap_min_dist", 0.1)),
        )
    else:
        raise ValueError("evaluation.visualization_backend must be 'umap' or 'pca'.")
    fig, ax = plt.subplots(figsize=(9.5, 7.0), dpi=160)
    colors = selected.astype(np.float32)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=10,
        alpha=0.72,
        c=colors,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(f"{title} ({method}, n={len(selected)})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="sample order in plotted split")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _project_2d_with_pca(features: np.ndarray, seed: int) -> tuple[np.ndarray, str]:
    values = np.asarray(features, dtype=np.float32)
    if len(values) == 0:
        return np.empty((0, 2), dtype=np.float32), "none"
    if len(values) == 1:
        return np.zeros((1, 2), dtype=np.float32), "single-point"
    values = StandardScaler().fit_transform(values)
    coords = PCA(n_components=2, random_state=seed).fit_transform(values)
    return coords.astype(np.float32), "PCA"
