from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from .conditions import cosine_similarity_matrix


AMBIGUOUS_LABEL = "ambiguous"


def independent_condition_probabilities(gates: np.ndarray) -> np.ndarray:
    values = np.asarray(gates, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"gates must be a 2D array; got shape {values.shape}")
    if values.shape[0] == 0:
        return values.astype(np.float32)
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def build_test_condition_predictions(
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    gates: np.ndarray,
    condition_labels: list[str],
    threshold: float = 0.5,
    ambiguous_label: str = AMBIGUOUS_LABEL,
) -> pd.DataFrame:
    indices = np.asarray(test_indices, dtype=np.int64)
    gate_values = np.asarray(gates, dtype=np.float32)
    if gate_values.ndim != 2:
        raise ValueError(f"gates must be a 2D array; got shape {gate_values.shape}")
    if len(indices) != len(gate_values):
        raise ValueError("test_indices and gates must have the same row count.")
    if gate_values.shape[1] != len(condition_labels):
        raise ValueError("gates column count must match condition_labels.")

    probabilities = independent_condition_probabilities(gate_values)
    base = metadata.iloc[indices].reset_index(drop=True).copy()
    metadata_columns = [col for col in base.columns if col not in {"row_index", "test_position"}]
    frame = pd.DataFrame(
        {
            "row_index": indices,
            "test_position": np.arange(len(indices), dtype=np.int64),
        }
    )
    frame = pd.concat([frame, base.loc[:, metadata_columns]], axis=1)
    for index, label in enumerate(condition_labels):
        frame[f"condition_prob__{label}"] = probabilities[:, index]

    if len(frame):
        max_indices = np.argmax(probabilities, axis=1)
        max_probabilities = probabilities[np.arange(len(probabilities)), max_indices]
        max_conditions = np.asarray(condition_labels, dtype=object)[max_indices]
        active = probabilities >= float(threshold)
        active_counts = active.sum(axis=1)
        predicted_multi = []
        for row in active:
            labels = [condition_labels[index] for index, is_active in enumerate(row) if is_active]
            predicted_multi.append("|".join(labels) if labels else ambiguous_label)
        predicted = np.where(max_probabilities >= float(threshold), max_conditions, ambiguous_label)
        frame["max_condition"] = max_conditions
        frame["max_condition_probability"] = max_probabilities
        frame["active_condition_count"] = active_counts
        frame["predicted_conditions"] = predicted_multi
        frame["predicted_condition"] = predicted
    else:
        frame["max_condition"] = []
        frame["max_condition_probability"] = []
        frame["active_condition_count"] = []
        frame["predicted_conditions"] = []
        frame["predicted_condition"] = []
    return frame


def write_testset_subset(
    predictions: pd.DataFrame,
    path: Path,
    per_class_limit: int = 100,
    class_column: str = "predicted_condition",
) -> None:
    if class_column not in predictions.columns:
        raise KeyError(f"Missing class column: {class_column}")
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.groupby(class_column, sort=False, group_keys=False).head(int(per_class_limit)).to_csv(path, index=False)


def write_condition_gate_summary(
    gates: np.ndarray,
    condition_labels: list[str],
    path: Path,
    activation_threshold: float = 0.5,
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
                "active_rate": float((values >= float(activation_threshold)).mean()),
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


def behavior_alignment_metrics(
    predictions: pd.DataFrame,
    condition_labels: list[str],
    gold_column: str = "gold_tactic",
    prediction_column: str = "predicted_condition",
    ambiguous_label: str = AMBIGUOUS_LABEL,
) -> dict[str, Any]:
    if gold_column not in predictions.columns:
        return {"enabled": False, "reason": f"missing column: {gold_column}"}
    frame = predictions.copy()
    gold = frame[gold_column].fillna("").astype(str).str.strip()
    valid = gold != ""
    if not bool(valid.any()):
        return {"enabled": True, "labeled_rows": 0}
    y_true = gold[valid].to_numpy()
    y_pred = frame.loc[valid, prediction_column].fillna(ambiguous_label).astype(str).to_numpy()
    labels = list(dict.fromkeys([*condition_labels, ambiguous_label]))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    exact_non_ambiguous = y_pred != ambiguous_label
    return {
        "enabled": True,
        "labeled_rows": int(len(y_true)),
        "non_ambiguous_rate": float(exact_non_ambiguous.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "labels": labels,
        "classification_report": report,
    }


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


def plot_training_reconstruction_losses(
    history: list[dict[str, float]] | pd.DataFrame,
    path: Path,
) -> None:
    frame = pd.DataFrame(history)
    required = ["epoch", "val_recon_mse", "val_h_only_mse", "val_c_only_mse"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Training history missing required columns: {missing}")

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160)
    ax.plot(frame["epoch"], frame["val_recon_mse"], label="full", linewidth=1.8)
    ax.plot(frame["epoch"], frame["val_h_only_mse"], label="H-only", linewidth=1.8)
    ax.plot(frame["epoch"], frame["val_c_only_mse"], label="C-only", linewidth=1.8)
    ax.set_title("Validation reconstruction loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(frameon=False)
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


def _ordered_categories(values: np.ndarray, category_order: list[str] | None) -> list[str]:
    observed = {str(value) for value in values}
    ordered: list[str] = []
    if category_order is not None:
        for label in category_order:
            if label in observed and label not in ordered:
                ordered.append(label)
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def plot_umap_projection(
    features: np.ndarray,
    path: Path,
    title: str,
    config: dict[str, Any],
    categories: np.ndarray | list[str] | None = None,
    category_order: list[str] | None = None,
) -> None:
    max_samples = int(config.get("visualization_max_samples", 5000))
    seed = int(config.get("random_state", 42))
    selected = _visualization_sample(len(features), max_samples, seed)
    category_values = None
    if categories is not None:
        category_values = np.asarray(categories, dtype=object)
        if len(category_values) != len(features):
            raise ValueError("categories must have the same row count as features.")
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
    scatter = None
    if category_values is None:
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
    else:
        selected_categories = category_values[selected].astype(str)
        ordered = _ordered_categories(selected_categories, category_order)
        category_to_index = {label: index for index, label in enumerate(ordered)}
        palette = plt.get_cmap("tab20", max(len(ordered), 1))
        point_colors = [palette(category_to_index[value]) for value in selected_categories]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=10,
            alpha=0.72,
            c=point_colors,
            linewidths=0,
        )
        if ordered:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette(index),
                    markeredgecolor="none",
                    markersize=6,
                    label=label,
                )
                for index, label in enumerate(ordered)
            ]
            ax.legend(
                handles=handles,
                title="predicted condition",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=7,
                title_fontsize=8,
                frameon=False,
            )
    ax.set_title(f"{title} ({method}, n={len(selected)})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="sample order in plotted split")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
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
