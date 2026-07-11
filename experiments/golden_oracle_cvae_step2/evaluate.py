from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(labels)), target_names=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
    }


def plot_histories(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4), dpi=150)
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, histories.items(), strict=True):
        epochs = [row["epoch"] for row in rows]
        ax.plot(epochs, [row["train_loss"] for row in rows], label="train")
        ax.plot(epochs, [row["val_loss"] for row in rows], label="val")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_confusions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    labels: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(predictions), figsize=(7 * len(predictions), 6), dpi=150)
    if len(predictions) == 1:
        axes = [axes]
    for ax, (name, pred) in zip(axes, predictions.items(), strict=True):
        matrix = confusion_matrix(y_true, pred, labels=np.arange(len(labels)), normalize="true")
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(name)
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("gold")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
