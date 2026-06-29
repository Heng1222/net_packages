from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..data.condition_loader import ConditionSet
from ..models.common import reconstruction_per_sample
from ..models.cvae import ConditionalVAE
from ..utils.io import write_json
from .metrics import classification_metrics


@torch.inference_mode()
def compatibility_scores(
    model: ConditionalVAE,
    x: np.ndarray,
    conditions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    condition_tensor = torch.from_numpy(np.asarray(conditions, dtype=np.float32)).to(device)
    rows: list[np.ndarray] = []
    candidate_count = len(conditions)
    for start in range(0, len(x), batch_size):
        x_batch = torch.from_numpy(np.asarray(x[start : start + batch_size], dtype=np.float32)).to(device)
        repeated_x = x_batch.repeat_interleave(candidate_count, dim=0)
        repeated_c = condition_tensor.repeat(len(x_batch), 1)
        output = model(repeated_x, repeated_c, sample=False)
        scores = reconstruction_per_sample(output["x_recon"], repeated_x, model.reconstruction_loss)
        rows.append(scores.reshape(len(x_batch), candidate_count).cpu().numpy())
    return np.vstack(rows).astype(np.float32)


def run_compatibility_test(
    model: ConditionalVAE,
    x_test: np.ndarray,
    sample_ids: np.ndarray,
    true_labels: np.ndarray | None,
    condition_set: ConditionSet,
    device: torch.device,
    batch_size: int,
    output_dir: str | Path,
    prefix: str = "compatibility",
) -> dict[str, Any]:
    output = Path(output_dir)
    scores = compatibility_scores(model, x_test, condition_set.matrix, device, batch_size)
    predicted = np.asarray([condition_set.labels[index] for index in np.argmin(scores, axis=1)], dtype=str)
    np.savez_compressed(
        output / f"{prefix}_score_matrix.npz",
        scores=scores,
        sample_id=np.asarray(sample_ids, dtype=str),
        candidate_labels=np.asarray(condition_set.labels, dtype=str),
        predicted_label=predicted,
    )
    frame = pd.DataFrame(scores, columns=[f"score::{label}" for label in condition_set.labels])
    frame.insert(0, "predicted_label", predicted)
    if true_labels is not None:
        frame.insert(0, "true_label", true_labels)
    frame.insert(0, "sample_id", sample_ids)
    frame.to_csv(output / f"{prefix}_results.csv", index=False)

    candidate_means = pd.DataFrame({"candidate_tactic": condition_set.labels, "mean_score": scores.mean(axis=0)})
    candidate_means.to_csv(output / f"{prefix}_candidate_mean_scores.csv", index=False)
    result: dict[str, Any] = {
        "score": "deterministic_reconstruction",
        "num_samples": len(x_test),
        "num_candidates": len(condition_set.labels),
        "candidate_mean_scores": dict(zip(condition_set.labels, scores.mean(axis=0).tolist(), strict=True)),
    }
    if true_labels is not None:
        result["classification"] = classification_metrics(true_labels.astype(str), predicted, condition_set.labels)
        mean_rows = []
        for label in condition_set.labels:
            mask = true_labels == label
            if mask.any():
                values = scores[mask].mean(axis=0)
                mean_rows.append({"true_tactic": label, **dict(zip(condition_set.labels, values, strict=True))})
        pd.DataFrame(mean_rows).to_csv(output / f"{prefix}_mean_by_true_tactic.csv", index=False)
    write_json(result, output / f"{prefix}_metrics.json")
    return result
