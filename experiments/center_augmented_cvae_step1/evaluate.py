from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .embedders import normalize_rows
from .model import CenterAugmentedCVAE
from .training import make_loader


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(target) - np.asarray(prediction)) ** 2))


def mean_cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    left, right = normalize_rows(target), normalize_rows(prediction)
    return float(np.mean(np.sum(left * right, axis=1)))


def reconstruction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {"recon_mse": mse(target, prediction), "recon_cosine": mean_cosine(target, prediction)}


def ablation_metrics(outputs: dict[str, np.ndarray], decode_matrix: np.ndarray,
                     seed: int = 42) -> tuple[dict[str, Any], pd.DataFrame]:
    x, residual, gates = outputs["x"], outputs["residual"], outputs["gates"]
    decode = np.asarray(decode_matrix, dtype=np.float32)
    full = residual + gates @ decode
    residual_only = residual
    zero_common = residual + gates[:, 1:] @ decode[1:]
    common_only = residual + gates[:, :1] @ decode[:1]
    shuffled_gates = gates[np.random.default_rng(seed).permutation(len(gates))]
    shuffled = residual + shuffled_gates @ decode
    summary: dict[str, Any] = {
        "full": reconstruction_metrics(x, full),
        "zero_all_conditions": reconstruction_metrics(x, residual_only),
        "zero_common_condition": reconstruction_metrics(x, zero_common),
        "zero_tactic_conditions": reconstruction_metrics(x, common_only),
        "shuffled_gates": reconstruction_metrics(x, shuffled),
        "condition_gain": mse(x, residual_only) - mse(x, full),
        "common_condition_gain": mse(x, zero_common) - mse(x, full),
        "tactic_condition_gain": mse(x, common_only) - mse(x, full),
        "mean_residual_norm": float(np.linalg.norm(residual, axis=1).mean()),
        "mean_condition_norm": float(np.linalg.norm(outputs["condition"], axis=1).mean()),
    }
    rows = []
    full_per_sample = ((x - full) ** 2).mean(axis=1)
    for index in range(1, len(decode)):
        ablated_gates = gates.copy(); ablated_gates[:, index] = 0.0
        prediction = residual + ablated_gates @ decode
        ablated_per_sample = ((x - prediction) ** 2).mean(axis=1)
        rows.append({"condition_index": index, "mean_delta_mse": float((ablated_per_sample - full_per_sample).mean()),
                     "p50_delta_mse": float(np.median(ablated_per_sample - full_per_sample))})
    return summary, pd.DataFrame(rows)


@torch.inference_mode()
def evaluate_main_model(model: CenterAugmentedCVAE, x: np.ndarray, indices: np.ndarray,
                        decode_matrix: np.ndarray, gate_matrix: np.ndarray,
                        device: torch.device, batch_size: int, seed: int = 42
                        ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    model.eval(); model.to(device)
    decode = torch.from_numpy(np.asarray(decode_matrix, dtype=np.float32)).to(device)
    gate = torch.from_numpy(np.asarray(gate_matrix, dtype=np.float32)).to(device)
    names = ("full", "zero_all_conditions", "zero_common_condition",
             "zero_tactic_conditions", "shuffled_gates")
    mse_sums = {name: 0.0 for name in names}; cosine_sums = {name: 0.0 for name in names}
    per_condition = np.zeros(model.condition_count - 1, dtype=np.float64)
    gate_parts: list[np.ndarray] = []; count = 0; residual_norm = 0.0; condition_norm = 0.0
    generator = torch.Generator(device=device).manual_seed(seed)
    for xb in make_loader(x, indices, batch_size, False, 0):
        xb = xb.to(device); output = model(xb, decode, gate, sample=False)
        gates, residual = output["gates"], output["residual_component"]
        shuffled = gates[torch.randperm(len(gates), generator=generator, device=device)]
        predictions = {
            "full": output["x_recon"], "zero_all_conditions": residual,
            "zero_common_condition": residual + gates[:, 1:] @ decode[1:],
            "zero_tactic_conditions": residual + gates[:, :1] @ decode[:1],
            "shuffled_gates": residual + shuffled @ decode,
        }
        full_per = (predictions["full"] - xb).pow(2).mean(dim=1)
        for name, prediction in predictions.items():
            mse_sums[name] += float((prediction - xb).pow(2).mean(dim=1).sum())
            cosine_sums[name] += float(torch.nn.functional.cosine_similarity(prediction, xb, dim=1).sum())
        for index in range(1, model.condition_count):
            changed = gates.clone(); changed[:, index] = 0.0
            ablated = residual + changed @ decode
            per_condition[index - 1] += float(((ablated - xb).pow(2).mean(dim=1) - full_per).sum())
        gate_parts.append(gates.cpu().numpy()); residual_norm += float(torch.linalg.vector_norm(residual, dim=1).sum())
        condition_norm += float(torch.linalg.vector_norm(output["condition_component"], dim=1).sum()); count += len(xb)
    metrics = {name: {"recon_mse": mse_sums[name] / max(count, 1),
                      "recon_cosine": cosine_sums[name] / max(count, 1)} for name in names}
    metrics.update({
        "condition_gain": metrics["zero_all_conditions"]["recon_mse"] - metrics["full"]["recon_mse"],
        "common_condition_gain": metrics["zero_common_condition"]["recon_mse"] - metrics["full"]["recon_mse"],
        "tactic_condition_gain": metrics["zero_tactic_conditions"]["recon_mse"] - metrics["full"]["recon_mse"],
        "mean_residual_norm": residual_norm / max(count, 1),
        "mean_condition_norm": condition_norm / max(count, 1),
    })
    rows = pd.DataFrame({"condition_index": np.arange(1, model.condition_count),
                         "mean_delta_mse": per_condition / max(count, 1)})
    return metrics, rows, np.vstack(gate_parts).astype(np.float32)


def gate_summary(gates: np.ndarray, labels: list[str], threshold: float) -> pd.DataFrame:
    rows = []
    for index, label in enumerate(labels):
        values = gates[:, index]
        rows.append({"condition_index": index, "condition": label, "mean_gate": float(values.mean()),
                     "std_gate": float(values.std()), "p10_gate": float(np.quantile(values, 0.1)),
                     "p50_gate": float(np.median(values)), "p90_gate": float(np.quantile(values, 0.9)),
                     "active_rate": float(np.mean(values >= threshold))})
    return pd.DataFrame(rows)


def prediction_frame(metadata: pd.DataFrame, indices: np.ndarray, gates: np.ndarray,
                     labels: list[str], threshold: float) -> pd.DataFrame:
    frame = metadata.iloc[indices][[c for c in ("sample_id", "datetime", "source_label") if c in metadata]].reset_index(drop=True)
    tactic_gates = gates[:, 1:]; tactic_labels = np.asarray(labels[1:], dtype=object)
    winners = tactic_gates.argmax(axis=1)
    frame["predicted_tactic"] = tactic_labels[winners]
    frame["predicted_tactic_gate"] = tactic_gates[np.arange(len(gates)), winners]
    frame["common_gate"] = gates[:, 0]
    frame["active_condition_count"] = (gates >= threshold).sum(axis=1)
    for index, label in enumerate(labels): frame[f"gate__{label}"] = gates[:, index]
    return frame


def random_condition_matrices(decode_matrix: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    decode = np.asarray(decode_matrix, dtype=np.float32)
    random = np.random.default_rng(seed).normal(size=decode.shape).astype(np.float32)
    random = normalize_rows(random) * np.linalg.norm(decode, axis=1, keepdims=True)
    return random.astype(np.float32), normalize_rows(random)


def _clean_gold_label(value: Any) -> str:
    text = str(value).strip()
    if text.startswith(("{", "[", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (set, list, tuple)) and len(parsed) == 1: return str(next(iter(parsed)))
        except (SyntaxError, ValueError):
            pass
    return text


def semantic_gates_numpy(x: np.ndarray, gate_matrix: np.ndarray, temperature: float) -> np.ndarray:
    cosine = normalize_rows(x) @ normalize_rows(gate_matrix).T
    logits = np.clip(cosine / float(temperature), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def golden_alignment(metadata: pd.DataFrame, x: np.ndarray, split_names: np.ndarray,
                     gate_matrix: np.ndarray, labels: list[str], config: dict[str, Any],
                     output_dir: Path, temperature: float) -> dict[str, Any]:
    path = Path(config["golden_path"])
    if not path.is_file(): return {"available": False, "reason": f"Missing golden file: {path}"}
    gold = pd.read_csv(path, usecols=[config["golden_sample_id_col"], config["golden_label_col"]], dtype=str)
    gold = gold.rename(columns={config["golden_sample_id_col"]: "sample_id",
                                config["golden_label_col"]: "gold_tactic"})
    gold["gold_tactic"] = gold["gold_tactic"].map(_clean_gold_label)
    lookup = metadata[["sample_id"]].copy(); lookup["row_index"] = np.arange(len(metadata)); lookup["split"] = split_names
    matched = gold.merge(lookup, on="sample_id", how="inner").drop_duplicates(["sample_id", "gold_tactic"])
    tactic_labels = labels[1:]; label_to_index = {label: index for index, label in enumerate(tactic_labels)}
    normal_label = "Normal (TA9000)"
    matched = matched[matched["gold_tactic"].isin([*tactic_labels, normal_label])].copy()
    if matched.empty: return {"available": True, "matched_rows": 0, "warning": "No recognized golden labels matched Step1 rows."}
    row_indices = matched["row_index"].astype(int).to_numpy()
    gates = semantic_gates_numpy(np.asarray(x[row_indices], dtype=np.float32), gate_matrix, temperature)
    tactic_gates = gates[:, 1:]; predicted_all = tactic_gates.argmax(axis=1)
    matched["predicted_tactic"] = np.asarray(tactic_labels, dtype=object)[predicted_all]
    matched["common_gate"] = gates[:, 0]
    matched["gold_gate"] = np.nan; matched["gold_gate_rank"] = np.nan
    is_tactic = matched["gold_tactic"].isin(label_to_index).to_numpy()
    tactic_rows = np.flatnonzero(is_tactic)
    targets = matched.iloc[tactic_rows]["gold_tactic"].map(label_to_index).astype(int).to_numpy()
    predicted = predicted_all[tactic_rows]
    ranks = np.argsort(np.argsort(-tactic_gates[tactic_rows], axis=1), axis=1)
    if len(tactic_rows):
        matched.loc[matched.index[tactic_rows], "gold_gate"] = tactic_gates[tactic_rows, targets]
        matched.loc[matched.index[tactic_rows], "gold_gate_rank"] = ranks[np.arange(len(targets)), targets] + 1
    matched.to_csv(output_dir / "golden_gate_alignment_predictions.csv", index=False)
    if not len(tactic_rows):
        return {"available": True, "evaluation_type": "transductive_semantic_alignment_diagnostic",
                "labels_used_for_training": False, "matched_rows": len(matched), "matched_tactic_rows": 0,
                "matched_normal_rows": int((matched["gold_tactic"] == normal_label).sum()),
                "mean_common_gate_normal": float(matched.loc[matched["gold_tactic"] == normal_label, "common_gate"].mean())}
    report = classification_report(targets, predicted, labels=np.arange(len(tactic_labels)),
                                   target_names=tactic_labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, predicted, labels=np.arange(len(tactic_labels)))
    pd.DataFrame(matrix, index=tactic_labels, columns=tactic_labels).to_csv(output_dir / "golden_gate_confusion_matrix.csv")
    return {
        "available": True, "evaluation_type": "transductive_semantic_alignment_diagnostic",
        "labels_used_for_training": False, "matched_rows": len(matched),
        "matched_tactic_rows": int(is_tactic.sum()),
        "matched_normal_rows": int((matched["gold_tactic"] == normal_label).sum()),
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "mean_gold_gate": float(matched.loc[is_tactic, "gold_gate"].mean()),
        "mean_non_gold_tactic_gate": float(((tactic_gates[tactic_rows].sum(axis=1) - tactic_gates[tactic_rows, targets]) / max(len(tactic_labels) - 1, 1)).mean()),
        "mean_gold_gate_rank": float(matched.loc[is_tactic, "gold_gate_rank"].mean()),
        "mean_common_gate_malicious": float(matched.loc[is_tactic, "common_gate"].mean()),
        "mean_common_gate_normal": float(matched.loc[matched["gold_tactic"] == normal_label, "common_gate"].mean()),
        "matched_by_split": {str(k): int(v) for k, v in matched["split"].value_counts().items()},
        "classification_report": report,
    }
