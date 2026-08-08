from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F

from .data import SplitIndices
from .model import UCDCVAE, compute_losses
from .training import make_loader


def _decode(model: UCDCVAE, latent: torch.Tensor) -> torch.Tensor:
    return model.decoder_out(model.decoder(latent))


@torch.inference_mode()
def evaluate_model(model: UCDCVAE, x: np.ndarray, indices: np.ndarray, device: torch.device,
                   batch_size: int, seed: int = 42) -> tuple[dict[str, Any], np.ndarray]:
    model.eval(); model.to(device); totals: dict[str, float] = {}; count = 0; gate_parts = []
    ablation_sums = np.zeros(15, dtype=np.float64); generator = torch.Generator(device=device).manual_seed(seed)
    for xb in make_loader(x, indices, batch_size, False, seed):
        xb = xb.to(device); output = model(xb, sample=False); losses = compute_losses(output, xb, model.alignment_temperature)
        basis, gates = output["projected_basis"], output["gates"]
        residual_hat = _decode(model, output["h_res_perp"]); concept_hat = _decode(model, output["concept_component"])
        permutation = torch.randperm(len(xb), generator=generator, device=device)
        shuffled_concept = gates[permutation] @ basis
        shuffled_hat = _decode(model, output["h_res_perp"] + shuffled_concept)
        predictions = {"full": output["x_hat"], "residual_only": residual_hat,
                       "concept_only": concept_hat, "shuffled_gates": shuffled_hat}
        for name, prediction in predictions.items():
            totals[f"{name}_cosine"] = totals.get(f"{name}_cosine", 0.0) + float(F.cosine_similarity(prediction, xb, dim=1).sum())
        for key in ("kl_loss", "sparse_loss", "align_loss"):
            totals[key] = totals.get(key, 0.0) + float(losses[key]) * len(xb)
        full_loss = 1.0 - F.cosine_similarity(output["x_hat"], xb, dim=1)
        for index in range(15):
            changed = gates.clone(); changed[:, index] = 0.0
            changed_hat = _decode(model, output["h_res_perp"] + changed @ basis)
            changed_loss = 1.0 - F.cosine_similarity(changed_hat, xb, dim=1)
            ablation_sums[index] += float((changed_loss - full_loss).sum())
        orth_error = float(torch.max(torch.abs(output["h_res_perp"] @ basis.T)))
        totals["max_residual_projection"] = max(totals.get("max_residual_projection", 0.0), orth_error)
        gate_parts.append(gates.cpu().numpy()); count += len(xb)
    gates = np.vstack(gate_parts).astype(np.float32)
    summary: dict[str, Any] = {key: value / max(count, 1) for key, value in totals.items()
                                 if key != "max_residual_projection"}
    summary["max_residual_projection"] = totals.get("max_residual_projection", 0.0)
    summary["condition_gain_cosine"] = summary["full_cosine"] - summary["residual_only_cosine"]
    summary["shuffled_gate_gain_cosine"] = summary["full_cosine"] - summary["shuffled_gates_cosine"]
    summary["basis_orthogonality_error"] = float(np.max(np.abs(
        model.projected_basis().detach().cpu().numpy() @ model.projected_basis().detach().cpu().numpy().T - np.eye(15))))
    summary["gate_summary"] = {"mean": gates.mean(axis=0), "std": gates.std(axis=0),
                                "p10": np.quantile(gates, 0.1, axis=0),
                                "p50": np.quantile(gates, 0.5, axis=0),
                                "p90": np.quantile(gates, 0.9, axis=0)}
    summary["per_condition_ablation_delta_cosine_loss"] = ablation_sums / max(count, 1)
    return summary, gates


def _clean_label(value: Any) -> str:
    text = str(value).strip()
    if text.startswith(("{", "[", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (set, list, tuple)) and len(parsed) == 1: return str(next(iter(parsed)))
        except (SyntaxError, ValueError): pass
    return text


def prediction_frame(metadata: pd.DataFrame, indices: np.ndarray, gates: np.ndarray,
                     labels: list[str]) -> pd.DataFrame:
    frame = metadata.iloc[indices][[column for column in ("sample_id", "datetime") if column in metadata]].reset_index(drop=True)
    winners = gates[:, 1:].argmax(axis=1); tactic_labels = np.asarray(labels[1:], dtype=object)
    frame["common_evidence"] = gates[:, 0]; frame["predicted_tactic"] = tactic_labels[winners]
    frame["predicted_tactic_evidence"] = gates[np.arange(len(gates)), winners + 1]
    for index, label in enumerate(labels): frame[f"evidence__{label}"] = gates[:, index]
    return frame


def golden_diagnostics(metadata: pd.DataFrame, indices: np.ndarray, gates: np.ndarray,
                       labels: list[str], config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    path = Path(config["golden_path"])
    if not path.is_file(): return {"available": False, "reason": f"Missing golden file: {path}"}
    gold = pd.read_csv(path, usecols=[config["golden_sample_id_col"], config["golden_label_col"]], dtype=str)
    gold = gold.rename(columns={config["golden_sample_id_col"]: "sample_id",
                                config["golden_label_col"]: "gold_tactic"})
    gold["gold_tactic"] = gold["gold_tactic"].map(_clean_label)
    lookup = pd.DataFrame({"sample_id": metadata.iloc[indices]["sample_id"].astype(str).to_numpy(),
                           "gate_row": np.arange(len(indices))})
    matched = gold.merge(lookup, on="sample_id", how="inner").drop_duplicates(["sample_id", "gold_tactic"])
    if matched.empty: return {"available": True, "matched_rows": 0}
    gate_rows = matched["gate_row"].astype(int).to_numpy(); matched_gates = gates[gate_rows]
    normal_label = str(config.get("normal_label", "Normal (TA9000)")); is_attack = (matched["gold_tactic"] != normal_label).astype(int).to_numpy()
    summary: dict[str, Any] = {"available": True, "labels_used_for_training": False,
                               "matched_rows": len(matched), "normal_rows": int((is_attack == 0).sum()),
                               "attack_rows": int(is_attack.sum())}
    if len(np.unique(is_attack)) == 2:
        summary["common_auroc"] = float(roc_auc_score(is_attack, matched_gates[:, 0]))
        summary["common_auprc"] = float(average_precision_score(is_attack, matched_gates[:, 0]))
    tactic_labels = labels[1:]; label_to_index = {label: index for index, label in enumerate(tactic_labels)}
    supported = matched["gold_tactic"].isin(label_to_index).to_numpy(); supported_rows = np.flatnonzero(supported)
    coverage = matched["gold_tactic"].value_counts().to_dict(); summary["label_coverage"] = {str(k): int(v) for k, v in coverage.items()}
    summary["unsupported_tactics"] = [label for label in tactic_labels if coverage.get(label, 0) == 0]
    if len(supported_rows):
        targets = matched.iloc[supported_rows]["gold_tactic"].map(label_to_index).astype(int).to_numpy()
        predicted = matched_gates[supported_rows, 1:].argmax(axis=1)
        active_labels = sorted(set(targets.tolist()))
        summary.update({"supported_rows": len(supported_rows),
                        "top1_accuracy": float(accuracy_score(targets, predicted)),
                        "macro_f1_supported": float(f1_score(targets, predicted, labels=active_labels,
                                                              average="macro", zero_division=0)),
                        "micro_f1_supported": float(f1_score(targets, predicted, labels=active_labels,
                                                              average="micro", zero_division=0))})
        ranks = np.argsort(np.argsort(-matched_gates[supported_rows, 1:], axis=1), axis=1)
        summary["mean_gold_rank"] = float((ranks[np.arange(len(targets)), targets] + 1).mean())
    matched["common_evidence"] = matched_gates[:, 0]
    matched["predicted_tactic"] = np.asarray(tactic_labels, dtype=object)[matched_gates[:, 1:].argmax(axis=1)]
    matched.to_csv(output_path, index=False)
    return summary
