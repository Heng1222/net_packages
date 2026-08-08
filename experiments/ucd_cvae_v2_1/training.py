from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import SplitIndices
from .model import UCDCVAE, compute_losses


@dataclass(slots=True)
class ScheduleState:
    phase: int
    multiplier: float
    projector_frozen: bool
    gradnorm_enabled: bool


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float | str]]
    best_epoch: int
    best_val_loss: float


class IndexedEmbeddingDataset(Dataset):
    def __init__(self, x: np.ndarray, indices: np.ndarray) -> None:
        self.x = x; self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self) -> int: return len(self.indices)
    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.x[int(self.indices[item])], dtype=np.float32, copy=True))


def make_loader(x: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool,
                seed: int, num_workers: int = 0) -> DataLoader:
    return DataLoader(IndexedEmbeddingDataset(x, indices), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=num_workers,
                      drop_last=shuffle and len(indices) > 1 and len(indices) % batch_size == 1)


def schedule_state(epoch: int, config: dict[str, Any]) -> ScheduleState:
    phase1_end = int(config.get("phase1_end", 5)); phase2_end = int(config.get("phase2_end", 15))
    if not 0 < phase1_end < phase2_end: raise ValueError("Expected 0 < phase1_end < phase2_end.")
    if epoch <= phase1_end: return ScheduleState(1, 0.0, True, False)
    if epoch <= phase2_end:
        multiplier = (epoch - phase1_end - 1) / max(phase2_end - phase1_end - 1, 1)
        return ScheduleState(2, float(np.clip(multiplier, 0.0, 1.0)), False, False)
    return ScheduleState(3, 1.0, False, True)


def gradient_norm(loss: torch.Tensor, parameters: Iterable[nn_parameter], epsilon: float = 0.0) -> float:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    squared = sum((gradient.detach().pow(2).sum() for gradient in gradients if gradient is not None),
                  start=loss.new_tensor(0.0))
    return float(torch.sqrt(squared + float(epsilon)).item())


# A small alias keeps the public annotation readable without importing private torch types.
nn_parameter = torch.nn.Parameter


def gradnorm_reconstruction_scale(reconstruction: torch.Tensor, sparse: torch.Tensor,
                                  align: torch.Tensor, gate_parameters: Iterable[torch.nn.Parameter],
                                  max_ratio: float = 10.0, epsilon: float = 1e-12
                                  ) -> tuple[float, dict[str, float]]:
    parameters = list(gate_parameters)
    g_rec = gradient_norm(reconstruction, parameters)
    g_sparse = gradient_norm(sparse, parameters)
    g_align = gradient_norm(align, parameters)
    if g_rec <= epsilon: scale = 1.0
    else:
        sparse_cap = 0.0 if g_sparse <= epsilon else max_ratio * g_sparse / g_rec
        align_cap = 0.0 if g_align <= epsilon else max_ratio * g_align / g_rec
        scale = float(np.clip(min(1.0, sparse_cap, align_cap), 0.0, 1.0))
    effective = scale * g_rec
    return scale, {"grad_rec": g_rec, "grad_sparse": g_sparse, "grad_align": g_align,
                   "effective_grad_rec": effective,
                   "ratio_rec_sparse": effective / max(g_sparse, epsilon),
                   "ratio_rec_align": effective / max(g_align, epsilon)}


def train_model(model: UCDCVAE, x: np.ndarray, split: SplitIndices, training: dict[str, Any],
                loss_weights: dict[str, float], device: torch.device, checkpoint_path: Path,
                seed: int, logger: logging.Logger | None = None) -> TrainingResult:
    model.to(device); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    train_loader = make_loader(x, split.train, int(training["batch_size"]), True, seed,
                               int(training.get("num_workers", 0)))
    val_loader = make_loader(x, split.val if len(split.val) else split.train,
                             int(training["batch_size"]), False, seed,
                             int(training.get("num_workers", 0)))
    history: list[dict[str, float | str]] = []; best = float("inf"); best_epoch = 0; stale = 0
    gradient_path = checkpoint_path.parent.parent / "metrics" / f"gradient_history_{model.geometry_variant}.csv"
    if gradient_path.exists(): gradient_path.unlink()
    gradient_header = True
    phase2_end = int(training.get("phase2_end", 15)); max_epochs = int(training["max_epochs"])
    tracked = ("loss", "reconstruction_loss", "kl_loss", "sparse_loss", "align_loss", "recon_cosine",
               "rec_scale", "grad_rec", "grad_sparse", "grad_align", "effective_grad_rec",
               "ratio_rec_sparse", "ratio_rec_align")
    for epoch in range(1, max_epochs + 1):
        state = schedule_state(epoch, training)
        model.concept_projector.requires_grad_(not state.projector_frozen)
        row: dict[str, float | str] = {"epoch": float(epoch), "phase": f"phase_{state.phase}",
                                       "anneal_multiplier": state.multiplier}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            is_train = phase == "train"; model.train(is_train); totals = {key: 0.0 for key in tracked}; count = 0
            context = torch.enable_grad() if is_train else torch.inference_mode()
            with context:
                batch_rows: list[dict[str, float]] = []
                for batch_index, xb in enumerate(loader):
                    xb = xb.to(device)
                    if is_train: optimizer.zero_grad(set_to_none=True)
                    output = model(xb, sample=is_train); raw = compute_losses(output, xb, model.alignment_temperature)
                    rec_term = float(loss_weights["reconstruction"]) * raw["reconstruction_loss"]
                    kl_term = float(loss_weights["kl"]) * raw["kl_loss"]
                    sparse_term = state.multiplier * float(loss_weights["sparse"]) * raw["sparse_loss"]
                    align_term = state.multiplier * float(loss_weights["align"]) * raw["align_loss"]
                    diagnostics = {key: 0.0 for key in tracked if key.startswith("grad") or key.startswith("ratio") or key == "effective_grad_rec"}
                    rec_scale = 1.0
                    if is_train and state.gradnorm_enabled:
                        rec_scale, diagnostics = gradnorm_reconstruction_scale(
                            rec_term, sparse_term, align_term, model.gate_encoder.parameters(),
                            float(training.get("gradnorm_max_ratio", 10.0)),
                            float(training.get("gradnorm_epsilon", 1e-12)))
                        if rec_scale == 0.0 and logger: logger.warning("epoch=%d batch reconstruction gate scale is zero", epoch)
                        batch_rows.append({"epoch": float(epoch), "batch": float(batch_index),
                            "reconstruction_weight": float(loss_weights["reconstruction"]) * rec_scale,
                            "kl_weight": float(loss_weights["kl"]),
                            "sparse_weight": state.multiplier * float(loss_weights["sparse"]),
                            "align_weight": state.multiplier * float(loss_weights["align"]),
                            "rec_scale": rec_scale, **diagnostics})
                    total = rec_scale * rec_term + kl_term + sparse_term + align_term
                    if is_train: total.backward(); optimizer.step()
                    values = {"loss": total, **{key: raw[key] for key in
                              ("reconstruction_loss", "kl_loss", "sparse_loss", "align_loss", "recon_cosine")},
                              "rec_scale": rec_scale, **diagnostics}
                    for key in tracked: totals[key] += float(values[key].detach() if torch.is_tensor(values[key]) else values[key]) * len(xb)
                    count += len(xb)
                if is_train and batch_rows:
                    pd.DataFrame(batch_rows).to_csv(gradient_path, mode="w" if gradient_header else "a",
                                                    header=gradient_header, index=False)
                    gradient_header = False
            row.update({f"{phase}_{key}": value / max(count, 1) for key, value in totals.items()})
        history.append(row)
        if epoch in {int(training.get("phase1_end", 5)), phase2_end}:
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "variant": model.geometry_variant},
                       checkpoint_path.with_name(f"{checkpoint_path.stem}_phase{state.phase}.pt"))
        if epoch > phase2_end:
            val_loss = float(row["val_loss"])
            if val_loss < best - 1e-10:
                best, best_epoch, stale = val_loss, epoch, 0
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                            "variant": model.geometry_variant}, checkpoint_path)
            else: stale += 1
        if logger:
            logger.info("variant=%s epoch=%d phase=%d anneal=%.3f train=%.6f val=%.6f scale=%.4f stale=%d",
                        model.geometry_variant, epoch, state.phase, state.multiplier,
                        row["train_loss"], row["val_loss"], row["train_rec_scale"], stale)
        if epoch > phase2_end and stale >= int(training["early_stopping_patience"]): break
    if best_epoch == 0:
        best_epoch = len(history); best = float(history[-1]["val_loss"])
        torch.save({"model_state": model.state_dict(), "epoch": best_epoch, "val_loss": best,
                    "variant": model.geometry_variant}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    pd.DataFrame(history).to_csv(checkpoint_path.parent.parent / "metrics" /
                                 f"training_history_{model.geometry_variant}.csv", index=False)
    return TrainingResult(history, best_epoch, best)
