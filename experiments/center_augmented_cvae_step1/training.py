from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .data import SplitIndices
from .model import CenterAugmentedCVAE, PlainVAE


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
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


def unsupervised_loss(output: dict[str, torch.Tensor], target: torch.Tensor,
                      weights: dict[str, float]) -> dict[str, torch.Tensor]:
    per_sample_mse = (output["x_recon"] - target).pow(2).mean(dim=1)
    reconstruction = per_sample_mse.mean()
    per_sample_kl = -0.5 * torch.sum(
        1.0 + output["z_logvar"] - output["z_mu"].pow(2) - output["z_logvar"].exp(), dim=1
    )
    kl = per_sample_kl.mean()
    cosine = F.cosine_similarity(output["x_recon"], target, dim=1).mean()
    total = float(weights["reconstruction"]) * reconstruction + float(weights["kl"]) * kl
    return {"loss": total, "recon_mse": reconstruction, "kl_loss": kl,
            "recon_cosine": cosine, "recon_mse_per_sample": per_sample_mse,
            "kl_per_sample": per_sample_kl}


def _forward(model: CenterAugmentedCVAE | PlainVAE, xb: torch.Tensor,
             decode_matrix: torch.Tensor | None, gate_matrix: torch.Tensor | None,
             sample: bool) -> dict[str, torch.Tensor]:
    if isinstance(model, CenterAugmentedCVAE):
        if decode_matrix is None or gate_matrix is None: raise ValueError("CVAE requires condition matrices.")
        return model(xb, decode_matrix, gate_matrix, sample=sample)
    return model(xb, sample=sample)


def train_model(model: CenterAugmentedCVAE | PlainVAE, x: np.ndarray, split: SplitIndices,
                training: dict[str, Any], loss_weights: dict[str, float], device: torch.device,
                checkpoint_path: Path, seed: int, logger: logging.Logger | None = None,
                decode_matrix: np.ndarray | None = None,
                gate_matrix: np.ndarray | None = None) -> TrainingResult:
    model.to(device); checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    decode_tensor = torch.from_numpy(np.asarray(decode_matrix, dtype=np.float32)).to(device) if decode_matrix is not None else None
    gate_tensor = torch.from_numpy(np.asarray(gate_matrix, dtype=np.float32)).to(device) if gate_matrix is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    train_loader = make_loader(x, split.train, int(training["batch_size"]), True, seed,
                               int(training.get("num_workers", 0)))
    val_indices = split.val if len(split.val) else split.train
    val_loader = make_loader(x, val_indices, int(training["batch_size"]), False, seed,
                             int(training.get("num_workers", 0)))
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    tracked = ("loss", "recon_mse", "kl_loss", "recon_cosine")
    for epoch in range(1, int(training["max_epochs"]) + 1):
        row: dict[str, float] = {"epoch": float(epoch)}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train"); totals = {key: 0.0 for key in tracked}; count = 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for xb in loader:
                    xb = xb.to(device)
                    if phase == "train": optimizer.zero_grad(set_to_none=True)
                    output = _forward(model, xb, decode_tensor, gate_tensor, sample=phase == "train")
                    losses = unsupervised_loss(output, xb, loss_weights)
                    if phase == "train": losses["loss"].backward(); optimizer.step()
                    for key in tracked: totals[key] += float(losses[key].detach()) * len(xb)
                    count += len(xb)
            row.update({f"{phase}_{key}": value / max(count, 1) for key, value in totals.items()})
        history.append(row); val_loss = row["val_loss"]
        if val_loss < best - 1e-10:
            best, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
        else:
            stale += 1
        if logger:
            logger.info("epoch=%d train=%.6f val=%.6f val_mse=%.6f val_kl=%.6f stale=%d",
                        epoch, row["train_loss"], val_loss, row["val_recon_mse"],
                        row["val_kl_loss"], stale)
        if stale >= int(training["early_stopping_patience"]): break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    pd.DataFrame(history).to_csv(checkpoint_path.parent.parent / "metrics" /
                                 f"{checkpoint_path.stem}_training_history.csv", index=False)
    return TrainingResult(history, best_epoch, best)


@torch.inference_mode()
def extract_main_outputs(model: CenterAugmentedCVAE, x: np.ndarray, indices: np.ndarray,
                         decode_matrix: np.ndarray, gate_matrix: np.ndarray,
                         device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval(); model.to(device)
    decode = torch.from_numpy(np.asarray(decode_matrix, dtype=np.float32)).to(device)
    gate = torch.from_numpy(np.asarray(gate_matrix, dtype=np.float32)).to(device)
    result: dict[str, list[np.ndarray]] = {key: [] for key in
        ("x", "x_recon", "residual", "condition", "gates", "gate_cosine", "z_mu")}
    for xb in make_loader(x, indices, batch_size, False, 0):
        xb = xb.to(device); output = model(xb, decode, gate, sample=False)
        values = {"x": xb, "x_recon": output["x_recon"],
                  "residual": output["residual_component"], "condition": output["condition_component"],
                  "gates": output["gates"], "gate_cosine": output["gate_cosine"], "z_mu": output["z_mu"]}
        for key, value in values.items(): result[key].append(value.cpu().numpy())
    return {key: np.vstack(parts).astype(np.float32) for key, parts in result.items()}


@torch.inference_mode()
def evaluate_plain_vae(model: PlainVAE, x: np.ndarray, indices: np.ndarray,
                       device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval(); model.to(device); total_mse = 0.0; total_cosine = 0.0; count = 0
    for xb in make_loader(x, indices, batch_size, False, 0):
        xb = xb.to(device); output = model(xb, sample=False)
        total_mse += float((output["x_recon"] - xb).pow(2).mean(dim=1).sum())
        total_cosine += float(F.cosine_similarity(output["x_recon"], xb, dim=1).sum())
        count += len(xb)
    return {"recon_mse": total_mse / max(count, 1), "recon_cosine": total_cosine / max(count, 1)}
