from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .data import SplitIndices
from .model import MultiLabelDisentangledConditionalVAE


TRACKED = (
    "loss", "recon_nll", "recon_mse", "kl_loss", "decorrelation_loss",
    "sparse_loss", "gate_entropy_loss", "utility_loss", "residual_constraint_loss",
    "behavior_infonce_loss", "residual_adversary_loss",
)


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float
    pos_weight: np.ndarray


class _IndexedArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, indices: np.ndarray) -> None:
        self.x = x
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        # np.load(..., mmap_mode="r") rows are read-only. PyTorch requires a
        # writable backing array even though the training loop never mutates x.
        row = np.array(self.x[index], dtype=np.float32, copy=True)
        return torch.from_numpy(row), torch.tensor(index, dtype=torch.long)


def make_loader(
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        _IndexedArrayDataset(x, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        drop_last=shuffle and len(indices) > 1 and len(indices) % batch_size == 1,
    )


def tactic_pos_weight(targets: np.ndarray, train_indices: np.ndarray, maximum: float) -> np.ndarray:
    train = np.asarray(targets[train_indices], dtype=np.float32)
    positive = train.sum(axis=0)
    negative = len(train) - positive
    weights = np.ones(train.shape[1], dtype=np.float32)
    present = positive > 0
    weights[present] = negative[present] / positive[present]
    return np.clip(weights, 1.0, float(maximum)).astype(np.float32)


def _loss_batch(
    model: MultiLabelDisentangledConditionalVAE,
    x_batch: torch.Tensor,
    index_batch: torch.Tensor,
    targets: np.ndarray,
    conditions: torch.Tensor,
    device: torch.device,
    sample: bool,
    diagnostics: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    labels = torch.from_numpy(np.asarray(targets[index_batch.numpy()], dtype=np.float32)).to(device)
    output = model(x_batch, conditions, sample=sample)
    return output, model.loss(output, x_batch, labels, compute_diagnostics=diagnostics)


def train_model(
    model: MultiLabelDisentangledConditionalVAE,
    x: np.ndarray,
    targets: np.ndarray,
    condition_matrix: np.ndarray,
    split: SplitIndices,
    training: dict[str, Any],
    model_config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    logger: logging.Logger | None = None,
) -> TrainingResult:
    model.to(device)
    pos_weight = tactic_pos_weight(targets, split.train, float(training.get("max_pos_weight", 50.0)))
    model.set_tactic_pos_weight(torch.from_numpy(pos_weight).to(device))
    conditions = torch.from_numpy(np.asarray(condition_matrix, dtype=np.float32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loaders = {
        "train": make_loader(x, split.train, int(training["batch_size"]), True, seed, int(training.get("num_workers", 0))),
        "val": make_loader(x, split.val, int(training["batch_size"]), False, seed, int(training.get("num_workers", 0))),
    }
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        epoch_row: dict[str, float] = {"epoch": float(epoch)}
        for phase in ("train", "val"):
            model.train(phase == "train")
            totals = {key: 0.0 for key in TRACKED}
            aux_totals = {"h_only_mse": 0.0, "c_only_mse": 0.0}
            count = 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for x_batch, index_batch in loaders[phase]:
                    x_batch = x_batch.to(device)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    output, losses = _loss_batch(
                        model, x_batch, index_batch, targets, conditions, device, sample=phase == "train"
                    )
                    if phase == "train":
                        losses["loss"].backward()
                        optimizer.step()
                    for key in TRACKED:
                        totals[key] += float(losses[key].detach()) * len(x_batch)
                    if phase == "val":
                        aux = model.auxiliary_reconstructions(output)
                        for key, value in (("h_only_mse", aux["h_only"]), ("c_only_mse", aux["c_only"])):
                            mse = torch.nn.functional.mse_loss(value, x_batch, reduction="none").mean(dim=1)
                            aux_totals[key] += float(mse.mean()) * len(x_batch)
                    count += len(x_batch)
            epoch_row.update({f"{phase}_{key}": value / max(count, 1) for key, value in totals.items()})
            if phase == "val":
                epoch_row.update({f"val_{key}": value / max(count, 1) for key, value in aux_totals.items()})
        history.append(epoch_row)
        val_loss = epoch_row["val_loss"]
        if val_loss < best - 1e-10:
            best = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "val_loss": best,
                    "pos_weight": pos_weight,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if logger is not None:
            logger.info(
                "Epoch %d/%d train_loss=%.6f val_loss=%.6f val_recon_mse=%.6f best=%d/%.6f elapsed=%.1fs",
                epoch, int(training["max_epochs"]), epoch_row["train_loss"], val_loss,
                epoch_row["val_recon_mse"], best_epoch, best, time.perf_counter() - started,
            )
        if stale >= int(training["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    pd.DataFrame(history).to_csv(checkpoint_path.parent.parent / "metrics" / "training_history.csv", index=False)
    return TrainingResult(history, best_epoch, best, pos_weight)


@torch.inference_mode()
def extract_batches(
    model: MultiLabelDisentangledConditionalVAE,
    x: np.ndarray,
    targets: np.ndarray,
    condition_matrix: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    conditions = torch.from_numpy(np.asarray(condition_matrix, dtype=np.float32)).to(device)
    loader = make_loader(x, indices, batch_size, False, 0)
    parts = {key: [] for key in ("h", "c", "hc", "gates", "ablation_delta_mse", "recon_mse", "h_only_mse", "c_only_mse")}
    for x_batch, index_batch in loader:
        x_batch = x_batch.to(device)
        output, losses = _loss_batch(model, x_batch, index_batch, targets, conditions, device, False, True)
        aux = model.auxiliary_reconstructions(output)
        h = output["h_mu"]
        c = model.semantic_summary(output["conditions"], output["gates"])
        parts["h"].append(h.cpu().numpy())
        parts["c"].append(c.cpu().numpy())
        parts["hc"].append(torch.cat((h, c), dim=1).cpu().numpy())
        parts["gates"].append(output["gates"].cpu().numpy())
        parts["ablation_delta_mse"].append(losses["ablation_delta_mse"].cpu().numpy())
        parts["recon_mse"].append(losses["recon_mse_per_sample"].cpu().numpy())
        parts["h_only_mse"].append(torch.nn.functional.mse_loss(aux["h_only"], x_batch, reduction="none").mean(dim=1).cpu().numpy())
        parts["c_only_mse"].append(torch.nn.functional.mse_loss(aux["c_only"], x_batch, reduction="none").mean(dim=1).cpu().numpy())
    return {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in parts.items()}
