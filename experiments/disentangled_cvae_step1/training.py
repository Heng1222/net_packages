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
from .model import DisentangledConditionalVAE


TRACKED = (
    "loss",
    "recon_nll",
    "recon_mse",
    "kl_loss",
    "decorrelation_loss",
    "sparse_loss",
    "gate_entropy_loss",
    "utility_loss",
    "residual_constraint_loss",
)


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float


class MemmapDataset(Dataset):
    def __init__(self, x: np.ndarray, indices: np.ndarray) -> None:
        self.x = x
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        return (
            torch.from_numpy(np.asarray(self.x[index], dtype=np.float32)),
            torch.tensor(index, dtype=torch.long),
        )


def _format_elapsed(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{remainder:02d}s"
    if minutes:
        return f"{minutes:d}m{remainder:02d}s"
    return f"{remainder:d}s"


def make_loader(
    x: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MemmapDataset(x, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=shuffle and len(indices) > 1 and len(indices) % batch_size == 1,
    )


def save_checkpoint(
    path: Path,
    model: DisentangledConditionalVAE,
    optimizer: torch.optim.Optimizer,
    model_config: dict[str, Any],
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def train_model(
    model: DisentangledConditionalVAE,
    x: np.ndarray,
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
    conditions = torch.from_numpy(np.asarray(condition_matrix, dtype=np.float32)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_loader = make_loader(
        x,
        split.train,
        int(training["batch_size"]),
        True,
        seed,
        int(training.get("num_workers", 0)),
    )
    val_loader = make_loader(
        x,
        split.val if len(split.val) else split.train,
        int(training["batch_size"]),
        False,
        seed,
        int(training.get("num_workers", 0)),
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    max_epochs = int(training["max_epochs"])
    patience = int(training["early_stopping_patience"])
    batch_size = int(training["batch_size"])
    if logger is not None:
        logger.info(
            "Training started: epochs=%d batch_size=%d train_rows=%d val_rows=%d "
            "train_batches=%d val_batches=%d conditions=%d",
            max_epochs,
            batch_size,
            len(split.train),
            len(split.val) if len(split.val) else len(split.train),
            len(train_loader),
            len(val_loader),
            conditions.shape[0],
        )
    started_at = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        train_totals = {key: 0.0 for key in TRACKED}
        train_count = 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch, conditions, sample=True)
            losses = model.loss(output, x_batch)
            losses["loss"].backward()
            optimizer.step()
            for key in TRACKED:
                train_totals[key] += float(losses[key].detach()) * len(x_batch)
            train_count += len(x_batch)

        model.eval()
        val_totals = {key: 0.0 for key in TRACKED}
        val_aux_totals = {"h_only_mse": 0.0, "c_only_mse": 0.0}
        val_count = 0
        with torch.inference_mode():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch, conditions, sample=False)
                losses = model.loss(output, x_batch)
                aux = model.auxiliary_reconstructions(output)
                for key in TRACKED:
                    val_totals[key] += float(losses[key]) * len(x_batch)
                val_aux_totals["h_only_mse"] += float(
                    torch.nn.functional.mse_loss(aux["h_only"], x_batch, reduction="none").mean(dim=1).mean()
                ) * len(x_batch)
                val_aux_totals["c_only_mse"] += float(
                    torch.nn.functional.mse_loss(aux["c_only"], x_batch, reduction="none").mean(dim=1).mean()
                ) * len(x_batch)
                val_count += len(x_batch)
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update({f"train_{key}": value / max(train_count, 1) for key, value in train_totals.items()})
        row.update({f"val_{key}": value / max(val_count, 1) for key, value in val_totals.items()})
        row.update({f"val_{key}": value / max(val_count, 1) for key, value in val_aux_totals.items()})
        history.append(row)
        val_loss = row["val_loss"]
        improved = val_loss < best - 1e-10
        status = ""
        if improved:
            best = val_loss
            best_epoch = epoch
            stale = 0
            save_checkpoint(checkpoint_path, model, optimizer, model_config, epoch, best)
            status = "best"
        else:
            stale += 1
            status = f"stale={stale}/{patience}"
        if logger is not None:
            logger.info(
                "Epoch %d/%d | train_loss=%.6f train_recon_mse=%.6f | "
                "val_loss=%.6f val_recon_mse=%.6f val_h_only_mse=%.6f val_c_only_mse=%.6f | "
                "best_val_loss=%.6f best_epoch=%d %s | epoch_time=%s elapsed=%s",
                epoch,
                max_epochs,
                row["train_loss"],
                row["train_recon_mse"],
                row["val_loss"],
                row["val_recon_mse"],
                row["val_h_only_mse"],
                row["val_c_only_mse"],
                best,
                best_epoch,
                status,
                _format_elapsed(time.perf_counter() - epoch_started_at),
                _format_elapsed(time.perf_counter() - started_at),
            )
        if not improved and stale >= patience:
            if logger is not None:
                logger.info("Early stopping at epoch %d after %d stale epoch(s).", epoch, stale)
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    pd.DataFrame(history).to_csv(checkpoint_path.parent.parent / "metrics" / "training_history.csv", index=False)
    if logger is not None:
        logger.info(
            "Training finished: best_epoch=%d best_val_loss=%.6f history_rows=%d",
            best_epoch,
            best,
            len(history),
        )
    return TrainingResult(history, best_epoch, best)


@torch.inference_mode()
def extract_batches(
    model: DisentangledConditionalVAE,
    x: np.ndarray,
    condition_matrix: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | dict[str, float]]:
    model.eval()
    conditions = torch.from_numpy(np.asarray(condition_matrix, dtype=np.float32)).to(device)
    loader = make_loader(x, indices, batch_size, False, 0)
    h_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    hc_parts: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    ablations: list[np.ndarray] = []
    totals = {"recon_mse": 0.0, "h_only_mse": 0.0, "c_only_mse": 0.0, "loss": 0.0}
    count = 0
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        output = model(x_batch, conditions, sample=False)
        losses = model.loss(output, x_batch)
        aux = model.auxiliary_reconstructions(output)
        h = output["h_mu"]
        c_summary = model.semantic_summary(output["conditions"], output["gates"])
        h_parts.append(h.cpu().numpy())
        c_parts.append(c_summary.cpu().numpy())
        hc_parts.append(torch.cat((h, c_summary), dim=1).cpu().numpy())
        gates.append(output["gates"].cpu().numpy())
        ablations.append(losses["ablation_delta_mse"].cpu().numpy())
        totals["recon_mse"] += float(losses["recon_mse"]) * len(x_batch)
        totals["loss"] += float(losses["loss"]) * len(x_batch)
        totals["h_only_mse"] += float(
            torch.nn.functional.mse_loss(aux["h_only"], x_batch, reduction="none").mean(dim=1).mean()
        ) * len(x_batch)
        totals["c_only_mse"] += float(
            torch.nn.functional.mse_loss(aux["c_only"], x_batch, reduction="none").mean(dim=1).mean()
        ) * len(x_batch)
        count += len(x_batch)
    return {
        "h": np.vstack(h_parts).astype(np.float32),
        "c": np.vstack(c_parts).astype(np.float32),
        "hc": np.vstack(hc_parts).astype(np.float32),
        "gates": np.vstack(gates).astype(np.float32),
        "ablation_delta_mse": np.vstack(ablations).astype(np.float32),
        "loss_summary": {key: value / max(count, 1) for key, value in totals.items()},
    }
