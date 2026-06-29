from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.ae import AutoEncoder
from .common import TrainingResult, load_best_state, make_loader, save_checkpoint


def train_ae(
    model: AutoEncoder,
    x_train: np.ndarray,
    x_val: np.ndarray,
    training: dict[str, Any],
    model_config: dict[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int,
) -> TrainingResult:
    logger = logging.getLogger("ae_cvae_tactic")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    train_loader = make_loader(
        x_train, batch_size=int(training["batch_size"]), shuffle=True, seed=seed,
        num_workers=int(training.get("num_workers", 0))
    )
    val_loader = make_loader(
        x_val, batch_size=int(training["batch_size"]), shuffle=False, seed=seed,
        num_workers=int(training.get("num_workers", 0))
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_sum = 0.0
        train_count = 0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model(x_batch), x_batch)["loss"]
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach()) * len(x_batch)
            train_count += len(x_batch)
        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                loss = model.loss(model(x_batch), x_batch)["loss"]
                val_sum += float(loss) * len(x_batch)
                val_count += len(x_batch)
        train_loss = train_sum / max(train_count, 1)
        val_loss = val_sum / max(val_count, 1) if val_count else train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        logger.info("AE epoch %d | train=%.6f | val=%.6f", epoch, train_loss, val_loss)
        if val_loss < best - 1e-10:
            best, best_epoch, stale = val_loss, epoch, 0
            save_checkpoint(checkpoint_path, model, optimizer, model_config, epoch, val_loss)
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                logger.info("AE early stopping at epoch %d", epoch)
                break
    load_best_state(model, checkpoint_path, device)
    return TrainingResult(history, best_epoch, best)


@torch.inference_mode()
def extract_ae_latent(
    model: AutoEncoder, x: np.ndarray, device: torch.device, batch_size: int
) -> tuple[np.ndarray, float]:
    model.eval()
    loader = make_loader(x, batch_size=batch_size, shuffle=False, seed=0)
    latents: list[np.ndarray] = []
    losses: list[np.ndarray] = []
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        output = model(x_batch)
        latents.append(output["latent"].cpu().numpy())
        losses.append(model.loss(output, x_batch)["recon_per_sample"].cpu().numpy())
    return np.vstack(latents).astype(np.float32), float(np.concatenate(losses).mean())
