from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.cvae import ConditionalVAE
from .common import TrainingResult, load_best_state, make_loader, save_checkpoint


def train_cvae(
    model: ConditionalVAE,
    x_train: np.ndarray,
    c_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
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
        x_train, c_train, batch_size=int(training["batch_size"]), shuffle=True, seed=seed,
        num_workers=int(training.get("num_workers", 0))
    )
    val_loader = make_loader(
        x_val, c_val, batch_size=int(training["batch_size"]), shuffle=False, seed=seed,
        num_workers=int(training.get("num_workers", 0))
    )
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "recon_nll": 0.0, "recon_mse": 0.0, "kl_loss": 0.0, "elbo": 0.0}
        count = 0
        for x_batch, c_batch in train_loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(model(x_batch, c_batch, sample=True), x_batch)
            losses["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * len(x_batch)
            count += len(x_batch)
        model.eval()
        val_totals = {"loss": 0.0, "recon_nll": 0.0, "recon_mse": 0.0, "kl_loss": 0.0, "elbo": 0.0}
        val_count = 0
        with torch.inference_mode():
            for x_batch, c_batch in val_loader:
                x_batch, c_batch = x_batch.to(device), c_batch.to(device)
                losses = model.loss(model(x_batch, c_batch, sample=False), x_batch)
                for key in val_totals:
                    val_totals[key] += float(losses[key]) * len(x_batch)
                val_count += len(x_batch)
        train_values = {key: value / max(count, 1) for key, value in totals.items()}
        val_values = (
            {key: value / val_count for key, value in val_totals.items()} if val_count else train_values
        )
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_values.items()})
        row.update({f"val_{key}": value for key, value in val_values.items()})
        history.append(row)
        logger.info(
            "CVAE epoch %d | train_nelbo=%.6f | val_nelbo=%.6f | val_recon_nll=%.6f | val_kl=%.6f",
            epoch, train_values["loss"], val_values["loss"], val_values["recon_nll"], val_values["kl_loss"]
        )
        if val_values["loss"] < best - 1e-10:
            best, best_epoch, stale = val_values["loss"], epoch, 0
            save_checkpoint(checkpoint_path, model, optimizer, model_config, epoch, best)
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                logger.info("CVAE early stopping at epoch %d", epoch)
                break
    load_best_state(model, checkpoint_path, device)
    return TrainingResult(history, best_epoch, best)


@torch.inference_mode()
def extract_cvae_latent(
    model: ConditionalVAE,
    x: np.ndarray,
    condition: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    loader = make_loader(x, condition, batch_size=batch_size, shuffle=False, seed=0)
    latents: list[np.ndarray] = []
    recon_nll: list[np.ndarray] = []
    recon_mse: list[np.ndarray] = []
    kl: list[np.ndarray] = []
    elbo: list[np.ndarray] = []
    for x_batch, c_batch in loader:
        x_batch, c_batch = x_batch.to(device), c_batch.to(device)
        output = model(x_batch, c_batch, sample=model.latent_representation == "z")
        losses = model.loss(output, x_batch)
        latents.append(model.representation(output).cpu().numpy())
        recon_nll.append(losses["recon_nll_per_sample"].cpu().numpy())
        recon_mse.append(losses["recon_mse_per_sample"].cpu().numpy())
        kl.append(losses["kl_per_sample"].cpu().numpy())
        elbo.append(losses["elbo_per_sample"].cpu().numpy())
    recon_nll_values = np.concatenate(recon_nll)
    recon_mse_values = np.concatenate(recon_mse)
    kl_values = np.concatenate(kl)
    elbo_values = np.concatenate(elbo)
    negative_elbo = float(-elbo_values.mean())
    return np.vstack(latents).astype(np.float32), {
        "recon_loss": float(recon_nll_values.mean()),
        "recon_nll": float(recon_nll_values.mean()),
        "recon_mse": float(recon_mse_values.mean()),
        "kl_loss": float(kl_values.mean()),
        "elbo": float(elbo_values.mean()),
        "negative_elbo": negative_elbo,
        "total_loss": negative_elbo,
    }
