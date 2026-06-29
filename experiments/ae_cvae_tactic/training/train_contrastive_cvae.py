from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models.contrastive_cvae import ContrastiveConditionalVAE
from .common import TrainingResult, load_best_state, make_loader, save_checkpoint


TRACKED_LOSSES = (
    "loss",
    "negative_elbo",
    "recon_nll",
    "recon_mse",
    "kl_loss",
    "elbo",
    "contrastive_loss",
    "weighted_contrastive_loss",
    "contrastive_accuracy",
)


def _mean_totals(totals: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_contrastive_cvae(
    model: ContrastiveConditionalVAE,
    x_train: np.ndarray,
    c_train: np.ndarray,
    target_train: np.ndarray,
    x_val: np.ndarray,
    c_val: np.ndarray,
    target_val: np.ndarray,
    candidate_conditions: np.ndarray,
    training: dict[str, Any],
    model_config: dict[str, Any],
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int,
) -> TrainingResult:
    logger = logging.getLogger("ae_cvae_tactic.contrastive")
    model.to(device)
    candidates = torch.from_numpy(
        np.asarray(candidate_conditions, dtype=np.float32)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_loader = make_loader(
        x_train,
        c_train,
        target_train,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(training.get("num_workers", 0)),
    )
    val_loader = make_loader(
        x_val,
        c_val,
        target_val,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(training.get("num_workers", 0)),
    )

    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_totals = {key: 0.0 for key in TRACKED_LOSSES}
        train_count = 0
        for x_batch, c_batch, target_batch in train_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            target_batch = target_batch.to(device).long()
            optimizer.zero_grad(set_to_none=True)
            output = model(x_batch, c_batch, candidates, sample=True)
            losses = model.loss(output, x_batch, target_batch)
            losses["loss"].backward()
            optimizer.step()
            for key in TRACKED_LOSSES:
                train_totals[key] += float(losses[key].detach()) * len(x_batch)
            train_count += len(x_batch)

        model.eval()
        val_totals = {key: 0.0 for key in TRACKED_LOSSES}
        val_count = 0
        with torch.inference_mode():
            for x_batch, c_batch, target_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                target_batch = target_batch.to(device).long()
                output = model(x_batch, c_batch, candidates, sample=False)
                losses = model.loss(output, x_batch, target_batch)
                for key in TRACKED_LOSSES:
                    val_totals[key] += float(losses[key]) * len(x_batch)
                val_count += len(x_batch)

        train_values = _mean_totals(train_totals, train_count)
        val_values = _mean_totals(val_totals, val_count) if val_count else train_values
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": value for key, value in train_values.items()})
        row.update({f"val_{key}": value for key, value in val_values.items()})
        history.append(row)
        logger.info(
            "Contrastive CVAE epoch %d | train_total=%.6f | val_total=%.6f | "
            "val_nelbo=%.6f | val_contrastive=%.6f | val_retrieval_acc=%.6f",
            epoch,
            train_values["loss"],
            val_values["loss"],
            val_values["negative_elbo"],
            val_values["contrastive_loss"],
            val_values["contrastive_accuracy"],
        )
        if val_values["loss"] < best - 1e-10:
            best, best_epoch, stale = val_values["loss"], epoch, 0
            save_checkpoint(checkpoint_path, model, optimizer, model_config, epoch, best)
        else:
            stale += 1
            if stale >= int(training["early_stopping_patience"]):
                logger.info("Contrastive CVAE early stopping at epoch %d", epoch)
                break

    load_best_state(model, checkpoint_path, device)
    return TrainingResult(history, best_epoch, best)


@torch.inference_mode()
def extract_contrastive_cvae(
    model: ContrastiveConditionalVAE,
    x: np.ndarray,
    condition: np.ndarray,
    target_indices: np.ndarray,
    candidate_conditions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    candidates = torch.from_numpy(
        np.asarray(candidate_conditions, dtype=np.float32)
    ).to(device)
    loader = make_loader(
        x,
        condition,
        target_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
    )
    latents: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    per_sample: dict[str, list[np.ndarray]] = {
        "total_loss": [],
        "negative_elbo": [],
        "recon_nll": [],
        "recon_mse": [],
        "kl_loss": [],
        "elbo": [],
        "contrastive_loss": [],
    }
    for x_batch, c_batch, target_batch in loader:
        x_batch = x_batch.to(device)
        c_batch = c_batch.to(device)
        target_batch = target_batch.to(device).long()
        output = model(
            x_batch,
            c_batch,
            candidates,
            sample=model.latent_representation == "z",
        )
        losses = model.loss(output, x_batch, target_batch)
        latents.append(model.representation(output).cpu().numpy())
        projections.append(output["payload_projection"].cpu().numpy())
        logits.append(output["contrastive_logits"].cpu().numpy())
        per_sample["total_loss"].append(losses["total_per_sample"].cpu().numpy())
        per_sample["negative_elbo"].append(losses["negative_elbo_per_sample"].cpu().numpy())
        per_sample["recon_nll"].append(losses["recon_nll_per_sample"].cpu().numpy())
        per_sample["recon_mse"].append(losses["recon_mse_per_sample"].cpu().numpy())
        per_sample["kl_loss"].append(losses["kl_per_sample"].cpu().numpy())
        per_sample["elbo"].append(losses["elbo_per_sample"].cpu().numpy())
        per_sample["contrastive_loss"].append(losses["contrastive_per_sample"].cpu().numpy())

    values = {key: np.concatenate(parts) for key, parts in per_sample.items()}
    contrastive_loss = float(values["contrastive_loss"].mean())
    return (
        np.vstack(latents).astype(np.float32),
        np.vstack(projections).astype(np.float32),
        np.vstack(logits).astype(np.float32),
        {
            "total_loss": float(values["total_loss"].mean()),
            "negative_elbo": float(values["negative_elbo"].mean()),
            "recon_nll": float(values["recon_nll"].mean()),
            "recon_mse": float(values["recon_mse"].mean()),
            "kl_loss": float(values["kl_loss"].mean()),
            "elbo": float(values["elbo"].mean()),
            "contrastive_loss": contrastive_loss,
            "weighted_contrastive_loss": model.contrastive_weight * contrastive_loss,
            "contrastive_accuracy": float(
                (np.vstack(logits).argmax(axis=1) == np.asarray(target_indices)).mean()
            ),
        },
    )
