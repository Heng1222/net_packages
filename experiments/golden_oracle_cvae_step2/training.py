from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model import GoldenConditionalVAE, PayloadClassifier


@dataclass(slots=True)
class TrainResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float


def _loader(x: np.ndarray, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x[indices], dtype=np.float32)),
        torch.from_numpy(np.asarray(indices, dtype=np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def _cvae_losses(
    model: GoldenConditionalVAE,
    output: dict[str, torch.Tensor],
    x: torch.Tensor,
    gold_gates: torch.Tensor,
    conditions: torch.Tensor,
    loss_config: dict,
    mode: Literal["oracle", "predicted"],
    pos_weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    reconstruction = F.mse_loss(output["x_recon"], x)
    kl = -0.5 * (
        1.0 + output["h_logvar"] - output["h_mu"].pow(2) - output["h_logvar"].exp()
    ).sum(dim=1).mean()
    gate_supervision = x.new_tensor(0.0)
    if mode == "predicted":
        gate_supervision = F.binary_cross_entropy_with_logits(
            output["gate_logits"], gold_gates, pos_weight=pos_weight
        )
    zero_gates = torch.zeros_like(gold_gates)
    h_only = model.decode(output["h"], zero_gates, conditions)
    h_only_mse = F.mse_loss(h_only, x)
    gain = h_only_mse - reconstruction
    condition_use = F.relu(float(loss_config["condition_use_margin"]) - gain)
    total = (
        float(loss_config["reconstruction"]) * reconstruction
        + float(loss_config["kl"]) * kl
        + float(loss_config["gate_supervision"]) * gate_supervision
        + float(loss_config["condition_use"]) * condition_use
    )
    return {
        "loss": total,
        "recon_mse": reconstruction,
        "kl_loss": kl,
        "gate_supervision_loss": gate_supervision,
        "condition_use_loss": condition_use,
        "h_only_mse": h_only_mse,
        "condition_gain": gain,
    }


def train_cvae(
    mode: Literal["oracle", "predicted"],
    model: GoldenConditionalVAE,
    x: np.ndarray,
    gate_targets: np.ndarray,
    conditions: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    training_config: dict,
    loss_config: dict,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    logger,
) -> TrainResult:
    model.to(device)
    condition_tensor = torch.from_numpy(np.asarray(conditions, dtype=np.float32)).to(device)
    train_targets = gate_targets[train_indices]
    positives = train_targets.sum(axis=0)
    negatives = len(train_targets) - positives
    pos_weight = torch.from_numpy(
        np.clip(negatives / np.maximum(positives, 1.0), 1.0, 10.0).astype(np.float32)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    train_loader = _loader(x, train_indices, int(training_config["batch_size"]), True, seed)
    val_loader = _loader(x, val_indices, int(training_config["batch_size"]), False, seed)
    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    tracked = ("loss", "recon_mse", "kl_loss", "gate_supervision_loss", "condition_use_loss", "h_only_mse", "condition_gain")
    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        row: dict[str, float] = {"epoch": float(epoch)}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            totals = {key: 0.0 for key in tracked}
            count = 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for xb, index_batch in loader:
                    xb = xb.to(device)
                    gates = torch.from_numpy(gate_targets[index_batch.numpy()]).to(device)
                    output = model(
                        xb,
                        condition_tensor,
                        gates_override=gates if mode == "oracle" else None,
                        sample=phase == "train",
                    )
                    losses = _cvae_losses(
                        model, output, xb, gates, condition_tensor, loss_config, mode, pos_weight
                    )
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        losses["loss"].backward()
                        optimizer.step()
                    for key in tracked:
                        totals[key] += float(losses[key].detach()) * len(xb)
                    count += len(xb)
            row.update({f"{phase}_{key}": value / max(count, 1) for key, value in totals.items()})
        history.append(row)
        improved = row["val_loss"] < best - 1e-8
        if improved:
            best = row["val_loss"]
            best_epoch = epoch
            stale = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": best}, checkpoint_path)
        else:
            stale += 1
        logger.info(
            "%s epoch=%d train_loss=%.6f val_loss=%.6f val_recon=%.6f val_h_only=%.6f val_gain=%.6f stale=%d",
            mode,
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_recon_mse"],
            row["val_h_only_mse"],
            row["val_condition_gain"],
            stale,
        )
        if stale >= int(training_config["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return TrainResult(history, best_epoch, best)


def train_classifier(
    model: PayloadClassifier,
    x: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    training_config: dict,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    logger,
) -> TrainResult:
    model.to(device)
    counts = np.bincount(targets[train_indices], minlength=int(targets.max()) + 1)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    train_loader = _loader(x, train_indices, int(training_config["batch_size"]), True, seed)
    val_loader = _loader(x, val_indices, int(training_config["batch_size"]), False, seed)
    best, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training_config["max_epochs"]) + 1):
        row: dict[str, float] = {"epoch": float(epoch)}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            total_loss, correct, count = 0.0, 0, 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for xb, index_batch in loader:
                    xb = xb.to(device)
                    yb = torch.from_numpy(targets[index_batch.numpy()]).to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                    total_loss += float(loss.detach()) * len(xb)
                    correct += int((logits.argmax(dim=1) == yb).sum())
                    count += len(xb)
            row[f"{phase}_loss"] = total_loss / max(count, 1)
            row[f"{phase}_accuracy"] = correct / max(count, 1)
        history.append(row)
        if row["val_loss"] < best - 1e-8:
            best, best_epoch, stale = row["val_loss"], epoch, 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_loss": best}, checkpoint_path)
        else:
            stale += 1
        logger.info(
            "classifier epoch=%d train_loss=%.6f val_loss=%.6f val_acc=%.4f stale=%d",
            epoch, row["train_loss"], row["val_loss"], row["val_accuracy"], stale,
        )
        if stale >= int(training_config["early_stopping_patience"]):
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return TrainResult(history, best_epoch, best)
