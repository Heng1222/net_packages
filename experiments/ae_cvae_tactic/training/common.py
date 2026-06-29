from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class TrainingResult:
    history: list[dict[str, float]]
    best_epoch: int
    best_val_loss: float


def make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(array, dtype=np.float32)) for array in arrays]
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    drop_last = shuffle and len(dataset) > 1 and len(dataset) % batch_size == 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
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


def load_best_state(model: nn.Module, path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint
