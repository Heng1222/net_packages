from __future__ import annotations

import numpy as np
import torch

from experiments.ucd_cvae_v2_1.geometry import symmetric_orthogonalize_numpy, uncentered_common_component


def small_basis(dimension: int = 32, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed); raw = rng.normal(size=(14, dimension)).astype(np.float32)
    common, residual, _ = uncentered_common_component(raw)
    tactics = symmetric_orthogonalize_numpy(residual, forbidden=common)
    return torch.from_numpy(common), torch.from_numpy(tactics)
