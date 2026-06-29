"""Reproducible AE/CVAE tactic latent-space experiments."""

import os

# joblib's Windows physical-core probe is unreliable in some restricted environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

__version__ = "0.1.0"
