from __future__ import annotations

import unittest
from pathlib import Path

from experiments.center_augmented_cvae_step1.utils import load_config


class IntegrationTests(unittest.TestCase):
    def test_default_config_resolves_independent_paths(self) -> None:
        root = Path(__file__).resolve().parents[3]
        config = load_config(root / "experiments" / "center_augmented_cvae_step1" / "configs" / "default.yaml", root)
        self.assertIn("center_augmented_cvae_step1", config["output"]["base_dir"])
        self.assertIn("center_augmented_cvae_step1", config["conditions"]["path"])
        self.assertTrue(config["data"]["embedder"]["normalize"])


if __name__ == "__main__": unittest.main()
