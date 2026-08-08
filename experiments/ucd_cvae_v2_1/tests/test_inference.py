from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.ucd_cvae_v2_1.inference import (
    GateInferenceEngine, decision_for_score, export_gate_checkpoint,
)
from experiments.ucd_cvae_v2_1.model import UCDCVAE
from experiments.ucd_cvae_v2_1.tests.helpers import small_basis


class InferenceTests(unittest.TestCase):
    def test_threshold_boundaries(self) -> None:
        self.assertEqual(decision_for_score(0.0999, 0.1, 0.5), "allow")
        self.assertEqual(decision_for_score(0.1, 0.1, 0.5), "review")
        self.assertEqual(decision_for_score(0.5, 0.1, 0.5), "block_recommended")

    def test_gate_only_export_and_python_api(self) -> None:
        common, tactics = small_basis(); model = UCDCVAE(common, tactics, input_dim=32,
            gate_hidden_dims=[20], residual_hidden_dims=[20], residual_up_hidden_dims=[20],
            decoder_hidden_dims=[20], concept_projector_hidden_dim=16, dropout=0)
        labels = ["Common", *[f"Tactic {i}" for i in range(14)]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "gate.pt"
            export_gate_checkpoint(model, path, labels, {"backend": "hashing", "output_dim": 32, "normalize": True},
                                   "auto", {"benign_threshold": 0.1, "block_threshold": 0.5, "top_k": 3})
            engine = GateInferenceEngine.from_checkpoint(path)
            result = engine.predict_texts(["GET / HTTP/1.1"], ["sample"])[0]
            self.assertEqual(len(result["tactic_evidence"]), 14); self.assertEqual(len(result["top_tactics"]), 3)
            self.assertEqual(result["score_semantics"], "uncalibrated_evidence")


if __name__ == "__main__": unittest.main()
