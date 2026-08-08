from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .data import payload_to_text
from .embedders import build_text_embedder
from .model import GateEncoder, UCDCVAE


def decision_for_score(score: float, benign_threshold: float, block_threshold: float) -> str:
    if not 0 <= benign_threshold < block_threshold <= 1: raise ValueError("Invalid decision thresholds.")
    if score < benign_threshold: return "allow"
    if score < block_threshold: return "review"
    return "block_recommended"


def export_gate_checkpoint(model: UCDCVAE, path: Path, labels: list[str], embedder_config: dict[str, Any],
                           payload_parser: str, evaluation_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "ucd-cvae-gate-only", "model_version": "2.1.0",
                "gate_state": model.gate_encoder.state_dict(), "input_dim": model.input_dim,
                "gate_hidden_dims": model.gate_encoder.hidden_dims,
                "activation": model.gate_encoder.activation_name,
                "dropout": model.gate_encoder.dropout_rate, "labels": labels,
                "embedder": embedder_config, "payload_parser": payload_parser,
                "benign_threshold": float(evaluation_config["benign_threshold"]),
                "block_threshold": float(evaluation_config["block_threshold"]),
                "top_k": int(evaluation_config.get("top_k", 3)),
                "score_semantics": "uncalibrated_evidence"}, path)


class GateInferenceEngine:
    def __init__(self, checkpoint: dict[str, Any], device: torch.device) -> None:
        if checkpoint.get("format") != "ucd-cvae-gate-only": raise ValueError("Not a UCD-CVAE gate-only checkpoint.")
        self.device = device; self.labels = list(map(str, checkpoint["labels"]))
        if len(self.labels) != 15: raise ValueError("Gate checkpoint must contain 15 labels.")
        self.model = GateEncoder(int(checkpoint["input_dim"]), list(checkpoint["gate_hidden_dims"]), 15,
                                 str(checkpoint.get("activation", "gelu")), float(checkpoint.get("dropout", 0.0)))
        self.model.load_state_dict(checkpoint["gate_state"]); self.model.to(device).eval()
        self.embedder_config = dict(checkpoint["embedder"]); self.payload_parser = str(checkpoint.get("payload_parser", "auto"))
        self.benign_threshold = float(checkpoint["benign_threshold"]); self.block_threshold = float(checkpoint["block_threshold"])
        self.top_k = int(checkpoint.get("top_k", 3)); self.model_version = str(checkpoint["model_version"])
        self.embedder = build_text_embedder(self.embedder_config, device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "GateInferenceEngine":
        target = torch.device(device); checkpoint = torch.load(Path(path), map_location=target, weights_only=False)
        return cls(checkpoint, target)

    @torch.inference_mode()
    def predict_embeddings(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        values = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        started = time.perf_counter(); gates = torch.sigmoid(self.model(values))
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return gates.cpu().numpy().astype(np.float32), elapsed

    def predict_texts(self, texts: Sequence[str], sample_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        clean = [payload_to_text(text, self.payload_parser) for text in texts]
        started = time.perf_counter(); embeddings = self.embedder.encode(clean); embedding_seconds = time.perf_counter() - started
        gates, gate_seconds = self.predict_embeddings(embeddings); total_seconds = embedding_seconds + gate_seconds
        identifiers = list(sample_ids) if sample_ids is not None else [str(i) for i in range(len(clean))]
        results = []
        for row, sample_id in zip(gates, identifiers, strict=True):
            tactic_values = {label: float(row[index + 1]) for index, label in enumerate(self.labels[1:])}
            top = sorted(tactic_values.items(), key=lambda item: item[1], reverse=True)[:self.top_k]
            common = float(row[0])
            results.append({"sample_id": str(sample_id), "common_evidence": common,
                "common_evidence_points": 100.0 * common, "tactic_evidence": tactic_values,
                "top_tactics": [{"tactic": label, "evidence": value,
                                  "evidence_points": 100.0 * value} for label, value in top],
                "decision": decision_for_score(common, self.benign_threshold, self.block_threshold),
                "benign_threshold": self.benign_threshold, "block_threshold": self.block_threshold,
                "model_version": self.model_version, "score_semantics": "uncalibrated_evidence",
                "latency_ms": {"embedding_batch": embedding_seconds * 1000,
                               "gate_batch": gate_seconds * 1000, "end_to_end_batch": total_seconds * 1000}})
        return results
