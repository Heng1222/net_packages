from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

import numpy as np
import torch


class TextEmbedder(Protocol):
    output_dim: int
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


class HashingTextEmbedder:
    """Deterministic offline embedder used only by tests and smoke configurations."""

    def __init__(self, output_dim: int = 768, normalize: bool = True) -> None:
        self.output_dim = int(output_dim)
        self.normalize = bool(normalize)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            rows.append(rng.normal(size=self.output_dim).astype(np.float32))
        matrix = np.vstack(rows) if rows else np.empty((0, self.output_dim), dtype=np.float32)
        return normalize_rows(matrix) if self.normalize and len(matrix) else matrix


class SentenceTransformerChunkingEmbedder:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for ModernBERT embedding.") from exc
        self.model = SentenceTransformer(
            config["model_name"], revision=config.get("model_revision"), device=str(device)
        )
        native_max = int(self.model.max_seq_length)
        requested = int(config.get("max_length", native_max))
        if requested > native_max:
            raise ValueError(f"max_length={requested} exceeds model maximum {native_max}.")
        self.max_length = requested
        self.model.max_seq_length = requested
        self.batch_size = int(config.get("batch_size", 4))
        self.normalize = bool(config.get("normalize", True))
        self.overflow_strategy = str(config.get("overflow_strategy", "error"))
        if self.overflow_strategy not in {"error", "truncate", "chunk_mean"}:
            raise ValueError("overflow_strategy must be error, truncate, or chunk_mean.")
        self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def _plain(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.asarray(self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100, convert_to_numpy=True,
        ), dtype=np.float32)

    def _token_ids(self, text: str) -> list[int]:
        return list(self.model.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(map(str, texts))
        if self.overflow_strategy == "truncate":
            return self._plain(values)
        token_ids = [self._token_ids(text) for text in values]
        overflow = [i for i, ids in enumerate(token_ids) if len(ids) + 2 > self.max_length]
        if overflow and self.overflow_strategy == "error":
            raise ValueError(f"{len(overflow)} text(s) exceed max_length={self.max_length}.")
        if not overflow:
            return self._plain(values)
        result = np.empty((len(values), self.output_dim), dtype=np.float32)
        overflow_set = set(overflow)
        normal = [i for i in range(len(values)) if i not in overflow_set]
        for index, vector in zip(normal, self._plain([values[i] for i in normal]), strict=True):
            result[index] = vector
        chunk_size = max(1, self.max_length - 2)
        for index in overflow:
            chunks = [token_ids[index][start:start + chunk_size] for start in range(0, len(token_ids[index]), chunk_size)]
            chunk_texts = self.model.tokenizer.batch_decode(chunks, skip_special_tokens=True)
            embedded = self._plain(chunk_texts)
            vector = np.average(embedded, axis=0, weights=[len(chunk) for chunk in chunks]).astype(np.float32)
            result[index] = normalize_rows(vector[None, :])[0] if self.normalize else vector
        return result


def build_text_embedder(config: dict[str, Any], device: torch.device) -> TextEmbedder:
    backend = str(config.get("backend", "sentence_transformers"))
    if backend == "hashing":
        return HashingTextEmbedder(int(config.get("output_dim", 768)), bool(config.get("normalize", True)))
    if backend != "sentence_transformers":
        raise ValueError(f"Unsupported embedder backend: {backend}")
    if not config.get("model_name"):
        raise ValueError("embedder.model_name is required.")
    return SentenceTransformerChunkingEmbedder(config, device)
