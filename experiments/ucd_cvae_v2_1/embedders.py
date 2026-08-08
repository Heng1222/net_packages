from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

import numpy as np
import torch


def normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    return (matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)).astype(np.float32)


class TextEmbedder(Protocol):
    output_dim: int
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingTextEmbedder:
    def __init__(self, output_dim: int = 768, normalize: bool = True) -> None:
        self.output_dim = int(output_dim); self.normalize = bool(normalize)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()[:8], "little")
            vector = np.random.default_rng(seed).normal(size=self.output_dim).astype(np.float32)
            if self.normalize: vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        return np.vstack(rows).astype(np.float32) if rows else np.empty((0, self.output_dim), np.float32)


class SentenceTransformerChunkingEmbedder:
    def __init__(self, model_name: str, model_revision: str | None, device: torch.device,
                 normalize: bool = True, batch_size: int = 4, max_length: int | None = None,
                 overflow_strategy: str = "error") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for ModernBERT embedding.") from exc
        if overflow_strategy not in {"error", "truncate", "chunk_mean"}:
            raise ValueError("overflow_strategy must be error, truncate, or chunk_mean.")
        self.model = SentenceTransformer(model_name, revision=model_revision, device=str(device))
        native = int(self.model.max_seq_length)
        if max_length is not None and int(max_length) > native:
            raise ValueError(f"max_length={max_length} exceeds model maximum {native}.")
        self.max_length = int(max_length or native); self.model.max_seq_length = self.max_length
        self.normalize = bool(normalize); self.batch_size = int(batch_size); self.overflow_strategy = overflow_strategy
        getter = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)
        self.output_dim = int(getter())

    def _plain(self, texts: Sequence[str]) -> np.ndarray:
        if not texts: return np.empty((0, self.output_dim), np.float32)
        return np.asarray(self.model.encode(list(texts), batch_size=self.batch_size,
                          normalize_embeddings=self.normalize, convert_to_numpy=True,
                          show_progress_bar=len(texts) > 100), dtype=np.float32)

    def _lengths(self, texts: Sequence[str]) -> list[int]:
        result = []
        for start in range(0, len(texts), 128):
            encoded = self.model.tokenizer(list(texts[start:start + 128]), add_special_tokens=True,
                                           truncation=False, padding=False)
            result.extend(len(ids) for ids in encoded["input_ids"])
        return result

    def _chunked(self, text: str) -> np.ndarray:
        ids = self.model.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids: return self._plain([""])[0]
        size = max(1, self.max_length - 2); chunks = [ids[i:i + size] for i in range(0, len(ids), size)]
        vectors = self._plain(self.model.tokenizer.batch_decode(chunks, skip_special_tokens=True))
        vector = np.average(vectors, axis=0, weights=[len(v) for v in chunks]).astype(np.float32)
        if self.normalize: vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(map(str, texts)); lengths = self._lengths(texts)
        overflow = [i for i, length in enumerate(lengths) if length > self.max_length]
        if overflow and self.overflow_strategy == "error":
            raise ValueError(f"{len(overflow)} payloads exceed max_length={self.max_length}.")
        if self.overflow_strategy != "chunk_mean" or not overflow: return self._plain(texts)
        result = np.empty((len(texts), self.output_dim), np.float32); overflow_set = set(overflow)
        normal = [i for i in range(len(texts)) if i not in overflow_set]
        for index, vector in zip(normal, self._plain([texts[i] for i in normal]), strict=True): result[index] = vector
        for index in overflow: result[index] = self._chunked(texts[index])
        return result


def build_text_embedder(config: dict[str, Any], device: torch.device) -> TextEmbedder:
    if config.get("backend", "sentence_transformers") == "hashing":
        return HashingTextEmbedder(int(config.get("output_dim", 768)), bool(config.get("normalize", True)))
    if config.get("backend", "sentence_transformers") != "sentence_transformers":
        raise ValueError(f"Unsupported embedder backend: {config.get('backend')}")
    return SentenceTransformerChunkingEmbedder(
        str(config["model_name"]), config.get("model_revision"), device,
        bool(config.get("normalize", True)), int(config.get("batch_size", 4)),
        config.get("max_length"), str(config.get("overflow_strategy", "error")),
    )
