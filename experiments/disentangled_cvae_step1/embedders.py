from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

import numpy as np
import torch


class TextEmbedder(Protocol):
    output_dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingTextEmbedder:
    """Deterministic local embedder for tests and smoke runs."""

    def __init__(self, output_dim: int = 768, normalize: bool = True) -> None:
        self.output_dim = int(output_dim)
        self.normalize = bool(normalize)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=self.output_dim).astype(np.float32)
            if self.normalize:
                vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rows.append(vector)
        if not rows:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)


class SentenceTransformerChunkingEmbedder:
    def __init__(
        self,
        model_name: str,
        model_revision: str | None,
        device: torch.device,
        normalize: bool = True,
        batch_size: int = 4,
        max_length: int | None = None,
        overflow_strategy: str = "error",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for ModernBERT embedding.") from exc
        if overflow_strategy not in {"error", "truncate", "chunk_mean"}:
            raise ValueError("overflow_strategy must be 'error', 'truncate', or 'chunk_mean'.")
        self.model = SentenceTransformer(model_name, revision=model_revision, device=str(device))
        native_max_length = int(self.model.max_seq_length)
        if max_length is not None and int(max_length) > native_max_length:
            raise ValueError(
                f"Configured max_length={max_length} exceeds model-native max_seq_length={native_max_length}."
            )
        self.max_length = int(max_length or native_max_length)
        self.model.max_seq_length = self.max_length
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size)
        self.overflow_strategy = overflow_strategy
        if hasattr(self.model, "get_embedding_dimension"):
            self.output_dim = int(self.model.get_embedding_dimension())
        else:
            self.output_dim = int(self.model.get_sentence_embedding_dimension())
        self.max_observed_tokens = 0
        self.overflow_count = 0

    def _token_lengths(self, texts: Sequence[str]) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(texts), 128):
            tokenized = self.model.tokenizer(
                list(texts[start : start + 128]),
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            lengths.extend(len(ids) for ids in tokenized["input_ids"])
        self.max_observed_tokens = max(self.max_observed_tokens, max(lengths, default=0))
        return lengths

    def _encode_plain(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def _encode_chunked_text(self, text: str) -> np.ndarray:
        tokenized = self.model.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        ids = list(tokenized["input_ids"])
        if not ids:
            return self._encode_plain([""])[0]
        chunk_size = max(1, self.max_length - 2)
        chunks = [ids[start : start + chunk_size] for start in range(0, len(ids), chunk_size)]
        chunk_texts = self.model.tokenizer.batch_decode(chunks, skip_special_tokens=True)
        embeddings = self._encode_plain(chunk_texts)
        weights = np.asarray([len(chunk) for chunk in chunks], dtype=np.float32)
        vector = np.average(embeddings, axis=0, weights=weights).astype(np.float32)
        if self.normalize:
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(map(str, texts))
        lengths = self._token_lengths(texts)
        overflow = [index for index, length in enumerate(lengths) if length > self.max_length]
        self.overflow_count += len(overflow)
        if overflow and self.overflow_strategy == "error":
            preview = ", ".join(f"row {index}: {lengths[index]}" for index in overflow[:10])
            raise ValueError(
                f"{len(overflow)} text(s) exceed max_length={self.max_length} tokens ({preview}). "
                "Use overflow_strategy='truncate' or 'chunk_mean'."
            )
        if self.overflow_strategy != "chunk_mean" or not overflow:
            return self._encode_plain(texts)

        result = np.empty((len(texts), self.output_dim), dtype=np.float32)
        overflow_set = set(overflow)
        normal_indices = [index for index in range(len(texts)) if index not in overflow_set]
        normal_embeddings = self._encode_plain([texts[index] for index in normal_indices])
        for row, embedding in zip(normal_indices, normal_embeddings, strict=True):
            result[row] = embedding
        for row in overflow:
            result[row] = self._encode_chunked_text(texts[row])
        return result


def build_text_embedder(config: dict[str, Any], device: torch.device) -> TextEmbedder:
    backend = str(config.get("backend", config.get("embedder_backend", "sentence_transformers")))
    normalize = bool(config.get("normalize", True))
    if backend == "hashing":
        return HashingTextEmbedder(int(config.get("output_dim", 768)), normalize)
    if backend != "sentence_transformers":
        raise ValueError(f"Unsupported text embedder backend: {backend}")

    model_name = config.get("model_name", config.get("embedder_model_name"))
    if not model_name:
        raise ValueError("Sentence transformer embedder requires model_name or embedder_model_name.")
    embedder = SentenceTransformerChunkingEmbedder(
        model_name,
        config.get("model_revision", config.get("embedder_model_revision")),
        device,
        normalize,
        int(config.get("batch_size", 4)),
        config.get("max_length"),
        config.get("overflow_strategy", "error"),
    )
    return embedder
