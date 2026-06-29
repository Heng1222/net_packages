from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch


NO_BACKEND_MESSAGE = (
    "No text embedding backend available. Please provide precomputed condition embeddings "
    "or install sentence-transformers."
)


class TextEmbedder(Protocol):
    model_name: str
    model_revision: str | None

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        model_revision: str | None,
        device: torch.device,
        normalize: bool = True,
        batch_size: int = 32,
        max_length: int | None = None,
        overflow_strategy: str = "error",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(NO_BACKEND_MESSAGE) from exc
        self.model_name = model_name
        self.model_revision = model_revision
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, revision=model_revision, device=str(device))
        native_max_length = int(self.model.max_seq_length)
        if max_length is not None and int(max_length) > native_max_length:
            raise ValueError(
                f"Configured max_length={max_length} exceeds model-native max_seq_length={native_max_length}."
            )
        if overflow_strategy not in {"error", "truncate"}:
            raise ValueError("overflow_strategy must be 'error' or 'truncate'.")
        self.max_length = int(max_length or native_max_length)
        self.model.max_seq_length = self.max_length
        self.overflow_strategy = overflow_strategy
        if hasattr(self.model, "get_embedding_dimension"):
            self.output_dim = int(self.model.get_embedding_dimension())
        else:  # pragma: no cover - compatibility with older sentence-transformers
            self.output_dim = int(self.model.get_sentence_embedding_dimension())
        self.max_observed_tokens = 0

    def _validate_lengths(self, texts: Sequence[str]) -> None:
        lengths: list[int] = []
        validation_batch_size = 128
        for start in range(0, len(texts), validation_batch_size):
            tokenized = self.model.tokenizer(
                list(texts[start : start + validation_batch_size]),
                add_special_tokens=True,
                truncation=False,
                padding=False,
            )
            lengths.extend(len(ids) for ids in tokenized["input_ids"])
        self.max_observed_tokens = max(lengths, default=0)
        overflow = [(index, length) for index, length in enumerate(lengths) if length > self.max_length]
        if overflow and self.overflow_strategy == "error":
            preview = ", ".join(f"row {index}: {length}" for index, length in overflow[:10])
            raise ValueError(
                f"{len(overflow)} text(s) exceed max_length={self.max_length} tokens ({preview}). "
                "Choose a longer-context model or explicitly set overflow_strategy='truncate'."
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._validate_lengths(texts)
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


def text_cache_key(texts: Sequence[str], config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    for key in ("backend", "model_name", "model_revision", "max_length", "overflow_strategy", "normalize"):
        digest.update(f"{key}={config.get(key)}".encode())
    return digest.hexdigest()[:24]


def load_or_embed_payloads(
    texts: Sequence[str], config: dict[str, Any], device: torch.device
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = text_cache_key(texts, config)
    cache_path = cache_dir / f"payload_{key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            features = archive["x"].astype(np.float32)
            max_observed = int(archive["max_observed_tokens"]) if "max_observed_tokens" in archive else None
        return features, {
            "cache_hit": True,
            "cache_path": str(cache_path),
            "key": key,
            "model_name": config.get("model_name"),
            "model_revision": config.get("model_revision"),
            "max_sequence_length": config.get("max_length"),
            "max_observed_tokens": max_observed,
            "output_dim": int(features.shape[1]),
        }
    if config.get("backend") != "sentence_transformers":
        raise ValueError(f"Unsupported payload embedder backend: {config.get('backend')}")
    embedder = SentenceTransformerEmbedder(
        config["model_name"],
        config.get("model_revision"),
        device,
        config.get("normalize", True),
        config.get("batch_size", 4),
        config.get("max_length"),
        config.get("overflow_strategy", "error"),
    )
    features = embedder.encode(texts)
    np.savez_compressed(
        cache_path,
        x=features,
        max_observed_tokens=np.asarray(embedder.max_observed_tokens, dtype=np.int64),
    )
    return features, {
        "cache_hit": False,
        "cache_path": str(cache_path),
        "key": key,
        "model_name": embedder.model_name,
        "model_revision": embedder.model_revision,
        "max_sequence_length": embedder.max_length,
        "max_observed_tokens": embedder.max_observed_tokens,
        "output_dim": features.shape[1],
    }
