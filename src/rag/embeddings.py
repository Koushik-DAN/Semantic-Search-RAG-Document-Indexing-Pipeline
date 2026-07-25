"""Sentence-Transformers embedding generation, normalized for cosine similarity."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import DEFAULT_EMBEDDING_MODEL


def _auto_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or _auto_device()
        self._model = SentenceTransformer(model_name, device=self.device)

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Returns an (N, dim) float32 array of L2-normalized embeddings."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Returns a (dim,) float32 normalized embedding for a single query string."""
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
