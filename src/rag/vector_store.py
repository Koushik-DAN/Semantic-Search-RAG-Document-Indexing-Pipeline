"""FAISS-backed vector store using cosine similarity via normalized inner product."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    source: str
    score: float


class FaissVectorStore:
    """Wraps a faiss.IndexFlatIP index. Vectors must be L2-normalized (both at
    embed time and again defensively here) so inner product == cosine similarity."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self._metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas must be the same length")
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self._metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if len(self._metadata) == 0:
            return []
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(top_k, len(self._metadata))
        scores, ids = self.index.search(query, k)
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    source=meta["source"],
                    score=float(score),
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._metadata)

    def save(self, index_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "faiss.index"))

        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for meta in self._metadata:
                f.write(json.dumps(meta) + "\n")

        meta_summary = {
            "embedding_model": embedding_model,
            "dim": self.dim,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(self._metadata),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (index_dir / "meta.json").write_text(json.dumps(meta_summary, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, expected_embedding_model: str | None = None) -> "FaissVectorStore":
        index_dir = Path(index_dir)
        meta_summary = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))

        if (
            expected_embedding_model is not None
            and meta_summary.get("embedding_model") != expected_embedding_model
        ):
            print(
                f"WARNING: index was built with embedding model "
                f"'{meta_summary.get('embedding_model')}', but current config uses "
                f"'{expected_embedding_model}'. Query results will be unreliable — "
                "re-run `index` to rebuild."
            )

        store = cls(dim=meta_summary["dim"])
        store.index = faiss.read_index(str(index_dir / "faiss.index"))

        metadata: list[dict] = []
        with (index_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
        store._metadata = metadata

        return store
