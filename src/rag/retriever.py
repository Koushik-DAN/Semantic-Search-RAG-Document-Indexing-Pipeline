"""Composes an Embedder and a FaissVectorStore to retrieve relevant chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass

from rag.config import DEFAULT_TOP_K
from rag.embeddings import Embedder
from rag.vector_store import FaissVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissVectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(query_vector, top_k=top_k)
        return [
            RetrievedChunk(chunk_id=r.chunk_id, text=r.text, source=r.source, score=r.score)
            for r in results
        ]
