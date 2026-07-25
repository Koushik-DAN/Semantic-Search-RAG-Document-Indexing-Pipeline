import numpy as np

from rag.retriever import Retriever
from rag.vector_store import FaissVectorStore


class StubEmbedder:
    """Deterministic stand-in for Embedder: maps known query strings to fixed vectors."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        vector[hash(text) % self.dim] = 1.0
        return vector


def test_retrieve_returns_top_k_sorted_by_score():
    dim = 4
    store = FaissVectorStore(dim=dim)
    vectors = np.eye(dim, dtype=np.float32)
    metadatas = [
        {"chunk_id": f"c{i}", "text": f"text {i}", "source": "doc.md", "index": i} for i in range(dim)
    ]
    store.add(vectors, metadatas)

    embedder = StubEmbedder(dim=dim)
    retriever = Retriever(embedder, store)

    results = retriever.retrieve("some query", top_k=2)
    assert len(results) == 2
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(hasattr(r, "chunk_id") and hasattr(r, "source") for r in results)
