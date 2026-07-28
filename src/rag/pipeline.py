"""Single orchestration layer shared by the CLI and the FastAPI service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import chunk_corpus
from rag.config import Config, load_config
from rag.embeddings import Embedder
from rag.generator import OllamaGenerator, OllamaUnavailableError
from rag.prompts import build_prompt
from rag.retriever import RetrievedChunk, Retriever
from rag.vector_store import FaissVectorStore


@dataclass(frozen=True)
class IndexStats:
    num_documents: int
    num_chunks: int
    index_dir: str
    embedding_model: str
    elapsed_seconds: float


@dataclass(frozen=True)
class QueryResult:
    answer: str
    sources: list[RetrievedChunk]


class RagPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.embedder = Embedder(self.config.embedding_model)
        self.generator = OllamaGenerator(self.config.ollama_host, self.config.ollama_model)
        self.store: FaissVectorStore | None = self._try_load_existing_index()

    def _try_load_existing_index(self) -> FaissVectorStore | None:
        index_dir = Path(self.config.index_dir)
        if not (index_dir / "meta.json").exists():
            return None
        try:
            return FaissVectorStore.load(index_dir, expected_embedding_model=self.config.embedding_model)
        except (OSError, KeyError, ValueError):
            return None

    def index_documents(self, docs_dir: Path) -> IndexStats:
        start = time.monotonic()
        docs_dir = Path(docs_dir)

        chunks = chunk_corpus(docs_dir, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            raise ValueError(f"No .md, .txt, or .pdf documents found in {docs_dir}")

        vectors = self.embedder.embed([c.text for c in chunks])
        store = FaissVectorStore(dim=self.embedder.dim)
        metadatas = [
            {"chunk_id": c.chunk_id, "text": c.text, "source": c.source, "index": c.index}
            for c in chunks
        ]
        store.add(vectors, metadatas)
        store.save(
            self.config.index_dir,
            embedding_model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        self.store = store

        num_documents = len({c.source for c in chunks})
        return IndexStats(
            num_documents=num_documents,
            num_chunks=len(chunks),
            index_dir=str(self.config.index_dir),
            embedding_model=self.config.embedding_model,
            elapsed_seconds=time.monotonic() - start,
        )

    def _retrieve(self, question: str, top_k: int | None) -> tuple[list[RetrievedChunk], str]:
        if self.store is None:
            raise RuntimeError("No index found — run `index_documents` (or `python -m rag index`) first.")

        retriever = Retriever(self.embedder, self.store)
        sources = retriever.retrieve(question, top_k=top_k or self.config.top_k)
        prompt = build_prompt(question, sources)
        return sources, prompt

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        sources, prompt = self._retrieve(question, top_k)
        answer = self.generator.generate(prompt)
        return QueryResult(answer=answer, sources=sources)

    def query_stream(self, question: str, top_k: int | None = None):
        sources, prompt = self._retrieve(question, top_k)
        return sources, self.generator.generate_stream(prompt)

    def health(self) -> dict:
        return {
            "index_loaded": self.store is not None,
            "num_chunks": len(self.store) if self.store is not None else 0,
            "ollama_reachable": self.generator.is_available(),
            "ollama_model": self.config.ollama_model,
            "embedding_model": self.config.embedding_model,
        }


__all__ = ["RagPipeline", "IndexStats", "QueryResult", "OllamaUnavailableError"]
