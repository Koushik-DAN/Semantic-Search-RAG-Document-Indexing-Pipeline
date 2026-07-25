"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel


class IndexRequest(BaseModel):
    docs_dir: str = "data/corpus"


class IndexResponse(BaseModel):
    num_documents: int
    num_chunks: int
    index_dir: str
    embedding_model: str
    elapsed_seconds: float


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class SourceItem(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_chunks: int
    ollama_reachable: bool
    ollama_model: str
    embedding_model: str
