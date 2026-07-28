"""FastAPI service exposing the RAG pipeline over HTTP.

Endpoints are thin adapters over a single RagPipeline instance built once at
startup — all real logic (chunking, embedding, retrieval, generation) lives
in rag.pipeline, shared with the CLI.
"""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from rag.chunking import SUPPORTED_EXTENSIONS
from rag.generator import OllamaUnavailableError
from rag.pipeline import RagPipeline
from rag.schemas import (
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
    UploadRequest,
    UploadResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RagPipeline()
    yield


app = FastAPI(title="RAG Document Pipeline", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    pipeline: RagPipeline = app.state.pipeline
    status = pipeline.health()
    return HealthResponse(status="ok", **status)


@app.post("/index", response_model=IndexResponse)
def post_index(request: IndexRequest) -> IndexResponse:
    pipeline: RagPipeline = app.state.pipeline
    try:
        stats = pipeline.index_documents(Path(request.docs_dir))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IndexResponse(**stats.__dict__)


@app.post("/upload", response_model=UploadResponse)
def post_upload(request: UploadRequest) -> UploadResponse:
    pipeline: RagPipeline = app.state.pipeline

    filename = Path(request.filename).name  # strip any directory components — no path traversal
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or '(none)'}")

    try:
        raw = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 content: {exc}") from exc

    docs_dir = Path(request.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    dest = docs_dir / filename
    dest.write_bytes(raw)

    try:
        stats = pipeline.index_documents(docs_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadResponse(filename=filename, saved_path=str(dest), **stats.__dict__)


@app.post("/query", response_model=QueryResponse)
def post_query(request: QueryRequest) -> QueryResponse:
    pipeline: RagPipeline = app.state.pipeline
    try:
        result = pipeline.query(request.question, top_k=request.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceItem(chunk_id=s.chunk_id, source=s.source, score=s.score, text=s.text)
            for s in result.sources
        ],
    )
