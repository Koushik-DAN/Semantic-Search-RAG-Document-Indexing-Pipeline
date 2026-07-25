"""Central, env-driven configuration for the pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "gemma4"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_INDEX_DIR = "index"


@dataclass(frozen=True)
class Config:
    embedding_model: str
    ollama_host: str
    ollama_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    index_dir: Path


def load_config() -> Config:
    """Reads current env vars (fresh each call) and returns a Config."""
    return Config(
        embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        ollama_host=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        ollama_model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        chunk_size=int(os.environ.get("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
        top_k=int(os.environ.get("TOP_K", DEFAULT_TOP_K)),
        index_dir=Path(os.environ.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR)),
    )
