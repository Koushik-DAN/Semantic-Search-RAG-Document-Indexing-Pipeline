# Semantic Search & RAG Document Indexing Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built with Python, Sentence-Transformers, and FAISS. Documents are chunked, embedded, and indexed for cosine-similarity search; retrieved snippets are assembled into a prompt and answered by a local Ollama model — no cloud API keys required.

## Features

- **Sentence-aware chunking** with configurable size and overlap, so context isn't lost at chunk boundaries.
- **Sentence-Transformers embeddings** (`all-MiniLM-L6-v2` by default), auto-using Apple Silicon MPS when available.
- **FAISS vector index** (`IndexFlatIP` on L2-normalized vectors) for exact cosine-similarity search.
- **Local generation via Ollama** (default `llama3.1:8b`) — fully offline after setup.
- **CLI and FastAPI service** sharing one pipeline implementation, so both interfaces behave identically.
- Bundled sample corpus (a fictional product, "Nimbus") so the pipeline runs out of the box.

## Architecture

```
Documents (.md/.txt) → Chunker → Embedder → FAISS Index
                                                  │
Question → Embedder ──────────────────────→ Retriever → top-k chunks
                                                  │
                                          Prompt Builder
                                                  │
                                         Ollama (local LLM)
                                                  │
                                               Answer
```

## Requirements

- Python 3.11+
- macOS, Linux, or Windows (Apple Silicon: embeddings automatically use MPS)
- [Ollama](https://ollama.com) for local LLM generation

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package
pip install -e .

# 3. Install and start Ollama
brew install ollama
ollama serve &          # or launch the Ollama menu-bar app
ollama pull llama3.1:8b

# 4. Copy the example environment file (optional — defaults work out of the box)
cp .env.example .env
```

## Quickstart

```bash
# Index the bundled sample corpus
python -m rag index data/corpus

# Ask a question against it
python -m rag query "How do I resolve a Nimbus sync conflict?"

# Or run the REST API
python -m rag serve
```

With the API running:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"docs_dir": "data/corpus"}'

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What roles can a workspace member have?"}'
```

## Configuration

All settings are read from environment variables (or a `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence-Transformers model used for embeddings |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model used for answer generation |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Max characters of trailing overlap seeded into the next chunk |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `RAG_INDEX_DIR` | `index` | Directory where the FAISS index and metadata are persisted |

## Chunking Strategy

Documents are split into sentences (markdown headings are kept as their own atomic unit), which are then greedily packed into chunks up to `CHUNK_SIZE` characters. Each new chunk is seeded with the trailing sentences of the previous chunk, up to `CHUNK_OVERLAP` characters, so a fact split across a chunk boundary is still fully present in at least one chunk. A single sentence longer than `CHUNK_SIZE` falls back to a hard character split so chunking never stalls on pathological input.

## Cosine Similarity with FAISS

Embeddings are L2-normalized both at embedding time (`normalize_embeddings=True`) and again defensively before insertion/search (`faiss.normalize_L2`). The index is a plain `faiss.IndexFlatIP` (inner product). The inner product of two unit-length vectors is exactly their cosine similarity, so this is the standard, numerically correct way to get cosine-similarity search out of FAISS without a custom distance function.

## Running Tests

```bash
pytest -q
```

The Ollama-dependent live test in `test_generator.py` automatically skips if Ollama isn't reachable at `OLLAMA_HOST`; everything else (chunking, embeddings, vector store, retrieval, prompts, pipeline) is hermetic and requires no external services.

## Project Structure

```
rag-document-pipeline/
├── data/corpus/          # bundled sample documents
├── index/                # generated FAISS index (gitignored)
├── src/rag/
│   ├── config.py         # env-driven settings
│   ├── chunking.py       # sentence-aware chunker
│   ├── embeddings.py     # Sentence-Transformers wrapper
│   ├── vector_store.py   # FAISS index wrapper (add/search/save/load)
│   ├── retriever.py      # embed query + search
│   ├── prompts.py        # RAG prompt template
│   ├── generator.py      # Ollama HTTP client
│   ├── pipeline.py        # orchestration layer used by both CLI and API
│   ├── schemas.py        # pydantic request/response models
│   ├── cli.py             # `python -m rag index|query|serve`
│   └── api.py              # FastAPI app
└── tests/
```

## Limitations / Possible Extensions

- Indexing is a full rebuild each run, not incremental upsert.
- No re-ranking step after initial retrieval.
- No response streaming (Ollama responses are returned in full).
- Single local FAISS index; no distributed/sharded index support.
- No authentication on the FastAPI service — add a reverse proxy or API-key middleware before exposing it beyond localhost.

## License

MIT
