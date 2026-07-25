"""Command-line interface: `python -m rag index|query|serve`."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from rag.config import load_config
from rag.generator import OllamaUnavailableError
from rag.pipeline import RagPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Semantic Search & RAG Document Indexing Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Chunk, embed, and index a directory of documents")
    index_parser.add_argument("docs_dir", type=Path, help="Directory containing .md/.txt documents")
    index_parser.add_argument("--chunk-size", type=int, default=None)
    index_parser.add_argument("--chunk-overlap", type=int, default=None)

    query_parser = subparsers.add_parser("query", help="Ask a question against the indexed documents")
    query_parser.add_argument("question")
    query_parser.add_argument("--top-k", type=int, default=None)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def cmd_index(args: argparse.Namespace) -> int:
    config = load_config()
    if args.chunk_size is not None:
        config = replace(config, chunk_size=args.chunk_size)
    if args.chunk_overlap is not None:
        config = replace(config, chunk_overlap=args.chunk_overlap)

    pipeline = RagPipeline(config)
    try:
        stats = pipeline.index_documents(args.docs_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {stats.num_documents} document(s) into {stats.num_chunks} chunk(s) "
        f"in {stats.elapsed_seconds:.1f}s using '{stats.embedding_model}' -> {stats.index_dir}/"
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    pipeline = RagPipeline()
    try:
        result = pipeline.query(args.question, top_k=args.top_k)
    except (RuntimeError, OllamaUnavailableError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.answer)
    print("\nSources:")
    for source in result.sources:
        print(f"  [{source.score:.3f}] {source.source} ({source.chunk_id})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("rag.api:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return cmd_index(args)
    if args.command == "query":
        return cmd_query(args)
    if args.command == "serve":
        return cmd_serve(args)

    parser.print_help()
    return 1
