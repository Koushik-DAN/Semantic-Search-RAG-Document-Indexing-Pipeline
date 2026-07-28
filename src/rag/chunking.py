"""Sentence-aware document chunking with overlap.

Documents are split into sentences (markdown headings are kept as their own
atomic unit), then greedily packed into chunks up to ``chunk_size`` characters.
Each new chunk is seeded with the trailing sentences of the previous chunk
(up to ``chunk_overlap`` characters) so retrieval doesn't lose context that
straddles a chunk boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rag.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    index: int


def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentence-like units, one per markdown heading line."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            sentences.append(line)
            continue
        for piece in _SENTENCE_SPLIT_RE.split(line):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def _split_long_sentence(sentence: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Hard char-splits a single sentence longer than chunk_size, with overlap."""
    if chunk_size <= 0:
        return [sentence]
    step = max(chunk_size - chunk_overlap, 1)
    parts: list[str] = []
    start = 0
    n = len(sentence)
    while start < n:
        end = min(start + chunk_size, n)
        parts.append(sentence[start:end])
        if end == n:
            break
        start += step
    return parts


def _joined_len(sentences: list[str]) -> int:
    if not sentences:
        return 0
    return sum(len(s) for s in sentences) + (len(sentences) - 1)


def _take_overlap(buffer: list[str], chunk_overlap: int) -> list[str]:
    """Returns the longest trailing run of sentences from buffer whose
    combined length (with single-space joins) is <= chunk_overlap."""
    overlap: list[str] = []
    length = 0
    for sentence in reversed(buffer):
        extra = len(sentence) + (1 if overlap else 0)
        if length + extra > chunk_overlap:
            break
        overlap.insert(0, sentence)
        length += extra
    return overlap


def _make_chunk(buffer: list[str], source: str, source_stem: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source_stem}::{index:04d}",
        text=" ".join(buffer),
        source=source,
        index=index,
    )


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    for sentence in split_into_sentences(text):
        if len(sentence) > chunk_size:
            sentences.extend(_split_long_sentence(sentence, chunk_size, chunk_overlap))
        else:
            sentences.append(sentence)

    source_stem = Path(source).stem
    chunks: list[Chunk] = []
    buffer: list[str] = []
    index = 0

    for sentence in sentences:
        candidate_len = _joined_len(buffer + [sentence])
        if buffer and candidate_len > chunk_size:
            chunks.append(_make_chunk(buffer, source, source_stem, index))
            index += 1
            buffer = _take_overlap(buffer, chunk_overlap)
        buffer.append(sentence)

    if buffer:
        chunks.append(_make_chunk(buffer, source, source_stem, index))

    return chunks


def chunk_document(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return chunk_text(text, source=path.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_corpus(
    dir_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    dir_path = Path(dir_path)
    chunks: list[Chunk] = []
    for path in sorted(dir_path.iterdir()):
        if path.is_file() and path.suffix.lower() in _DOC_EXTENSIONS:
            chunks.extend(chunk_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return chunks
