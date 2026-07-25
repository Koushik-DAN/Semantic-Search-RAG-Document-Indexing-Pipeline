"""Assembles the RAG prompt from a question and retrieved context chunks."""

from __future__ import annotations

from rag.retriever import RetrievedChunk

SYSTEM_PREAMBLE = (
    "You are a helpful assistant answering questions about the Nimbus product "
    "documentation. Use ONLY the provided context to answer the question. "
    "If the answer is not contained in the context, say you don't know. "
    "Cite the sources you used with their [n] marker."
)


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> str:
    if contexts:
        context_block = "\n\n".join(
            f"[{i + 1}] (source: {c.source})\n{c.text}" for i, c in enumerate(contexts)
        )
    else:
        context_block = "(no relevant context was found)"

    return (
        f"{SYSTEM_PREAMBLE}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
