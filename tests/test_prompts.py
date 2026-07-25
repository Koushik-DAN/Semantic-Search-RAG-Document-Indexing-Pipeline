from rag.prompts import build_prompt
from rag.retriever import RetrievedChunk


def test_build_prompt_includes_question_and_numbered_sources():
    contexts = [
        RetrievedChunk(chunk_id="a::0000", text="Nimbus supports SSO.", source="a.md", score=0.9),
        RetrievedChunk(chunk_id="b::0000", text="Nimbus encrypts data at rest.", source="b.md", score=0.8),
    ]
    prompt = build_prompt("Does Nimbus support SSO?", contexts)

    assert "Does Nimbus support SSO?" in prompt
    assert "[1]" in prompt and "[2]" in prompt
    assert "a.md" in prompt and "b.md" in prompt
    assert "Nimbus supports SSO." in prompt


def test_build_prompt_handles_empty_context_gracefully():
    prompt = build_prompt("What is Nimbus?", [])
    assert "What is Nimbus?" in prompt
    assert "no relevant context" in prompt.lower()
