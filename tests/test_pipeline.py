import pytest

from rag.pipeline import RagPipeline


def test_index_and_query_end_to_end(tmp_corpus_dir, tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    monkeypatch.setenv("RAG_INDEX_DIR", str(index_dir))
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")

    pipeline = RagPipeline()
    # Hermetic: stub the generator so this test never needs a real Ollama server.
    monkeypatch.setattr(
        pipeline.generator, "generate", lambda prompt, temperature=0.2: "Stubbed answer citing [1]."
    )

    stats = pipeline.index_documents(tmp_corpus_dir)
    assert stats.num_documents == 2
    assert stats.num_chunks >= 2
    assert (index_dir / "faiss.index").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "meta.json").exists()

    result = pipeline.query("How do I resolve a sync conflict?", top_k=2)
    assert result.answer == "Stubbed answer citing [1]."
    assert any(s.source == "sync.md" for s in result.sources)


def test_query_stream_yields_tokens_and_sources(tmp_corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")

    pipeline = RagPipeline()
    monkeypatch.setattr(
        pipeline.generator, "generate_stream", lambda prompt, temperature=0.2: iter(["Hello", " world"])
    )

    pipeline.index_documents(tmp_corpus_dir)
    sources, token_iter = pipeline.query_stream("How do I resolve a sync conflict?", top_k=2)

    assert any(s.source == "sync.md" for s in sources)
    assert "".join(token_iter) == "Hello world"


def test_query_without_index_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "does-not-exist"))
    pipeline = RagPipeline()
    with pytest.raises(RuntimeError):
        pipeline.query("anything")


def test_health_reports_index_state(tmp_corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    pipeline = RagPipeline()
    assert pipeline.health()["index_loaded"] is False

    pipeline.index_documents(tmp_corpus_dir)
    status = pipeline.health()
    assert status["index_loaded"] is True
    assert status["num_chunks"] > 0
