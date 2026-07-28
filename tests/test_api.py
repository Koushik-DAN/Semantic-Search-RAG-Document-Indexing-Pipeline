import base64
from pathlib import Path

from fastapi.testclient import TestClient

from rag.api import app

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_upload_md_file_indexes_successfully(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"
    content = b"# Title\nSome useful onboarding content sentence here."

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            json={
                "filename": "notes.md",
                "content_base64": base64.b64encode(content).decode(),
                "docs_dir": str(docs_dir),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["num_documents"] == 1
    assert body["num_chunks"] >= 1
    assert (docs_dir / "notes.md").read_bytes() == content


def test_upload_pdf_file_indexes_successfully(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"
    pdf_bytes = (FIXTURES_DIR / "sample.pdf").read_bytes()

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            json={
                "filename": "sample.pdf",
                "content_base64": base64.b64encode(pdf_bytes).decode(),
                "docs_dir": str(docs_dir),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["num_chunks"] == 3
    assert (docs_dir / "sample.pdf").read_bytes() == pdf_bytes


def test_upload_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            json={
                "filename": "malware.exe",
                "content_base64": base64.b64encode(b"anything").decode(),
                "docs_dir": str(docs_dir),
            },
        )

    assert response.status_code == 400
    assert not (docs_dir / "malware.exe").exists()


def test_upload_rejects_invalid_base64(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            json={
                "filename": "notes.md",
                "content_base64": "not-valid-base64!!!",
                "docs_dir": str(docs_dir),
            },
        )

    assert response.status_code == 400


def test_upload_sanitizes_path_traversal_in_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"
    content = b"Traversal attempt content sentence."

    with TestClient(app) as client:
        response = client.post(
            "/upload",
            json={
                "filename": "../../etc/evil.md",
                "content_base64": base64.b64encode(content).decode(),
                "docs_dir": str(docs_dir),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "evil.md"
    assert (docs_dir / "evil.md").exists()
    assert not (tmp_path / "etc").exists()


def test_query_stream_emits_sources_tokens_then_done(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "index"))
    docs_dir = tmp_path / "corpus"
    content = b"# Sync Conflicts\nWhen two devices edit the same file offline, Nimbus creates a conflicted copy."

    with TestClient(app) as client:
        upload_response = client.post(
            "/upload",
            json={
                "filename": "sync.md",
                "content_base64": base64.b64encode(content).decode(),
                "docs_dir": str(docs_dir),
            },
        )
        assert upload_response.status_code == 200

        app.state.pipeline.generator.generate_stream = lambda prompt, temperature=0.2: iter(
            ["Nimbus", " creates", " a conflicted copy."]
        )

        response = client.post("/query/stream", json={"question": "What happens on a sync conflict?"})

    assert response.status_code == 200
    frames = [f for f in response.text.split("\n\n") if f.strip()]
    events = [f.split("\n")[0] for f in frames]

    assert events[0] == "event: sources"
    assert events[1:-1] == ["event: token"] * 3
    assert events[-1] == "event: done"

    token_frames = frames[1:-1]
    tokens = [json.loads(f.split("\n")[1][len("data: ") :])["text"] for f in token_frames]
    assert "".join(tokens) == "Nimbus creates a conflicted copy."
