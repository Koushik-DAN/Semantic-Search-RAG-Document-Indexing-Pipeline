from pathlib import Path

from rag.chunking import (
    chunk_corpus,
    chunk_document,
    chunk_text,
    split_into_sentences,
    _split_long_sentence,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_split_into_sentences_does_not_break_mid_word():
    text = "Dr. Smith went to Washington. He arrived at 3 p.m. yesterday."
    sentences = split_into_sentences(text)
    assert all(sentences)
    assert "".join(sentences).count("Washington") == 1


def test_split_into_sentences_keeps_headings_atomic():
    text = "# Title Heading\nSome body sentence here. Another sentence follows."
    sentences = split_into_sentences(text)
    assert sentences[0] == "# Title Heading"
    assert len(sentences) == 3


def test_chunk_text_respects_chunk_size_with_tolerance():
    sentence = "This is a moderately long sentence used for testing chunk boundaries. "
    text = sentence * 40
    chunks = chunk_text(text, source="doc.txt", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200 + len(sentence)


def test_consecutive_chunks_overlap():
    sentence = "Sentence number {} provides some unique padding content here. "
    text = "".join(sentence.format(i) for i in range(30))
    chunks = chunk_text(text, source="doc.txt", chunk_size=300, chunk_overlap=100)
    assert len(chunks) > 1
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.strip(". ") in chunks[1].text


def test_short_text_produces_single_chunk():
    text = "Just one short sentence."
    chunks = chunk_text(text, source="doc.txt", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "doc::0000"


def test_empty_text_produces_no_chunks():
    assert chunk_text("   ", source="doc.txt") == []


def test_split_long_sentence_fallback_terminates():
    giant = "x" * 5000
    parts = _split_long_sentence(giant, chunk_size=1000, chunk_overlap=200)
    assert len(parts) > 1
    assert all(len(p) <= 1000 for p in parts)
    assert "".join(dict.fromkeys(parts))  # sanity: produced non-empty output


def test_chunk_text_handles_long_unpunctuated_sentence_without_hanging():
    text = "word " * 2000  # one giant "sentence" with no terminal punctuation
    chunks = chunk_text(text, source="doc.txt", chunk_size=500, chunk_overlap=100)
    assert len(chunks) > 1


def test_chunk_document_sets_source_and_chunk_ids(tmp_path):
    doc_path = tmp_path / "readme.md"
    doc_path.write_text("# Readme\nFirst sentence here. Second sentence follows here.")
    chunks = chunk_document(doc_path, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].source == "readme.md"
    assert chunks[0].chunk_id == "readme::0000"


def test_chunk_corpus_reads_all_md_and_txt_files(tmp_path):
    (tmp_path / "a.md").write_text("First document content sentence.")
    (tmp_path / "b.txt").write_text("Second document content sentence.")
    (tmp_path / "c.json").write_text('{"ignored": true}')

    chunks = chunk_corpus(tmp_path)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "b.txt"}


def test_chunk_document_extracts_pdf_text_per_page_with_page_in_chunk_id():
    chunks = chunk_document(FIXTURES_DIR / "sample.pdf", chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 3
    assert all(c.source == "sample.pdf" for c in chunks)
    assert [c.chunk_id for c in chunks] == ["sample::p001::0000", "sample::p002::0000", "sample::p003::0000"]
    assert "onboarding" in chunks[0].text
    assert "two factor authentication" in chunks[1].text
    assert "audit log" in chunks[2].text


def test_chunk_corpus_picks_up_pdf_alongside_md_and_txt(tmp_path):
    import shutil

    (tmp_path / "a.md").write_text("First document content sentence.")
    shutil.copy(FIXTURES_DIR / "sample.pdf", tmp_path / "sample.pdf")

    chunks = chunk_corpus(tmp_path)
    sources = {c.source for c in chunks}
    assert sources == {"a.md", "sample.pdf"}


def test_chunk_pdf_skips_pages_with_no_extractable_text(monkeypatch):
    from rag import chunking

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage("Real content on this page."), FakePage(""), FakePage("   ")]

    monkeypatch.setattr(chunking, "PdfReader", FakeReader, raising=False)
    monkeypatch.setitem(
        __import__("sys").modules,
        "pypdf",
        type("M", (), {"PdfReader": FakeReader})(),
    )

    chunks = chunking._chunk_pdf(FIXTURES_DIR / "sample.pdf", chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "sample::p001::0000"
