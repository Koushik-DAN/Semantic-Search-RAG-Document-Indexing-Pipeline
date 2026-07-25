import pytest


@pytest.fixture
def tmp_corpus_dir(tmp_path):
    docs_dir = tmp_path / "corpus"
    docs_dir.mkdir()
    (docs_dir / "sync.md").write_text(
        "# Sync Conflicts\n"
        "When two devices edit the same file offline, Nimbus creates a conflicted "
        "copy named with the device name and timestamp. To resolve it, open both "
        "versions, merge the changes manually, then delete the extra copy."
    )
    (docs_dir / "billing.md").write_text(
        "# Billing\n"
        "Nimbus offers monthly and annual billing plans. Annual billing gives a "
        "20 percent discount compared to paying monthly."
    )
    return docs_dir
