import numpy as np

from rag.vector_store import FaissVectorStore


def _make_metadata(n: int) -> list[dict]:
    return [
        {"chunk_id": f"doc::{i:04d}", "text": f"chunk text {i}", "source": "doc.md", "index": i}
        for i in range(n)
    ]


def test_search_returns_nearest_neighbor_in_rank_order():
    dim = 8
    store = FaissVectorStore(dim=dim)

    vectors = np.eye(dim, dtype=np.float32)  # orthogonal unit vectors
    store.add(vectors, _make_metadata(dim))

    # query near-identical to row 3, should rank it first
    query = vectors[3].copy()
    query[3] += 0.01

    results = store.search(query, top_k=3)
    assert results[0].chunk_id == "doc::0003"
    assert all(-1.0 <= r.score <= 1.0 for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_len_reflects_number_of_added_vectors():
    store = FaissVectorStore(dim=4)
    assert len(store) == 0
    store.add(np.random.rand(3, 4).astype(np.float32), _make_metadata(3))
    assert len(store) == 3


def test_save_and_load_round_trip(tmp_path):
    dim = 6
    store = FaissVectorStore(dim=dim)
    vectors = np.random.rand(5, dim).astype(np.float32)
    store.add(vectors, _make_metadata(5))

    index_dir = tmp_path / "index"
    store.save(index_dir, embedding_model="test-model", chunk_size=1000, chunk_overlap=200)

    loaded = FaissVectorStore.load(index_dir, expected_embedding_model="test-model")
    assert len(loaded) == len(store)

    query = vectors[2]
    original_results = store.search(query, top_k=3)
    loaded_results = loaded.search(query, top_k=3)
    assert [r.chunk_id for r in original_results] == [r.chunk_id for r in loaded_results]
