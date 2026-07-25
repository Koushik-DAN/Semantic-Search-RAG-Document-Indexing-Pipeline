import numpy as np
import pytest

from rag.embeddings import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_returns_normalized_matrix_of_correct_shape(embedder):
    texts = ["The cat sat on the mat.", "FAISS is a vector search library.", "Ollama runs local LLMs."]
    vectors = embedder.embed(texts)
    assert vectors.shape == (len(texts), embedder.dim)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_embed_query_shape_matches_embed(embedder):
    vector = embedder.embed_query("What is semantic search?")
    assert vector.shape == (embedder.dim,)
    assert abs(np.linalg.norm(vector) - 1.0) < 1e-4


def test_dim_matches_actual_model_output(embedder):
    vectors = embedder.embed(["hello world"])
    assert vectors.shape[1] == embedder.dim
