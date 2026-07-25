from unittest.mock import MagicMock, patch

import pytest
import requests

from rag.generator import OllamaGenerator, OllamaUnavailableError


def test_generate_posts_expected_request_body():
    generator = OllamaGenerator(host="http://localhost:11434", model="llama3.1:8b")

    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Nimbus is a file-sync tool."}
    mock_response.raise_for_status.return_value = None

    with patch("rag.generator.requests.post", return_value=mock_response) as mock_post:
        result = generator.generate("What is Nimbus?", temperature=0.1)

    assert result == "Nimbus is a file-sync tool."
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "model": "llama3.1:8b",
        "prompt": "What is Nimbus?",
        "stream": False,
        "options": {"temperature": 0.1},
    }


def test_generate_raises_actionable_error_on_connection_failure():
    generator = OllamaGenerator(host="http://localhost:11434", model="llama3.1:8b")
    with patch("rag.generator.requests.post", side_effect=requests.ConnectionError()):
        with pytest.raises(OllamaUnavailableError, match="ollama pull llama3.1:8b"):
            generator.generate("hello")


def test_is_available_false_on_connection_error():
    generator = OllamaGenerator()
    with patch("rag.generator.requests.get", side_effect=requests.ConnectionError()):
        assert generator.is_available() is False


def test_is_available_true_on_200():
    generator = OllamaGenerator()
    mock_response = MagicMock(status_code=200)
    with patch("rag.generator.requests.get", return_value=mock_response):
        assert generator.is_available() is True


def test_live_ollama_smoke_test():
    generator = OllamaGenerator()
    if not generator.is_available():
        pytest.skip("Ollama not running locally — start with `ollama serve`")
    result = generator.generate("Reply with the single word: pong")
    assert isinstance(result, str) and len(result) > 0
