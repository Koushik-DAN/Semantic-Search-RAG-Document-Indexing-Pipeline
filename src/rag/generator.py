"""Client for a local Ollama server, used as the generation step of the RAG pipeline."""

from __future__ import annotations

import requests

from rag.config import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL


class OllamaUnavailableError(RuntimeError):
    pass


class OllamaGenerator:
    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = 120.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host} with model '{self.model}'. "
                "Make sure Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {self.model}`)."
            ) from exc

        data = response.json()
        return data["response"]
