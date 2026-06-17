"""
LLM providers (modular).

RAG concept
-----------
The LLM is the "generation" half of Retrieval-Augmented Generation. It does NOT
answer from its own training knowledge here -- we instruct it to answer ONLY from
the retrieved context, and to refuse when the context is insufficient. That is
what keeps answers grounded and citable.

Two providers are supported and selected via `LLM_PROVIDER` (auto/openai/ollama):
  * openai  -> hosted, paid; used only when OPENAI_API_KEY is present
  * ollama  -> local, free; talked to over its simple HTTP API

Add another provider by implementing `BaseLLM.generate` and registering it in
`get_llm()`.
"""

from __future__ import annotations

import abc
from functools import lru_cache
from typing import Dict, List

import requests

from backend.config import settings
from backend.utils import logger


class LLMError(RuntimeError):
    """Raised when an LLM provider fails to produce a response."""


class BaseLLM(abc.ABC):
    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """
        Generate a completion.

        `messages` is a list of {"role": "user"|"assistant", "content": str}
        representing the conversation so far (for multi-turn context).
        """

    def health(self) -> Dict[str, str]:
        return {"provider": self.name, "model": self.model, "status": "unknown"}


class OpenAILLM(BaseLLM):
    """Hosted chat completion via the OpenAI API."""

    def __init__(self) -> None:
        from openai import OpenAI

        if not settings.openai_enabled:
            raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set.")
        kwargs = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs["base_url"] = settings.OPENAI_BASE_URL
        self._client = OpenAI(**kwargs)
        self.model = settings.OPENAI_MODEL
        self.name = "openai"

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        payload = [{"role": "system", "content": system_prompt}, *messages]
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # surface a clean error to the API layer
            logger.exception("OpenAI generation failed")
            raise LLMError(f"OpenAI request failed: {exc}") from exc

    def health(self) -> Dict[str, str]:
        return {"provider": "openai", "model": self.model, "status": "configured"}


class OllamaLLM(BaseLLM):
    """Local chat completion via the Ollama HTTP API (free)."""

    def __init__(self) -> None:
        self._base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.OLLAMA_MODEL
        self.name = "ollama"

    def generate(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat", json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("message", {}).get("content") or "").strip()
        except requests.RequestException as exc:
            logger.exception("Ollama generation failed")
            raise LLMError(
                f"Ollama request failed ({self._base_url}). "
                f"Is Ollama running and is model '{self.model}' pulled? Error: {exc}"
            ) from exc

    def health(self) -> Dict[str, str]:
        status = "unknown"
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=3)
            status = "reachable" if r.ok else "unreachable"
        except requests.RequestException:
            status = "unreachable"
        return {"provider": "ollama", "model": self.model, "status": status}


@lru_cache
def get_llm() -> BaseLLM:
    """Return a cached LLM based on the resolved provider (auto/openai/ollama)."""
    provider = settings.resolved_llm_provider
    if provider == "openai":
        return OpenAILLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'.")
