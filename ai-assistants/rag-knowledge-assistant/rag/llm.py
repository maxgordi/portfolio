"""Talking to the local Ollama server: embeddings and chat with a fallback model."""
from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Both the main and the fallback model failed (timeout, 5xx, bad JSON)."""


class OllamaEmbedder:
    def __init__(self, url: str, model: str, timeout: float = 60):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["embeddings"]


def parse_answer(raw: str) -> dict:
    """Accept only {"status": "success", "answer": str} or {"status": "not_found", "answer": null}."""
    raw = raw.strip()
    if raw.startswith("```"):  # some models still wrap JSON in a code fence
        raw = raw.strip("`").removeprefix("json").strip()
    data = json.loads(raw)
    status = data.get("status")
    if status == "success" and isinstance(data.get("answer"), str) and data["answer"].strip():
        return {"status": "success", "answer": data["answer"].strip()}
    if status == "not_found":
        return {"status": "not_found", "answer": None}
    raise ValueError(f"unexpected answer shape: {data!r}")


class OllamaChat:
    def __init__(self, url: str, model: str, fallback_model: str | None, timeout: float):
        self.url, self.model, self.fallback, self.timeout = url.rstrip("/"), model, fallback_model, timeout

    async def _ask(self, client: httpx.AsyncClient, model: str, system: str, user: str) -> dict:
        r = await client.post(
            f"{self.url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": "json",  # Ollama constrains the output to valid JSON
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return parse_answer(r.json()["message"]["content"])

    async def answer(self, system: str, user: str) -> dict:
        models = [m for m in (self.model, self.fallback) if m]
        async with httpx.AsyncClient() as client:
            for model in models:
                try:
                    return await self._ask(client, model, system, user)
                except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError, ValueError, KeyError) as e:
                    log.warning("model %s failed: %s", model, e)
        raise LLMUnavailable("all models failed")
