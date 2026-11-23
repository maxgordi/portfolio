"""Minimal provider-agnostic LLM client (Gemini / Anthropic / OpenAI) returning parsed JSON."""
import json
import re

import httpx

from .config import LLM_API_KEY, LLM_BASE_URL, LLM_EFFORT, LLM_MODEL, LLM_PROVIDER

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-haiku-4-5",
                  "openai": "gpt-4.1-mini", "local": "qwen2.5:7b"}


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError(f"no JSON in LLM answer: {text[:200]}")
    return json.loads(m.group(0))


async def ask_json(system: str, user: str, max_tokens: int = 3000) -> dict:
    if not LLM_API_KEY and LLM_PROVIDER != "local":
        raise LLMError("LLM_API_KEY is not set")
    model = LLM_MODEL or DEFAULT_MODELS[LLM_PROVIDER]
    async with httpx.AsyncClient(timeout=300) as c:
        if LLM_PROVIDER == "gemini":
            r = await c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": LLM_API_KEY},
                json={"systemInstruction": {"parts": [{"text": system}]},
                      "contents": [{"role": "user", "parts": [{"text": user}]}],
                      "generationConfig": {"responseMimeType": "application/json",
                                           "maxOutputTokens": max_tokens, "temperature": 0.3}})
            if r.status_code >= 400:
                raise LLMError(f"{LLM_PROVIDER} {r.status_code}: {r.text[:300]}")
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        elif LLM_PROVIDER == "anthropic":
            r = await c.post("https://api.anthropic.com/v1/messages",
                             headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01"},
                             json={"model": model, "max_tokens": max_tokens, "system": system,
                                   "messages": [{"role": "user", "content": user}],
                                   **({"output_config": {"effort": LLM_EFFORT}} if LLM_EFFORT else {})})
            if r.status_code >= 400:
                raise LLMError(f"{LLM_PROVIDER} {r.status_code}: {r.text[:300]}")
            text = "".join(b.get("text", "") for b in r.json()["content"])
        elif LLM_PROVIDER in ("openai", "local"):
            # "local" = Ollama / LM Studio (OpenAI-compatible API on your computer, free)
            base = LLM_BASE_URL or ("https://api.openai.com/v1" if LLM_PROVIDER == "openai"
                                    else "http://localhost:11434/v1")
            r = await c.post(f"{base}/chat/completions",
                             headers={"Authorization": f"Bearer {LLM_API_KEY or 'local'}"},
                             json={"model": model, "max_tokens": max_tokens, "temperature": 0.3,
                                   "response_format": {"type": "json_object"},
                                   "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": user}]})
            if r.status_code >= 400:
                raise LLMError(f"{LLM_PROVIDER} {r.status_code}: {r.text[:300]}")
            text = r.json()["choices"][0]["message"]["content"]
        else:
            raise LLMError(f"unknown LLM_PROVIDER {LLM_PROVIDER}")
    return _extract_json(text)
