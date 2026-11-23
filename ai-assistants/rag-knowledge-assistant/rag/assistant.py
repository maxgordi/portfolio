"""Retrieve -> filter by similarity -> ask the LLM only when there is real context."""
from __future__ import annotations

import asyncio

from .llm import LLMUnavailable

SYSTEM_PROMPT = """You are an assistant for an internal corporate knowledge base.
Your only job is to answer questions strictly from the context fragments provided.

RULES:
1. Use ONLY information that is explicitly stated in the context. No outside knowledge,
   no assumptions, no reasoning beyond the given text.
2. If the context contains a clear answer, return it. Answer in the language of the question.
3. If the context is missing, ambiguous or does not contain the answer, return not_found.
   Never guess or extrapolate.
4. Output a single valid JSON object. No preamble, no explanations, no markdown outside JSON.

OUTPUT FORMAT:
Answer found:     {"status": "success", "answer": "<answer>"}
Answer not found: {"status": "not_found", "answer": null}"""

NOT_FOUND = {"status": "not_found", "answer": None, "sources": []}
UNAVAILABLE = {
    "status": "error",
    "answer": "Сервіс тимчасово недоступний. Спробуйте, будь ласка, за хвилину.",
    "sources": [],
}
BUSY = {
    "status": "error",
    "answer": "Зараз забагато запитів. Спробуйте, будь ласка, за хвилину.",
    "sources": [],
}


def build_user_message(question: str, hits) -> str:
    context = "\n\n".join(f"[{i}] ({h.doc_id}) {h.text}" for i, h in enumerate(hits, 1))
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"


class Assistant:
    def __init__(self, store, embedder, chat, settings):
        self.store, self.embedder, self.chat, self.s = store, embedder, chat, settings
        # Only N generations at a time; the rest wait here instead of overloading the model.
        self.slots = asyncio.Semaphore(settings.llm_concurrency)

    def retrieve(self, question: str):
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, self.s.top_k)
        return [h for h in hits if h.score >= self.s.min_score]

    async def ask(self, question: str) -> dict:
        hits = await asyncio.to_thread(self.retrieve, question)
        if not hits:
            return NOT_FOUND  # nothing relevant: do not even call the LLM
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=self.s.queue_timeout_s)
        except asyncio.TimeoutError:
            return BUSY
        try:
            result = await self.chat.answer(SYSTEM_PROMPT, build_user_message(question, hits))
        except LLMUnavailable:
            return UNAVAILABLE
        finally:
            self.slots.release()
        if result["status"] == "not_found":
            return NOT_FOUND
        sources = sorted({h.doc_id for h in hits})
        return {**result, "sources": sources}
