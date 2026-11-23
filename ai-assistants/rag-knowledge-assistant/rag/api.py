"""HTTP API: POST /ask, POST /sync, GET /health.

Run: uvicorn rag.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .assistant import Assistant
from .config import settings
from .ingest import sync_folder
from .llm import OllamaChat, OllamaEmbedder
from .store import make_store

logging.basicConfig(level=logging.INFO)
KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

store = make_store(settings)
embedder = OllamaEmbedder(settings.ollama_url, settings.embed_model)
chat = OllamaChat(settings.ollama_url, settings.llm_model, settings.fallback_model, settings.llm_timeout_s)
assistant = Assistant(store, embedder, chat, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    stats = await asyncio.to_thread(sync_folder, KB_DIR, store, embedder)
    logging.info("knowledge base synced: %s", stats)
    yield


app = FastAPI(title="RAG knowledge assistant (demo)", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


@app.post("/ask")
async def ask(q: Question):
    return await assistant.ask(q.question)


@app.post("/sync")
async def sync():
    """Call from a webhook or cron (e.g. an n8n Schedule node) after the wiki changes."""
    return await asyncio.to_thread(sync_folder, KB_DIR, store, embedder)


@app.get("/health")
async def health():
    return {"ok": True, "documents": len(store.doc_ids())}
