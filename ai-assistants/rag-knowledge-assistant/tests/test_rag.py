"""Offline tests: a toy embedder and a fake chat model, no Ollama or Qdrant needed."""
import asyncio
import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from rag.assistant import NOT_FOUND, UNAVAILABLE, Assistant
from rag.config import Settings
from rag.ingest import chunk_document, make_splitter, sync_folder
from rag.llm import LLMUnavailable, parse_answer
from rag.store import MemoryStore

KB = Path(__file__).resolve().parent.parent / "knowledge_base"


class BagOfWordsEmbedder:
    """Hashing trick: good enough to rank chunks that share words with the question."""

    dim = 512

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w{3,}", t.lower()):
                v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % self.dim] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


class FakeChat:
    def __init__(self, reply=None, fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    async def answer(self, system, user):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable()
        return self.reply


def settings(**kw):
    return replace(Settings(), **{"min_score": 0.2, "top_k": 3, **kw})


def test_chunk_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_document("doc", [(1, text)], make_splitter(200, 40))
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)
    tail = chunks[0].text.split()[-2:]
    assert " ".join(tail) in chunks[1].text  # overlap keeps context across the cut


def test_sync_skips_unchanged_and_replaces_changed(tmp_path):
    (tmp_path / "a.md").write_text("Перша версія правил про паролі.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Документ, який потім видалять.", encoding="utf-8")
    store, emb = MemoryStore(), BagOfWordsEmbedder()
    assert sync_folder(tmp_path, store, emb)["added"] == 2
    assert sync_folder(tmp_path, store, emb)["unchanged"] == 2

    (tmp_path / "a.md").write_text("Друга версія правил про паролі.", encoding="utf-8")
    (tmp_path / "b.md").unlink()
    stats = sync_folder(tmp_path, store, emb)
    assert stats["updated"] == 1 and stats["deleted"] == 1
    texts = [c.text for c, _ in store.rows]
    assert texts == ["Друга версія правил про паролі."]  # old vectors are gone


@pytest.fixture
def kb_store():
    store, emb = MemoryStore(), BagOfWordsEmbedder()
    sync_folder(KB, store, emb)
    return store, emb


async def test_answer_with_sources(kb_store):
    store, emb = kb_store
    chat = FakeChat({"status": "success", "answer": "14 символів"})
    res = await Assistant(store, emb, chat, settings()).ask("Скільки символів має бути в паролі?")
    assert res["status"] == "success" and "it-security.md" in res["sources"]
    assert chat.calls == 1


async def test_low_similarity_skips_llm(kb_store):
    store, emb = kb_store
    chat = FakeChat({"status": "success", "answer": "should not be used"})
    res = await Assistant(store, emb, chat, settings(min_score=0.99)).ask("Яка погода на Марсі?")
    assert res == NOT_FOUND and chat.calls == 0


async def test_model_failure_gives_friendly_error(kb_store):
    store, emb = kb_store
    res = await Assistant(store, emb, FakeChat(fail=True), settings()).ask("Які добові під час відрядження за кордоном?")
    assert res == UNAVAILABLE


async def test_concurrency_limit(kb_store):
    store, emb = kb_store
    active = peak = 0

    class SlowChat:
        async def answer(self, system, user):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"status": "success", "answer": "ok"}

    a = Assistant(store, emb, SlowChat(), settings(llm_concurrency=2))
    await asyncio.gather(*(a.ask("Які добові під час відрядження за кордоном?") for _ in range(10)))
    assert peak == 2


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"status": "success", "answer": "24 дні"}', {"status": "success", "answer": "24 дні"}),
        ('```json\n{"status": "not_found", "answer": null}\n```', {"status": "not_found", "answer": None}),
    ],
)
def test_parse_answer_ok(raw, expected):
    assert parse_answer(raw) == expected


@pytest.mark.parametrize("raw", ['Звісно! {"status": "success"}', '{"status": "success", "answer": ""}', '{"foo": 1}'])
def test_parse_answer_rejects_bad_shapes(raw):
    with pytest.raises(ValueError):
        parse_answer(raw)
