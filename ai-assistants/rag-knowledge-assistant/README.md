# RAG Knowledge Assistant (self-hosted)

> **Status: demo / work in progress.** The core pipeline and tests are done; the n8n part and
> production hardening are described in [docs/architecture.md](docs/architecture.md) but not built.

An internal assistant that answers employees' questions **only** from a private knowledge base
(policies, instructions, wiki). Everything runs inside a closed perimeter: no OpenAI / Anthropic /
Gemini APIs, the LLM and the vector database are local.

The sample knowledge base in `knowledge_base/` describes a **fictional** company, Acme Corp.

## How it works

```
question ──► embed (Ollama, nomic-embed-text) ──► Qdrant top-k
                                                     │
                     all scores < MIN_SCORE ◄────────┤
                     → {"status": "not_found"}       │ relevant chunks
                       (LLM is not called)           ▼
                                      queue (max LLM_CONCURRENCY at once)
                                                     ▼
                              Qwen2.5 7B ──timeout / 5xx / bad JSON──► Qwen2.5 3B
                                                     │                     │ fails too
                                                     ▼                     ▼
                         {"status": "success", "answer": ..., "sources": [...]}
                                                     "Сервіс тимчасово недоступний"
```

| Concern | Where |
|---|---|
| Chunking (`chunk_size=512`, `chunk_overlap=64`, paragraph → line → sentence → word) | `rag/ingest.py` |
| PDF text layer, OCR fallback for scanned pages | `rag/ingest.py` → `read_pdf` |
| Re-indexing changed / deleted documents by `doc_id` + sha256 | `rag/ingest.py` → `sync_folder` |
| Similarity threshold, no LLM call when nothing is relevant | `rag/assistant.py` |
| Strict JSON system prompt and output validation | `rag/assistant.py`, `rag/llm.py` → `parse_answer` |
| Fallback model and friendly error | `rag/llm.py` → `OllamaChat.answer` |
| Queue / concurrency limit for the local model | `rag/assistant.py` (semaphore + wait timeout) |

## Run

```bash
cp .env.example .env
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:7b-instruct
docker compose exec ollama ollama pull qwen2.5:3b-instruct
docker compose exec ollama ollama pull nomic-embed-text
docker compose restart api

curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Скільки днів відпустки на рік?"}'
# {"status":"success","answer":"24 календарні дні.","sources":["vacation-policy.md"]}
```

After editing files in `knowledge_base/`, call `POST /sync` (from cron, a webhook or an n8n
Schedule node). Unchanged documents are skipped; changed ones are re-embedded; deleted ones are
removed from the index.

Without Docker: run Ollama locally, leave `QDRANT_URL` empty (in-memory store) and start
`uvicorn rag.api:app`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a toy embedder and a fake model, so they run offline in under a second.

## Model choice

* **Generation:** Qwen2.5 7B Instruct (good Ukrainian and English, follows JSON instructions);
  Qwen2.5 3B as the fallback or for a weak server; Mistral 7B Instruct as an alternative.
* **Embeddings:** `nomic-embed-text` through Ollama, or `intfloat/multilingual-e5-large` for
  better multilingual recall (runs on CPU).
