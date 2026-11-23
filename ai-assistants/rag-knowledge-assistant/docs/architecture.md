# Architecture notes

Design for a fully self-hosted corporate assistant. Parts marked *(planned)* are not in the code yet.

## Stack (Docker)

| Role | Tool |
|---|---|
| Pipeline orchestrator | n8n self-hosted *(planned; the Python API is the core today)* |
| Local LLM runtime | Ollama |
| Vector database | Qdrant (or Postgres + pgvector / Supabase self-hosted) |
| Queue / cache | Redis + n8n queue mode *(planned)* |
| Reverse proxy, TLS, auth | nginx *(planned)* |

With more resources: a GPU for Ollama and a larger model (Qwen2.5 14B/32B).

## Keeping vectors in sync with the wiki

Every chunk stores `doc_id` and the sha256 of the whole document.

1. A webhook from the wiki (or a cron job) triggers a sync.
2. For each document: fetch the current text, compute the hash.
3. Same hash → skip. Different hash → delete every chunk with this `doc_id`, embed and insert the new ones.
4. Documents that no longer exist → delete their chunks.

Implemented in `rag/ingest.py::sync_folder`; in n8n this is *Schedule/Webhook → HTTP Request (fetch) →
Code (hash) → IF (changed?) → Qdrant delete by filter → Embeddings → Qdrant insert*.

## Fighting hallucinations

* Similarity threshold (`MIN_SCORE`, ~0.7–0.8 for e5/nomic). If the best chunk is below it, the
  LLM is **not called** and the user gets `not_found`.
* The system prompt allows only facts from the context and forces a single JSON object.
* Ollama `format: "json"` + `temperature: 0` + server-side validation (`parse_answer`). Anything
  that does not match the schema counts as a failure and goes to the fallback model.

## Failures and load

* Timeout or 5xx from the main model → retry on the smaller fallback model → if it also fails,
  a polite "service temporarily unavailable" message. The user never waits in silence.
* At most `LLM_CONCURRENCY` generations at a time; other requests wait up to `QUEUE_TIMEOUT_S`,
  then get a "too many requests" answer instead of hanging.
* *(planned)* n8n queue mode with workers, per-user rate limit, dead-letter after 3 failed tries.
