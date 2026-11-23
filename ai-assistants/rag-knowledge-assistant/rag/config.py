"""All settings come from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:7b-instruct"))
    fallback_model: str = field(default_factory=lambda: _env("FALLBACK_MODEL", "qwen2.5:3b-instruct"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    # Empty QDRANT_URL = in-memory vector store (good for a quick local demo).
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "knowledge_base"))

    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "512")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "64")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    # Below this similarity a chunk is treated as noise. If nothing passes, the LLM is not called.
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.70")))

    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "60")))
    # How many generations may run at once on the local GPU/CPU. The rest wait in a queue.
    llm_concurrency: int = field(default_factory=lambda: int(_env("LLM_CONCURRENCY", "2")))
    queue_timeout_s: float = field(default_factory=lambda: float(_env("QUEUE_TIMEOUT_S", "90")))


settings = Settings()
