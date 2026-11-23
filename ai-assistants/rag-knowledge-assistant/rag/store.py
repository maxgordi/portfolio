"""Vector stores: Qdrant for real use, an in-memory one for demos and tests.

Both return cosine similarity in [0, 1]-ish range (1 = identical direction).
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

from .ingest import Chunk


@dataclass
class Hit:
    score: float
    text: str
    doc_id: str
    page: int | None


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MemoryStore:
    def __init__(self):
        self.rows: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]):
        self.rows += list(zip(chunks, vectors))

    def delete_doc(self, doc_id: str):
        self.rows = [r for r in self.rows if r[0].doc_id != doc_id]

    def doc_hash(self, doc_id: str) -> str | None:
        return next((c.doc_hash for c, _ in self.rows if c.doc_id == doc_id), None)

    def doc_ids(self) -> set[str]:
        return {c.doc_id for c, _ in self.rows}

    def search(self, vector: list[float], top_k: int) -> list[Hit]:
        scored = [Hit(_cos(vector, v), c.text, c.doc_id, c.page) for c, v in self.rows]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:top_k]


class QdrantStore:
    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url)
        self.collection = collection

    def _ensure(self, dim: int):
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            self.client.create_payload_index(self.collection, "doc_id", PayloadSchemaType.KEYWORD)

    def _doc_filter(self, doc_id: str):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])

    def add(self, chunks: list[Chunk], vectors: list[list[float]]):
        from qdrant_client.models import PointStruct

        self._ensure(len(vectors[0]))
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.doc_id}#{c.index}")),
                vector=v,
                payload={"doc_id": c.doc_id, "text": c.text, "page": c.page, "doc_hash": c.doc_hash},
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(self.collection, points=points)

    def delete_doc(self, doc_id: str):
        if self.client.collection_exists(self.collection):
            self.client.delete(self.collection, points_selector=self._doc_filter(doc_id))

    def doc_hash(self, doc_id: str) -> str | None:
        if not self.client.collection_exists(self.collection):
            return None
        points, _ = self.client.scroll(self.collection, scroll_filter=self._doc_filter(doc_id), limit=1)
        return points[0].payload["doc_hash"] if points else None

    def doc_ids(self) -> set[str]:
        if not self.client.collection_exists(self.collection):
            return set()
        ids, offset = set(), None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset, with_vectors=False)
            ids |= {p.payload["doc_id"] for p in points}
            if offset is None:
                return ids

    def search(self, vector: list[float], top_k: int) -> list[Hit]:
        res = self.client.query_points(self.collection, query=vector, limit=top_k).points
        return [Hit(p.score, p.payload["text"], p.payload["doc_id"], p.payload.get("page")) for p in res]


def make_store(settings):
    if settings.qdrant_url:
        return QdrantStore(settings.qdrant_url, settings.collection)
    return MemoryStore()
