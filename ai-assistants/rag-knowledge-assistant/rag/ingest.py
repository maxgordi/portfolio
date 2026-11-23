"""Load documents, split them into chunks and keep the vector store in sync.

Update logic: every document has a stable doc_id (its path relative to the knowledge base)
and a sha256 of its text. On re-ingest:
  * same hash      -> nothing to do;
  * hash changed   -> delete all chunks of doc_id, write the new ones;
  * file is gone   -> delete all chunks of doc_id.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

TEXT_SUFFIXES = {".md", ".txt"}


@dataclass
class Chunk:
    doc_id: str
    index: int
    text: str
    page: int | None
    doc_hash: str


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pdf(path: Path, ocr: bool = True) -> list[tuple[int, str]]:
    """Text layer first; if a page has none (a scan), fall back to OCR."""
    import fitz  # pymupdf

    pages = []
    with fitz.open(path) as doc:
        for num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if not text and ocr:
                try:
                    import pytesseract
                    from PIL import Image

                    img = Image.open(io.BytesIO(page.get_pixmap(dpi=200).tobytes("png")))
                    text = pytesseract.image_to_string(img, lang="ukr+eng").strip()
                except ImportError:
                    text = ""
            if text:
                pages.append((num, text))
    return pages


def read_document(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return [(None, path.read_text(encoding="utf-8"))]


def make_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        # Paragraphs first, then lines, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(doc_id: str, pages: list[tuple[int | None, str]], splitter=None) -> list[Chunk]:
    splitter = splitter or make_splitter()
    full = "\n\n".join(t for _, t in pages)
    h = text_hash(full)
    chunks, i = [], 0
    for page, text in pages:
        for piece in splitter.split_text(text):
            chunks.append(Chunk(doc_id=doc_id, index=i, text=piece, page=page, doc_hash=h))
            i += 1
    return chunks


def sync_folder(folder: Path, store, embedder) -> dict:
    """Bring the store in line with the folder. Returns counts for logging."""
    stats = {"added": 0, "updated": 0, "unchanged": 0, "deleted": 0}
    seen = set()
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES | {".pdf"}:
            continue
        doc_id = path.relative_to(folder).as_posix()
        seen.add(doc_id)
        chunks = chunk_document(doc_id, read_document(path))
        if not chunks:
            continue
        old_hash = store.doc_hash(doc_id)
        if old_hash == chunks[0].doc_hash:
            stats["unchanged"] += 1
            continue
        store.delete_doc(doc_id)
        store.add(chunks, embedder.embed([c.text for c in chunks]))
        stats["updated" if old_hash else "added"] += 1
    for doc_id in store.doc_ids() - seen:
        store.delete_doc(doc_id)
        stats["deleted"] += 1
    return stats
