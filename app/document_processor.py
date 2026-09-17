"""
Turns raw files into (text, metadata) chunks ready for embedding.

Supports .pdf, .docx, and .txt/.md. Each chunk carries metadata (source
filename + page number where available) so the LLM can cite it later --
that citation requirement is the whole reason we track page numbers at
all, so don't drop this metadata upstream.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from pypdf import PdfReader
import docx

from .chunking import recursive_split
from .config import get_settings


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str
    metadata: dict = field(default_factory=dict)


def _extract_pdf(path: str) -> list[tuple[str, int]]:
    """Returns list of (page_text, page_number)."""
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, i + 1))
    return out


def _extract_docx(path: str) -> list[tuple[str, int]]:
    """docx has no native 'page' concept, so we treat the whole doc as page 1
    and let chunking do the splitting. This is a real limitation worth
    knowing about when you compare parser strategies."""
    document = docx.Document(path)
    text = "\n\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(text, 1)] if text.strip() else []


def _extract_txt(path: str) -> list[tuple[str, int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [(text, 1)] if text.strip() else []


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
    ".md": _extract_txt,
}


def process_file(path: str) -> list[Chunk]:
    """Parse a file, chunk each page/section, return Chunk objects."""
    settings = get_settings()
    ext = os.path.splitext(path)[1].lower()
    if ext not in EXTRACTORS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(EXTRACTORS)}")

    filename = os.path.basename(path)
    pages = EXTRACTORS[ext](path)

    chunks: list[Chunk] = []
    for page_text, page_num in pages:
        pieces = recursive_split(page_text, settings.chunk_size, settings.chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=piece,
                    source=filename,
                    page=page_num,
                    chunk_id=f"{filename}::p{page_num}::c{i}",
                )
            )
    return chunks
