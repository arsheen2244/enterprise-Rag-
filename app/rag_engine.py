"""
Glues retrieval (vector_store) and generation (llm) together. Kept
deliberately thin -- if this file starts accumulating parsing or prompt
logic, that's a sign it belongs in document_processor.py or llm.py instead.
"""
from __future__ import annotations

from .llm import generate_answer
from .vector_store import get_store


def answer_question(question: str, top_k: int | None = None) -> dict:
    store = get_store()
    hits = store.search(question, top_k=top_k)
    answer = generate_answer(question, hits)

    # Dedupe sources for a clean citation list in the API response
    seen = set()
    sources = []
    for h in hits:
        key = (h["source"], h["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"source": h["source"], "page": h["page"]})

    return {"answer": answer, "sources": sources, "retrieved_chunks": len(hits)}
