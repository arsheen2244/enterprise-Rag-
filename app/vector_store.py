"""
Thin wrapper around ChromaDB so the rest of the app never talks to Chroma
directly. This isolation matters: if you ever swap ChromaDB for Pinecone,
Qdrant, or pgvector, only this file changes.

We pass in our own embeddings (from embeddings.py) rather than letting
Chroma call an embedding function itself -- that keeps embedding logic in
one obvious place instead of split across two files.
"""
from __future__ import annotations

import chromadb

from .config import get_settings
from .document_processor import Chunk
from .embeddings import embed_texts


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = embed_texts([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        settings = get_settings()
        k = top_k or settings.top_k
        query_vector = embed_texts([query])[0]
        results = self._collection.query(query_embeddings=[query_vector], n_results=k)

        hits = []
        if results["documents"] and results["documents"][0]:
            for text, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                hits.append({"text": text, "source": meta["source"], "page": meta["page"], "distance": dist})
        return hits

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        settings = get_settings()
        self._client.delete_collection(settings.collection_name)
        self._collection = self._client.get_or_create_collection(name=settings.collection_name)


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
