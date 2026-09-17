"""
Embedding generation.

Deliberate design choice: use a LOCAL sentence-transformers model instead
of OpenAI's embedding API. Two reasons worth knowing for interviews:
1. Cost/latency -- embeddings are called on every chunk at ingest time and
   every query at search time; a local model has zero marginal cost.
2. It decouples "storage/retrieval" from "generation" -- you can swap the
   generation LLM (Anthropic/OpenAI/local) without re-embedding your whole
   corpus, since the embedding model is independent of the chat model.

Resilience: the first time it runs, sentence-transformers needs to
download model weights from huggingface.co. If that network call fails
(locked-down CI runner, offline dev machine, air-gapped enterprise
network), we fall back to a deterministic hashing-trick vectorizer instead
of crashing. It's lower quality than a real embedding model, but it keeps
ingest/search/tests fully functional offline -- worth knowing this pattern
even if you never hit it in production, because "the vector store is down
because HuggingFace is down" is a real failure mode people hit.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)

_HASH_DIM = 384  # matches all-MiniLM-L6-v2's dimension so ChromaDB collections stay compatible
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache
def _get_sentence_transformer():
    """Returns a loaded SentenceTransformer, or None if it can't be loaded."""
    try:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        return SentenceTransformer(settings.embedding_model_name)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any failure -> fallback
        logger.warning(
            "Falling back to offline hashing embeddings: could not load '%s' (%s). "
            "Real embedding quality requires network access to huggingface.co.",
            get_settings().embedding_model_name,
            e,
        )
        return None


def _hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-words feature-hashing vectorizer. No network, no
    dependencies beyond stdlib. Good enough to prove the pipeline end-to-end;
    not a substitute for a real embedding model's semantic quality."""
    vec = [0.0] * _HASH_DIM
    tokens = _TOKEN_RE.findall(text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % _HASH_DIM
        sign = 1.0 if (h // _HASH_DIM) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_sentence_transformer()
    if model is not None:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
    return [_hash_embed(t) for t in texts]
