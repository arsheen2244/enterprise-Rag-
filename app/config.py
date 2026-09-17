"""
Central configuration for the Enterprise AI Knowledge Assistant.

Why this file exists (learning note):
Hard-coding model names, chunk sizes, and API keys throughout the codebase
makes a project brittle. Every production RAG system centralizes its knobs
in one place so you can tune retrieval quality without hunting through files.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider ---
    # "anthropic" or "openai". You only need ONE key set for the provider you pick.
    llm_provider: str = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # --- Embeddings ---
    # Local, free, no API key needed. Runs on CPU. 384-dim vectors.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking ---
    chunk_size: int = 800        # characters, not tokens (simple + good enough to learn on)
    chunk_overlap: int = 120     # ~15% overlap so context isn't severed at chunk edges

    # --- Retrieval ---
    top_k: int = 4

    # --- Storage ---
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "enterprise_docs"


@lru_cache
def get_settings() -> Settings:
    """Cached so we parse env vars once per process, not on every request."""
    return Settings()
