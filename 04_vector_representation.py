"""Embedding model configuration."""

from __future__ import annotations

from langchain_community.embeddings import SentenceTransformerEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings() -> SentenceTransformerEmbeddings:
    """Return the SentenceTransformers embedding model used by Chroma."""
    return SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
