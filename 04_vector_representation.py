"""Embedding model configuration."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the SentenceTransformers embedding model used by Chroma.

    Uses the maintained ``langchain-huggingface`` package instead of the
    deprecated ``langchain_community.embeddings.SentenceTransformerEmbeddings``,
    which is being sunset.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
