"""Context retrieval utilities."""

from __future__ import annotations

import importlib
from pathlib import Path

from langchain_core.documents import Document

store_module = importlib.import_module("05_create_chroma_store")


def retrieve_documents(
    question: str,
    persist_dir: str | Path = store_module.DEFAULT_PERSIST_DIR,
    k: int = 4,
) -> list[Document]:
    """Retrieve the top-k similar chunks for a question."""
    if not question.strip():
        return []

    vector_store = store_module.load_chroma_store(persist_dir)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return list(retriever.invoke(question))


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents as prompt context."""
    return "\n\n".join(document.page_content for document in documents)
