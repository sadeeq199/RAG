"""Text preprocessing helpers."""

from __future__ import annotations

import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Normalize text while preserving useful document content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    non_empty_lines = [line for line in lines if line]
    return "\n".join(non_empty_lines)


def preprocess_documents(documents: list[Document]) -> list[Document]:
    """Apply text cleaning to LangChain Document objects."""
    cleaned_documents: list[Document] = []
    for document in documents:
        cleaned_content = clean_text(document.page_content)
        if cleaned_content:
            cleaned_documents.append(
                Document(page_content=cleaned_content, metadata=dict(document.metadata))
            )
    return cleaned_documents
