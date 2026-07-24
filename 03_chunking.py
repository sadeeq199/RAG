"""Document chunking utilities."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the configured recursive character splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into retrieval-sized chunks."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks
