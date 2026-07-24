"""Create and manage the persistent Chroma vector store."""

from __future__ import annotations

import importlib
import logging
import shutil
from pathlib import Path
import os
import tempfile

from langchain_chroma import Chroma
from langchain_core.documents import Document

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vectors_module = importlib.import_module("04_vector_representation")

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path("data")
if os.getenv("STREAMLIT_RUNTIME") or os.getenv("STREAMLIT_SERVER_PORT"):
    DEFAULT_PERSIST_DIR = Path(tempfile.gettempdir()) / "chroma_db"
else:
    DEFAULT_PERSIST_DIR = Path("chroma_db")COLLECTION_NAME = "rag_documents"


def is_chroma_initialized(persist_dir: str | Path = DEFAULT_PERSIST_DIR) -> bool:
    """Return whether a Chroma persistence directory contains a database."""
    persist_path = Path(persist_dir)
    return persist_path.exists() and (persist_path / "chroma.sqlite3").exists()


def load_prepare_chunks(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Document]:
    """Load, clean, and chunk source documents."""
    documents = documents_module.load_documents(data_dir)
    cleaned_documents = preprocessing_module.preprocess_documents(documents)
    return chunking_module.chunk_documents(cleaned_documents)


def create_chroma_store(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    rebuild: bool = False,
) -> Chroma:
    """Create or rebuild a persistent Chroma store from local documents."""
    persist_path = Path(persist_dir)
    if rebuild and persist_path.exists():
        shutil.rmtree(persist_path)
        LOGGER.info("Removed existing Chroma database at %s", persist_path)

    chunks = load_prepare_chunks(data_dir)
    if not chunks:
        raise ValueError("No valid PDF or TXT content found in the data directory.")

    embeddings = vectors_module.get_embeddings()
    persist_path.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=COLLECTION_NAME,
    )
    LOGGER.info("Persisted %s chunks to %s", len(chunks), persist_path)
    return vector_store


def load_chroma_store(persist_dir: str | Path = DEFAULT_PERSIST_DIR) -> Chroma:
    """Load an existing persistent Chroma store."""
    persist_path = Path(persist_dir)
    if not is_chroma_initialized(persist_path):
        raise FileNotFoundError("Chroma database does not exist yet.")

    embeddings = vectors_module.get_embeddings()
    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def get_indexed_chunk_count(persist_dir: str | Path = DEFAULT_PERSIST_DIR) -> int:
    """Return the number of chunks stored in Chroma."""
    try:
        store = load_chroma_store(persist_dir)
        return int(store._collection.count())  # noqa: SLF001 - Chroma exposes count here.
    except Exception as exc:  # noqa: BLE001 - status helper should never crash UI.
        LOGGER.warning("Could not read Chroma chunk count: %s", exc)
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_chroma_store(rebuild=True)
