"""Document loading utilities for the RAG application."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

LOGGER = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def discover_documents(data_dir: str | Path = "data") -> list[Path]:
    """Return all supported document paths inside the data directory."""
    root = Path(data_dir)
    if not root.exists():
        LOGGER.info("Data directory does not exist yet: %s", root)
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_documents(data_dir: str | Path = "data") -> list[Document]:
    """Load PDF and TXT files from data_dir as LangChain Document objects."""
    documents: list[Document] = []

    for path in discover_documents(data_dir):
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)

            loaded = loader.load()
            for document in loaded:
                document.metadata["source"] = str(path)
                document.metadata["filename"] = path.name
            documents.extend(loaded)
            LOGGER.info("Loaded %s document page(s) from %s", len(loaded), path.name)
        except Exception as exc:  # noqa: BLE001 - log and continue with other files.
            LOGGER.exception("Failed to load %s: %s", path, exc)

    return documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loaded_documents = load_documents()
    print(f"Loaded {len(loaded_documents)} document(s).")
