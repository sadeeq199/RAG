"""Create and manage the persistent Chroma vector store.

This module is the single source of truth for the Chroma persistence
directory. Every other module (06_retrieve_context.py, streamlit_app.py)
imports ``DEFAULT_PERSIST_DIR`` or calls ``get_persist_directory()`` from
here instead of hard-coding a path, so the app can never read from one
location while writing to another.
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:  # pragma: no cover - Streamlit is present in the app runtime.
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None  # type: ignore[assignment]

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - python-dotenv is a hard requirement.
    dotenv_values = None  # type: ignore[assignment]

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vectors_module = importlib.import_module("04_vector_representation")

LOGGER = logging.getLogger(__name__)

# تعديل مسار البيانات الافتراضي ليتحول تلقائياً إلى مجلد المؤقت /tmp إذا كان المجلد المحلي غير قابل للكتابة
def _get_default_data_dir() -> Path:
    local_data = Path("data").resolve()
    if local_data.exists() and os.access(local_data, os.W_OK):
        return local_data
    # بديل مؤقت على السيرفرات السحابية مثل Streamlit Cloud
    cloud_data = Path(tempfile.gettempdir()) / "rag_project_data"
    cloud_data.mkdir(parents=True, exist_ok=True)
    return cloud_data

DEFAULT_DATA_DIR = _get_default_data_dir()
COLLECTION_NAME = "rag_documents"
PERSIST_DIR_ENV_VAR = "CHROMA_PERSIST_DIRECTORY"
_CLOUD_TEMP_SUBDIR = "rag_project_chroma_db"

_ENV_FILE_VALUES = dotenv_values(".env") if dotenv_values is not None else {}


def _read_config_value(name: str) -> str | None:
    """Read a config value using the required priority order.

    Priority: 1) st.secrets, 2) .env file, 3) real environment variables.
    """
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value)
        except Exception:  # noqa: BLE001 - secrets.toml may not exist locally.
            pass

    env_file_value = _ENV_FILE_VALUES.get(name)
    if env_file_value:
        return env_file_value

    return os.getenv(name)


def _directory_is_writable(path: Path) -> bool:
    """Return whether `path` can actually be created and written to."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def get_persist_directory() -> Path:
    """Resolve the one persistence directory used everywhere in the app.

    Order of resolution:
      1. An explicit ``CHROMA_PERSIST_DIRECTORY`` override (secrets/.env/env).
      2. ``./chroma_db`` when the current working directory is writable
         (local development).
      3. A directory under the OS temp folder (``tempfile.gettempdir()``)
         when the working directory is read-only.
    """
    override = _read_config_value(PERSIST_DIR_ENV_VAR)
    if override:
        return Path(override)

    local_candidate = Path("chroma_db").resolve()
    if _directory_is_writable(local_candidate):
        return local_candidate

    cloud_fallback = Path(tempfile.gettempdir()) / _CLOUD_TEMP_SUBDIR
    LOGGER.info(
        "%s is not writable; using temp persistence directory %s",
        local_candidate,
        cloud_fallback,
    )
    cloud_fallback.mkdir(parents=True, exist_ok=True)
    return cloud_fallback


DEFAULT_PERSIST_DIR = get_persist_directory()


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