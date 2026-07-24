"""Streamlit interface for the RAG Document Assistant."""

from __future__ import annotations

import importlib
import logging
from html import escape
from pathlib import Path

import streamlit as st

documents_module = importlib.import_module("01_documents")
store_module = importlib.import_module("05_create_chroma_store")
retrieval_module = importlib.import_module("06_retrieve_context")
prompting_module = importlib.import_module("07_prompting")

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
LOGGER = logging.getLogger(__name__)


def configure_page() -> None:
    """Configure Streamlit page settings and base styling."""
    st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1080px;}
        div[data-testid="stSidebar"] {border-right: 1px solid #e6e8eb;}
        .source-card {
            border: 1px solid #e6e8eb;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            background: #ffffff;
        }
        .source-meta {font-size: 0.9rem; color: #4b5563; margin-bottom: 0.35rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_directories() -> None:
    """Create local persistence directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def has_database() -> bool:
    """Return whether the Chroma directory appears initialized."""
    return store_module.is_chroma_initialized(CHROMA_DIR)


def save_uploads(uploaded_files: list) -> list[Path]:
    """Save Streamlit uploads into the data directory."""
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target = DATA_DIR / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def build_database(rebuild: bool = False) -> bool:
    """Build the Chroma database and report errors in the UI."""
    try:
        with st.spinner("Indexing documents..."):
            store_module.create_chroma_store(DATA_DIR, CHROMA_DIR, rebuild=rebuild)
        st.success("Database indexed successfully.")
        return True
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:  # noqa: BLE001 - user-facing app boundary.
        LOGGER.exception("Database build failed: %s", exc)
        st.error(f"Database build failed: {exc}")
    return False


def render_sidebar() -> None:
    """Render database status controls."""
    st.sidebar.header("Database Status")
    source_files = documents_module.discover_documents(DATA_DIR)
    chunk_count = store_module.get_indexed_chunk_count(CHROMA_DIR)
    status = "Ready" if chunk_count > 0 else "Not indexed"

    st.sidebar.metric("Status", status)
    st.sidebar.metric("Source Files", len(source_files))
    st.sidebar.metric("Indexed Chunks", chunk_count)

    if st.sidebar.button("Rebuild Database", use_container_width=True):
        build_database(rebuild=True)


def render_sources(retrieved_documents: list) -> None:
    """Display retrieved source chunks."""
    st.subheader("Retrieved Sources")
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.metadata
        filename = metadata.get("filename") or Path(str(metadata.get("source", ""))).name
        page = metadata.get("page")
        page_number = int(page) + 1 if isinstance(page, int) else "N/A"
        preview = document.page_content[:500].strip()
        if len(document.page_content) > 500:
            preview = f"{preview}..."
        safe_filename = escape(str(filename))
        safe_preview = escape(preview)

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-meta"><strong>Source {index}</strong> ·
                Filename: {safe_filename} · Page Number: {page_number}</div>
                <div>{safe_preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Streamlit app."""
    logging.basicConfig(level=logging.INFO)
    configure_page()
    ensure_directories()

    st.title("📚 RAG Document Assistant")
    render_sidebar()

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXTs",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        saved_paths = save_uploads(uploaded_files)
        st.success(f"Saved {len(saved_paths)} file(s) to data/.")
        build_database(rebuild=True)
    elif documents_module.discover_documents(DATA_DIR) and not has_database():
        st.info("Documents found. Building the database automatically.")
        build_database(rebuild=False)

    question = st.text_area("Question", placeholder="Ask a question about your documents...")
    ask_clicked = st.button("Ask", type="primary", use_container_width=False)

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a question first.")
            return
        if not has_database():
            st.warning("No database found. Upload documents or rebuild the database first.")
            return

        try:
            with st.spinner("Retrieving context..."):
                retrieved_documents = retrieval_module.retrieve_documents(question, CHROMA_DIR)
            if not retrieved_documents:
                st.warning("No relevant context was retrieved.")
                return

            context = retrieval_module.format_context(retrieved_documents)
            with st.spinner("Generating answer..."):
                answer = prompting_module.answer_question(context, question)

            st.subheader("Answer")
            st.write(answer)
            render_sources(retrieved_documents)
        except ValueError as exc:
            st.error(str(exc))
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - keep UI graceful.
            LOGGER.exception("Question answering failed: %s", exc)
            st.error(f"Could not answer the question: {exc}")


if __name__ == "__main__":
    main()
