"""Streamlit interface for the RAG Document Assistant."""

from __future__ import annotations

import importlib
import logging
import time
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables for local development
load_dotenv()

documents_module = importlib.import_module("01_documents")
store_module = importlib.import_module("05_create_chroma_store")
retrieval_module = importlib.import_module("06_retrieve_context")
prompting_module = importlib.import_module("07_prompting")

# Single source of truth for dynamic writable directories
DATA_DIR = store_module.DEFAULT_DATA_DIR
CHROMA_DIR = store_module.DEFAULT_PERSIST_DIR
LOGGER = logging.getLogger(__name__)


def inject_custom_css() -> None:
    """Inject modern Dark Theme CSS styled after ChatGPT, NotebookLM & Perplexity AI."""
    st.markdown(
        """
        <style>
        /* Import Modern Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Global Theme Override */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
        }

        /* Hide Sidebar Completely */
        section[data-testid="stSidebar"] {
            display: none !important;
        }

        /* Container Alignment & Max Width */
        .block-container {
            max-width: 1100px !important;
            padding-top: 2rem !important;
            padding-bottom: 4rem !important;
            margin: 0 auto !important;
        }

        /* Page Header */
        .app-header {
            text-align: center;
            padding: 1.5rem 1rem 2rem 1rem;
            margin-bottom: 2rem;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0) 100%);
            border-bottom: 1px solid #334155;
            border-radius: 16px;
        }
        .app-header-title {
            color: #F8FAFC;
            font-size: 2.25rem;
            font-weight: 800;
            margin: 0 0 0.5rem 0;
            letter-spacing: -0.025em;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        .app-header-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            margin: 0;
            font-weight: 500;
        }

        /* Metrics Horizontal Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.25rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: #475569;
            transform: translateY(-2px);
        }
        .metric-label {
            color: #94A3B8;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .metric-value {
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 700;
        }
        .status-badge-ready {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-badge-notready {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 600;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: #1E293B;
            padding: 0.5rem;
            border-radius: 16px;
            border: 1px solid #334155;
            margin-bottom: 1.75rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 12px;
            color: #94A3B8;
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0 1.5rem;
            border: none !important;
            transition: all 0.2s ease;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        /* Content Card Containers */
        .content-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }
        .content-card-title {
            color: #F8FAFC;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Upload Area */
        div[data-testid="stFileUploader"] {
            background: #1E293B !important;
            border: 2px dashed #334155 !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            transition: border-color 0.2s ease, background-color 0.2s ease !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #2563EB !important;
            background-color: rgba(37, 99, 235, 0.05) !important;
        }
        div[data-testid="stFileUploader"] section {
            background: transparent !important;
        }
        div[data-testid="stFileUploader"] label {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }

        /* File Chips */
        .file-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: #0F172A;
            border: 1px solid #334155;
            padding: 0.5rem 0.85rem;
            border-radius: 10px;
            font-size: 0.875rem;
            color: #F8FAFC;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* Text Area / Inputs */
        .stTextArea textarea {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 14px !important;
            font-size: 1rem !important;
            padding: 1rem !important;
        }
        .stTextArea textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25) !important;
        }

        /* Primary Buttons */
        div.stButton > button[kind="primary"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.75rem 1.75rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1D4ED8 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
        }

        /* Secondary Buttons */
        div.stButton > button[kind="secondary"] {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.25rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: #2563EB !important;
            color: #2563EB !important;
        }

        /* AI Answer Card */
        .answer-card {
            background: #1E293B;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.75rem;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }
        .answer-card-header {
            color: #F8FAFC;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            border-bottom: 1px solid #334155;
            padding-bottom: 0.75rem;
        }

        /* Source Card */
        .source-card {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .source-meta {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        /* Chat History Bubbles */
        .chat-user-message {
            background: #2563EB;
            color: #FFFFFF;
            padding: 1rem 1.25rem;
            border-radius: 16px 16px 4px 16px;
            margin-bottom: 1rem;
            max-width: 85%;
            margin-left: auto;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
            font-weight: 500;
        }
        .chat-assistant-message {
            background: #1E293B;
            border: 1px solid #334155;
            color: #F8FAFC;
            padding: 1.5rem;
            border-radius: 16px 16px 16px 4px;
            margin-bottom: 1.25rem;
            max-width: 95%;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        }

        /* Progress Steps Bar */
        .progress-step {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: #1E293B;
            border: 1px solid #334155;
            padding: 0.85rem 1.25rem;
            border-radius: 12px;
            color: #60A5FA;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Custom Notification Banners */
        .alert-box {
            padding: 1rem 1.25rem;
            border-radius: 14px;
            margin-bottom: 1.25rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .alert-success { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); color: #10B981; }
        .alert-warning { background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.3); color: #F59E0B; }
        .alert-error { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); color: #EF4444; }
        .alert-info { background: rgba(37, 99, 235, 0.12); border: 1px solid rgba(37, 99, 235, 0.3); color: #60A5FA; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    """Configure Streamlit page settings and inject styling."""
    st.set_page_config(
        page_title="RAG Document Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()


def ensure_directories() -> None:
    """Create local persistence directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def has_database() -> bool:
    """Return whether the Chroma directory appears initialized."""
    return store_module.is_chroma_initialized(CHROMA_DIR)


def save_uploads(uploaded_files: list) -> list[Path]:
    """Save Streamlit uploads into the writable data directory."""
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        target = DATA_DIR / Path(uploaded_file.name).name
        target.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(target)
    return saved_paths


def build_database(rebuild: bool = False) -> bool:
    """Build the Chroma database with step-by-step progress feedback."""
    status_placeholder = st.empty()
    try:
        status_placeholder.markdown(
            '<div class="progress-step">⚙️ <strong>Building vector database & embeddings...</strong></div>',
            unsafe_allow_html=True,
        )
        store_module.create_chroma_store(DATA_DIR, CHROMA_DIR, rebuild=rebuild)
        status_placeholder.markdown(
            '<div class="alert-box alert-success">✅ Database indexed successfully. Ready to answer questions!</div>',
            unsafe_allow_html=True,
        )
        return True
    except ValueError as exc:
        status_placeholder.markdown(
            f'<div class="alert-box alert-warning">⚠️ {escape(str(exc))}</div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001 - user-facing app boundary.
        LOGGER.exception("Database build failed: %s", exc)
        status_placeholder.markdown(
            f'<div class="alert-box alert-error">❌ Database build failed: {escape(str(exc))}</div>',
            unsafe_allow_html=True,
        )
    return False


def format_file_size(size_in_bytes: int) -> str:
    """Format size in bytes into human-readable text."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    if size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    return f"{size_in_bytes / (1024 * 1024):.1f} MB"


def render_page_header() -> None:
    """Render the top app header."""
    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-header-title">🤖 📚 RAG Document Assistant</h1>
            <p class="app-header-subtitle">Ask questions about your uploaded documents using AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_metrics() -> None:
    """Render top metrics cards horizontally."""
    source_files = documents_module.discover_documents(DATA_DIR)
    chunk_count = store_module.get_indexed_chunk_count(CHROMA_DIR)
    is_ready = chunk_count > 0

    status_html = (
        '<span class="status-badge-ready">🟢 Ready</span>'
        if is_ready
        else '<span class="status-badge-notready">🟡 Not Ready</span>'
    )

    st.markdown(
        f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">📄 Documents</div>
                <div class="metric-value">{len(source_files)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🧩 Indexed Chunks</div>
                <div class="metric-value">{chunk_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">🟢 Database Status</div>
                <div style="margin-top: 0.25rem;">{status_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources_list(retrieved_documents: list) -> None:
    """Display retrieved source chunks inside collapsible expanders."""
    st.markdown('<div style="font-weight: 700; color: #94A3B8; margin-top: 1rem; margin-bottom: 0.5rem;">📚 Retrieved Context Sources</div>', unsafe_allow_html=True)
    for index, document in enumerate(retrieved_documents, start=1):
        metadata = document.metadata
        filename = metadata.get("filename") or Path(str(metadata.get("source", ""))).name
        page = metadata.get("page")
        page_number = int(page) + 1 if isinstance(page, int) else "N/A"
        preview = document.page_content[:400].strip()
        if len(document.page_content) > 400:
            preview = f"{preview}..."

        with st.expander(f"📄 Source {index}: {filename} (Page {page_number})"):
            st.markdown(f"**Filename:** `{filename}` | **Page:** `{page_number}`")
            st.markdown(f"```text\n{preview}\n```")


def main() -> None:
    """Run the redesigned RAG Streamlit application."""
    logging.basicConfig(level=logging.INFO)
    configure_page()
    ensure_directories()

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_processed_uploads" not in st.session_state:
        st.session_state.last_processed_uploads = []

    render_page_header()
    render_top_metrics()

    # 3 Main Navigation Tabs
    tab_chat, tab_docs, tab_db = st.tabs(["💬 Chat", "📄 Documents", "📊 Database"])

    # =========================================================
    # TAB 1: 💬 CHAT
    # =========================================================
    with tab_chat:
        st.markdown('<div class="content-card-title">📤 Upload Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Drag & Drop your PDFs here or click to browse",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to query with AI.",
        )

        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            if current_file_names != st.session_state.last_processed_uploads:
                saved_paths = save_uploads(uploaded_files)
                st.markdown(
                    f'<div class="alert-box alert-success">📥 Saved {len(saved_paths)} file(s) to temporary directory.</div>',
                    unsafe_allow_html=True,
                )
                build_database(rebuild=False)
                st.session_state.last_processed_uploads = current_file_names
                st.rerun()

        # Display uploaded file chips if documents exist
        existing_docs = documents_module.discover_documents(DATA_DIR)
        if existing_docs:
            chips_html = "".join(
                f'<span class="file-chip">📄 {escape(doc.name)} ({format_file_size(doc.stat().st_size)})</span>'
                for doc in existing_docs[:6]
            )
            if len(existing_docs) > 6:
                chips_html += f'<span class="file-chip">+{len(existing_docs) - 6} more</span>'
            st.markdown(f'<div style="margin-bottom: 1.5rem;">{chips_html}</div>', unsafe_allow_html=True)
        elif existing_docs and not has_database():
            st.markdown(
                '<div class="alert-box alert-info">ℹ️ Documents found. Building database automatically...</div>',
                unsafe_allow_html=True,
            )
            build_database(rebuild=False)

        # Question Input Box
        st.markdown('<div class="content-card-title">💬 Ask AI Assistant</div>', unsafe_allow_html=True)
        question = st.text_area(
            "Question",
            placeholder="Ask anything about your documents...",
            height=110,
            label_visibility="collapsed",
        )

        col_ask, col_clear = st.columns([1, 4])
        with col_ask:
            ask_clicked = st.button("✨ Ask AI", type="primary", use_container_width=True)
        with col_clear:
            if st.session_state.chat_history:
                if st.button("🗑️ Clear History", type="secondary"):
                    st.session_state.chat_history = []
                    st.rerun()

        # Handle Question Answering
        if ask_clicked:
            if not question.strip():
                st.markdown(
                    '<div class="alert-box alert-warning">⚠️ Please enter a question first.</div>',
                    unsafe_allow_html=True,
                )
            elif not has_database():
                st.markdown(
                    '<div class="alert-box alert-warning">⚠️ No database found. Please upload documents first.</div>',
                    unsafe_allow_html=True,
                )
            else:
                progress_container = st.empty()
                try:
                    progress_container.markdown(
                        '<div class="progress-step">🔍 <strong>Retrieving context from ChromaDB...</strong></div>',
                        unsafe_allow_html=True,
                    )
                    retrieved_documents = retrieval_module.retrieve_documents(question, CHROMA_DIR)

                    if not retrieved_documents:
                        progress_container.markdown(
                            '<div class="alert-box alert-warning">⚠️ No relevant context was retrieved for your question.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        progress_container.markdown(
                            '<div class="progress-step">🤖 <strong>Generating answer via OpenRouter AI...</strong></div>',
                            unsafe_allow_html=True,
                        )
                        context = retrieval_module.format_context(retrieved_documents)
                        answer = prompting_module.answer_question(context, question)
                        progress_container.empty()

                        # Append to history
                        st.session_state.chat_history.insert(
                            0,
                            {
                                "question": question,
                                "answer": answer,
                                "sources": retrieved_documents,
                            },
                        )
                        st.rerun()

                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Question answering failed: %s", exc)
                    progress_container.markdown(
                        f'<div class="alert-box alert-error">❌ Could not answer question: {escape(str(exc))}</div>',
                        unsafe_allow_html=True,
                    )

        # Render Conversation History (ChatGPT / NotebookLM style)
        if st.session_state.chat_history:
            st.markdown('<div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC; margin-top: 2rem; margin-bottom: 1rem;">💬 Conversation History</div>', unsafe_allow_html=True)
            for item in st.session_state.chat_history:
                # User Question Bubble
                st.markdown(
                    f'<div class="chat-user-message">👤 <strong>You:</strong> {escape(item["question"])}</div>',
                    unsafe_allow_html=True,
                )
                # Assistant Answer Card
                with st.container():
                    st.markdown(
                        '<div class="answer-card-header">🤖 <strong>AI Answer</strong></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(item["answer"])
                    if item.get("sources"):
                        render_sources_list(item["sources"])
                    st.markdown("<hr style='border-color: #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # =========================================================
    # TAB 2: 📄 DOCUMENTS
    # =========================================================
    with tab_docs:
        st.markdown('<div class="content-card-title">📄 Document Library</div>', unsafe_allow_html=True)
        discovered_docs = documents_module.discover_documents(DATA_DIR)

        if not discovered_docs:
            st.markdown(
                """
                <div class="content-card" style="text-align: center; padding: 3rem 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">No Documents Uploaded</div>
                    <div style="color: #94A3B8;">Upload PDF or TXT files in the Chat tab to start exploring.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for doc in discovered_docs:
                file_stat = doc.stat()
                mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_stat.st_mtime))
                size_str = format_file_size(file_stat.st_size)
                ext = doc.suffix.upper().replace(".", "")

                st.markdown(
                    f"""
                    <div class="content-card" style="padding: 1.25rem 1.5rem; margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.25rem;">
                                    📄 {escape(doc.name)}
                                </div>
                                <div style="font-size: 0.875rem; color: #94A3B8;">
                                    Size: <strong>{size_str}</strong> &bull; Modified: <strong>{mod_time}</strong>
                                </div>
                            </div>
                            <div>
                                <span style="background: rgba(37, 99, 235, 0.2); color: #60A5FA; padding: 0.35rem 0.75rem; border-radius: 8px; font-weight: 700; font-size: 0.85rem; border: 1px solid rgba(37, 99, 235, 0.4);">
                                    {ext}
                                </span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # =========================================================
    # TAB 3: 📊 DATABASE
    # =========================================================
    with tab_db:
        st.markdown('<div class="content-card-title">📊 Vector Database Architecture & Status</div>', unsafe_allow_html=True)

        chunk_count = store_module.get_indexed_chunk_count(CHROMA_DIR)
        status_text = "Ready (Indexed)" if chunk_count > 0 else "Not Indexed"

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(
                f"""
                <div class="content-card">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Database Status</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #10B981; margin-top: 0.5rem;">{status_text}</div>
                    <hr style="border-color: #334155; margin: 1rem 0;">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Vector Store</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-top: 0.25rem;">ChromaDB Persistent Store</div>
                    <hr style="border-color: #334155; margin: 1rem 0;">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Embedding Model</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #60A5FA; margin-top: 0.25rem;">all-MiniLM-L6-v2</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_d2:
            st.markdown(
                f"""
                <div class="content-card">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Chunk Size</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-top: 0.5rem;">800 characters</div>
                    <hr style="border-color: #334155; margin: 1rem 0;">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Chunk Overlap</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; margin-top: 0.25rem;">150 characters</div>
                    <hr style="border-color: #334155; margin: 1rem 0;">
                    <div style="font-size: 0.9rem; color: #94A3B8; font-weight: 600; text-transform: uppercase;">Total Indexed Chunks</div>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #60A5FA; margin-top: 0.25rem;">{chunk_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="content-card-title">🔄 Index Management</div>', unsafe_allow_html=True)
        if st.button("🔄 Rebuild Database Index", type="primary"):
            build_database(rebuild=True)
            st.rerun()


if __name__ == "__main__":
    main()