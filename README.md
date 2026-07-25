# RAG Document Assistant

A production-ready Retrieval-Augmented Generation application built with Python, LangChain, ChromaDB, SentenceTransformers, OpenRouter, and Streamlit.

## Project Overview

RAG Document Assistant lets users upload PDF and TXT files, indexes their content into a persistent Chroma vector database, retrieves the most relevant chunks for a question, and answers only from the retrieved context.

## Features

- PDF and TXT ingestion from `data/`
- Text cleaning, chunking, embeddings, and persistent Chroma indexing
- SentenceTransformers embeddings using `all-MiniLM-L6-v2`
- Similarity retrieval with `k=4`
- OpenRouter chat completions with strict context-only prompting
- Streamlit Cloud compatible secrets handling
- Source display with filename, page number, and chunk preview
- Graceful handling for missing files, missing database, API issues, bad documents, and indexing errors

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Create your local environment variables or configure Streamlit secrets, then run:

```bash
streamlit run streamlit_app.py
```

You can either upload files in the Streamlit interface or place `.pdf` and `.txt` files directly inside `data/`. The app automatically builds the vector database when documents are available.

## Deployment

This project is ready for Streamlit Cloud. Push the project to a repository, create a Streamlit Cloud app, and set the main file to:

```text
streamlit_app.py
```

### Persistence directory

The Chroma persistence directory is resolved once, in `05_create_chroma_store.py`,
and reused everywhere else (`06_retrieve_context.py`, `streamlit_app.py`) — there
is a single source of truth, never a hard-coded path duplicated across files.

Resolution order:

1. An explicit `CHROMA_PERSIST_DIRECTORY` override (via Streamlit secrets, `.env`, or an environment variable).
2. `./chroma_db` when the working directory is writable (local development).
3. A folder under `tempfile.gettempdir()` when the working directory is **not**
   writable. On Streamlit Community Cloud the cloned repository checkout is
   read-only, which is the real cause of `attempt to write a readonly database`
   errors — the app detects this automatically (by testing writability, not by
   guessing the platform) and never attempts to write inside the cloned repo.

Because the fallback directory lives under the OS temp folder, the index is
rebuilt from `data/` (or from uploads) each time the app starts on Streamlit
Cloud, since temp storage isn't guaranteed to persist across restarts.

## Secrets Configuration

For Streamlit Cloud, add these secrets:

```toml
OPENROUTER_API_KEY="your_key"
OPENROUTER_MODEL="openai/gpt-4o-mini"
# Optional override; otherwise resolved automatically (see Persistence directory above).
CHROMA_PERSIST_DIRECTORY=""
```

For local development, you can use environment variables or a `.env` file
(loaded automatically via `python-dotenv`):

```bash
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

Configuration values are read with this priority: `st.secrets` first, then
`.env`, then real environment variables.

## Folder Structure

```text
RAG_Project/
├── data/
├── chroma_db/
├── 01_documents.py
├── 02_preprocessing.py
├── 03_chunking.py
├── 04_vector_representation.py
├── 05_create_chroma_store.py
├── 06_retrieve_context.py
├── 07_prompting.py
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## Screenshots Placeholder

Add screenshots of the Streamlit upload view, answer view, and retrieved sources after deployment.
