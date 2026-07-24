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

The Chroma database is stored in `chroma_db/` at runtime and can be rebuilt from uploaded or bundled documents.

## Secrets Configuration

For Streamlit Cloud, add these secrets:

```toml
OPENROUTER_API_KEY="your_key"
OPENROUTER_MODEL="openai/gpt-4o-mini"
```

For local development, you can use environment variables:

```bash
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=openai/gpt-4o-mini
```

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
