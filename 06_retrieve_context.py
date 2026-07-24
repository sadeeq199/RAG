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