def retrieve_documents(
    question: str,
    persist_dir: str | Path | None = None,
    k: int = 4,
) -> list[Document]:
    """Retrieve the top-k similar chunks for a question."""

    if not question.strip():
        return []

    if persist_dir is None:
        persist_dir = store_module.DEFAULT_PERSIST_DIR

    vector_store = store_module.load_chroma_store(persist_dir)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    return list(retriever.invoke(question))