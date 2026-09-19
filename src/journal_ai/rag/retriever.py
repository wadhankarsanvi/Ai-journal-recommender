from journal_ai.rag.vector_store import JournalVectorStore


class JournalRetriever:
    """Retrieve relevant journal evidence from ChromaDB."""

    def __init__(self):
        self.vector_store = JournalVectorStore()

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict]:

        return self.vector_store.search(
            query=query,
            n_results=n_results,
        )