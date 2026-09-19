from pathlib import Path
from typing import Any
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from journal_ai.config.settings import settings


class JournalVectorStore:
    """Persistent ChromaDB vector store for academic journal evidence and RAG retrieval."""

    def __init__(self, collection_name: str = "journal_evidence"):
        path = Path(settings.vectorstore_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(path))
        self.embedding_function = DefaultEmbeddingFunction()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "Academic Journal Scopes and Publication Evidence"},
        )

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert documents with metadata into vector store."""
        if not documents:
            return

        # Sanitize metadata values to primitive types supported by Chroma
        clean_metadatas = []
        for meta in metadatas:
            clean = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
                elif isinstance(v, list):
                    clean[k] = ", ".join(str(x) for x in v[:5])
                else:
                    clean[k] = str(v)
            clean_metadatas.append(clean)

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=clean_metadatas,
        )

    def search(
        self,
        query: str,
        n_results: int = 8,
    ) -> list[dict[str, Any]]:
        """Retrieve most semantically similar journal documents."""
        if not query.strip():
            return []

        doc_count = self.collection.count()
        if doc_count == 0:
            return []

        limit = min(n_results, doc_count)
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved.append({
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist),
            })

        return retrieved