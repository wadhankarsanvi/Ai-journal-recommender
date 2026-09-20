import hashlib
import logging
import threading
from typing import Any
from journal_ai.rag.chunker import chunk_text
from journal_ai.rag.vector_store import JournalVectorStore


class RAGPipeline:
    """
    RAG Pipeline that indexes academic journal metadata and recent published papers,
    and retrieves semantically relevant evidence for incoming manuscripts.
    """

    def __init__(self):
        self.vector_store = JournalVectorStore(collection_name="journal_evidence")

    def index_journals(self, journals: list[dict[str, Any]]) -> int:
        """
        Convert journal metadata and recent works into rich searchable documents
        and index them in ChromaDB.
        """
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for journal in journals:
            journal_id = str(journal.get("id", "unknown"))
            journal_name = journal.get("display_name", "Unknown Journal")
            publisher = journal.get("publisher", "Unknown Publisher")
            topics = journal.get("topics") or []
            topic_str = ", ".join(str(t) for t in topics)

            # 1. Index Core Journal Profile
            core_doc = (
                f"Academic Journal: {journal_name}. "
                f"Publisher: {publisher}. "
                f"Aims and Scope Topics: {topic_str}. "
                f"ISSN: {journal.get('issn_l', 'N/A')}. "
                f"Open Access: {journal.get('is_oa', False)}. "
                f"DOAJ Indexed: {journal.get('is_in_doaj', False)}. "
                f"2-Year Citedness Percentile: {journal.get('citedness_2yr_percentile', 'N/A')}."
            )

            chunks = chunk_text(core_doc, chunk_size=300, overlap=40)
            for idx, chunk in enumerate(chunks):
                doc_id = f"{hashlib.md5(journal_id.encode()).hexdigest()[:12]}-meta-{idx}"
                documents.append(chunk)
                ids.append(doc_id)
                metadatas.append({
                    "journal_id": journal_id,
                    "journal": journal_name,
                    "publisher": str(publisher),
                    "source": str(journal.get("source", "OpenAlex")),
                    "type": "journal_scope",
                })

            # 2. Index Recent Published Papers as Ground Truth Evidence
            recent_works = journal.get("recent_works") or []
            for w_idx, work in enumerate(recent_works[:5]):
                title = work.get("title", "")
                doi = work.get("doi", "")
                if title:
                    work_doc = (
                        f"Recently Published in {journal_name}: '{title}'. "
                        f"DOI: {doi}. "
                        f"Keywords/Themes: {topic_str}."
                    )
                    w_id = f"{hashlib.md5(journal_id.encode()).hexdigest()[:12]}-work-{w_idx}"
                    documents.append(work_doc)
                    ids.append(w_id)
                    metadatas.append({
                        "journal_id": journal_id,
                        "journal": journal_name,
                        "doi": doi,
                        "title": title,
                        "type": "recent_publication",
                    })

        if not documents:
            return 0

        existing_ids = self.vector_store.existing_ids(ids)

        new_documents = []
        new_ids = []
        new_metadatas = []

        for document, doc_id, metadata in zip(
            documents,
            ids,
            metadatas,
        ):
            if doc_id not in existing_ids:
                new_documents.append(document)
                new_ids.append(doc_id)
                new_metadatas.append(metadata)

        if not new_documents:
            return 0

        self.vector_store.add_documents(
            documents=new_documents,
            ids=new_ids,
            metadatas=new_metadatas,
        )

        return len(new_documents)

    def retrieve_evidence(self, query: str, n_results: int = 10) -> list[dict[str, Any]]:
        """Retrieve most semantically relevant evidence chunks for the query."""
        return self.vector_store.search(query=query, n_results=n_results)

# ---------------------------------------------------------------------------
# Process-wide singleton + warmup
#
# Building a RAGPipeline is EXPENSIVE: it opens a ChromaDB PersistentClient
# (sqlite) and constructs DefaultEmbeddingFunction, which lazily downloads the
# ~80 MB all-MiniLM-L6-v2 ONNX model and builds an onnxruntime session.
# Doing that per request is the main reason the UI hangs on the first click.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

_pipeline_instance: "RAGPipeline | None" = None
_pipeline_lock = threading.Lock()


def get_rag_pipeline() -> "RAGPipeline":
    """Return the shared RAGPipeline, building it at most once per process."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                _pipeline_instance = RAGPipeline()
    return _pipeline_instance


def warmup_rag() -> bool:
    """
    Force the model download + ONNX session build at boot instead of on the
    user's first click. Safe to call from a background thread.
    """
    try:
        rag = get_rag_pipeline()
        # Touch the embedding function directly so the model is actually loaded
        # (search() short-circuits and returns [] while the collection is empty).
        rag.vector_store.embedding_function(["warmup"])
        logger.info("RAG warmup complete (embedding model loaded).")
        return True
    except Exception as exc:
        logger.warning("RAG warmup failed, will fall back at request time: %s", exc)
        return False