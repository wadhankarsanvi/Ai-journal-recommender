from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class EmbeddingModel:
    """Embedding model using Chroma's lightweight default embedding backend."""

    def __init__(self):
        self.embedding_function = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.embedding_function(texts)

        return [list(embedding) for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.embedding_function([text])

        return list(embedding[0])