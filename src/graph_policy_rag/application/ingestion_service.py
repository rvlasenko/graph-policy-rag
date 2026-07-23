from graph_policy_rag.application.chunk_repository import ChunkRepository
from graph_policy_rag.chunking.chunker import TextChunker
from graph_policy_rag.domain.document import Document
from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore


class IngestionService:
    def __init__(
        self,
        chunker: TextChunker,
        chunk_embedder: ChunkEmbedder,
        chunk_repository: ChunkRepository,
        vector_store: InMemoryVectorStore,
    ) -> None:
        self._chunker = chunker
        self._chunk_embedder = chunk_embedder
        self._chunk_repository = chunk_repository
        self._vector_store = vector_store

    def ingest(self, document: Document) -> None:
        chunks = self._chunker.chunk(document)
        embeddings = self._chunk_embedder.embed(chunks)

        self._chunk_repository.upsert(chunks)
        self._vector_store.upsert(embeddings)
