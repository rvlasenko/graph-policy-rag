from graph_policy_rag.application.chunk_repository import ChunkRepository
from graph_policy_rag.chunking.chunker import TextChunker
from graph_policy_rag.domain.document import Document
from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.vector_store.store import VectorStore


class IngestionService:
    def __init__(
        self,
        chunker: TextChunker,
        chunk_embedder: ChunkEmbedder,
        chunk_repository: ChunkRepository,
        vector_store: VectorStore,
    ) -> None:
        self._chunker = chunker
        self._chunk_embedder = chunk_embedder
        self._chunk_repository = chunk_repository
        self._vector_store = vector_store

    def ingest(self, document: Document) -> None:
        chunks = self._chunker.chunk(document)
        embeddings = self._chunk_embedder.embed(chunks)

        current_chunk_ids = {chunk.id for chunk in chunks}
        stale_chunk_ids = [
            chunk.id
            for chunk in self._chunk_repository.list_by_document(document.id)
            if chunk.id not in current_chunk_ids
        ]

        self._chunk_repository.upsert(chunks)
        self._vector_store.upsert(embeddings)

        self._vector_store.delete(stale_chunk_ids)
        self._chunk_repository.delete(stale_chunk_ids)
