from collections.abc import Sequence

from graph_policy_rag.application.ingestion_service import IngestionService
from graph_policy_rag.chunk_repository.in_memory import InMemoryChunkRepository
from graph_policy_rag.chunking.chunker import TextChunker
from graph_policy_rag.domain.document import Document
from graph_policy_rag.domain.query_embedding import QueryEmbedding
from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore


class FakeEmbeddingModel:
    model_name = "fake-model"
    vector_dimension = 2

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        return [[1.0, float(len(text))] for text in texts]


def _make_document(document_id: str, content: str) -> Document:
    return Document(
        id=document_id,
        title=document_id,
        content=content,
        source_uri=f"policies/{document_id}.md",
        metadata={},
    )


def _build_service() -> tuple[
    IngestionService,
    InMemoryChunkRepository,
    InMemoryVectorStore,
]:
    embedding_model = FakeEmbeddingModel()
    chunk_repository = InMemoryChunkRepository()
    vector_store = InMemoryVectorStore(
        model_name=embedding_model.model_name,
        vector_dimension=embedding_model.vector_dimension,
    )
    service = IngestionService(
        chunker=TextChunker(chunk_size_words=2, overlap_words=0),
        chunk_embedder=ChunkEmbedder(embedding_model),
        chunk_repository=chunk_repository,
        vector_store=vector_store,
    )

    return service, chunk_repository, vector_store


def _indexed_chunk_ids(vector_store: InMemoryVectorStore) -> set[str]:
    results = vector_store.search(
        query_embedding=QueryEmbedding(vector=(1.0, 0.0), model="fake-model"),
        top_k=100,
    )

    return {result.chunk_id for result in results}


def test_ingest_stores_chunks_and_embeddings():
    service, chunk_repository, vector_store = _build_service()

    service.ingest(_make_document("returns", "one two three four"))

    chunks = chunk_repository.list_by_document("returns")
    assert len(chunks) == 2
    assert _indexed_chunk_ids(vector_store) == {chunk.id for chunk in chunks}


def test_reingesting_shorter_document_removes_stale_chunks():
    service, chunk_repository, vector_store = _build_service()

    service.ingest(_make_document("returns", "one two three four five six"))
    service.ingest(_make_document("returns", "one two"))

    chunks = chunk_repository.list_by_document("returns")
    assert [chunk.text for chunk in chunks] == ["one two"]
    assert _indexed_chunk_ids(vector_store) == {chunks[0].id}


def test_reingesting_document_does_not_touch_other_documents():
    service, chunk_repository, vector_store = _build_service()

    service.ingest(_make_document("returns", "one two three four"))
    service.ingest(_make_document("shipping", "five six seven eight"))
    service.ingest(_make_document("returns", "one two"))

    shipping_chunks = chunk_repository.list_by_document("shipping")
    assert len(shipping_chunks) == 2
    assert {chunk.id for chunk in shipping_chunks} <= _indexed_chunk_ids(
        vector_store
    )
