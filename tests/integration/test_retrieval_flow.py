from collections.abc import Sequence

from graph_policy_rag.domain.document import Document
from graph_policy_rag.domain.query_embedding import QueryEmbedding
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore
from graph_policy_rag.chunking.chunker import TextChunker
from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.vector_store.search_result import VectorSearchResult


class FakeEmbeddingModel:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.received_texts: list[str] = []
        self.call_count = 0

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        self.call_count += 1
        self.received_texts = list(texts)

        return [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]


def test_indexes_document_chunks_and_finds_relevant_chunk():
    document = Document(
        id="policy-1",
        title="Seller returns policy",
        content=(
            "Sellers must accept returns within fourteen days. "
            "Damaged products must be reported with photographic evidence."
        ),
        source_uri="policies/returns.md",
        metadata={},
        version="version-1",
        updated_at=None,
    )

    chunker = TextChunker(
        chunk_size_words=7,
        overlap_words=0,
    )
    chunks = chunker.chunk(document)

    embedding_model = FakeEmbeddingModel()
    embedder = ChunkEmbedder(embedding_model)
    embeddings = embedder.embed(chunks)

    store = InMemoryVectorStore(
        model_name=embedding_model.model_name,
        vector_dimension=2,
    )
    store.upsert(embeddings)

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model=embedding_model.model_name,
        ),
        top_k=1,
    )

    assert results == [
        VectorSearchResult(
            chunk_id=chunks[0].id,
            score=1.0,
        )
    ]
    assert embedding_model.received_texts == [chunk.text for chunk in chunks]
    assert embedding_model.call_count == 1
