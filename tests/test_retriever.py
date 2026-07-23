from collections.abc import Sequence

from graph_policy_rag.application.retrieval_hit import RetrievalHit
from graph_policy_rag.retrieval.vector_store_retriever import VectorStoreRetriever
from graph_policy_rag.vector_store.search_result import VectorSearchResult
from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding
from graph_policy_rag.domain.query_embedding import QueryEmbedding


class FakeEmbeddingModel:
    model_name = "fake-model"
    vector_dimension = 2

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        self.received_texts = list(texts)
        return [[0.1, 0.2]]


class FakeVectorStore:
    def __init__(self) -> None:
        self.received_query_embedding = None
        self.received_top_k = None

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int,
    ) -> list[VectorSearchResult]:
        self.received_query_embedding = query_embedding
        self.received_top_k = top_k

        return [
            VectorSearchResult(
                chunk_id="chunk-1",
                score=0.9,
            )
        ]

    def upsert(
        self,
        embeddings: Sequence[ChunkEmbedding],
    ) -> None: ...


def test_retrieves_hits_from_vector_store():
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()

    retriever = VectorStoreRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    results = retriever.retrieve(
        query="return policy",
        top_k=3,
    )

    assert embedding_model.received_texts == ["return policy"]
    assert vector_store.received_top_k == 3
    assert results == [
        RetrievalHit(
            chunk_id="chunk-1",
            score=0.9,
        )
    ]

    assert vector_store.received_query_embedding == QueryEmbedding(
        vector=(0.1, 0.2),
        model="fake-model",
    )
