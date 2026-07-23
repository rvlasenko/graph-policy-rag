from collections.abc import Sequence

from graph_policy_rag.application.retrieval_hit import RetrievalHit
from graph_policy_rag.domain.query_embedding import QueryEmbedding
from graph_policy_rag.embeddings.model import EmbeddingModel
from graph_policy_rag.vector_store.store import VectorStore


class VectorStoreRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_model = embedding_model
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        vectors = self._embedding_model.embed_texts([query])
        query_vector = vectors[0]

        search_results = self._vector_store.search(
            query_embedding=QueryEmbedding(
                vector=tuple(query_vector),
                model=self._embedding_model.model_name,
            ),
            top_k=top_k,
        )

        return [
            RetrievalHit(
                chunk_id=result.chunk_id,
                score=result.score,
            )
            for result in search_results
        ]
