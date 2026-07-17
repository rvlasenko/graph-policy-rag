import math

from collections.abc import Sequence

from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding
from graph_policy_rag.vector_store.search_result import VectorSearchResult
from graph_policy_rag.domain.query_embedding import QueryEmbedding


def _cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimensions")

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for zero vectors")

    return dot_product / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(
        self,
        model_name: str,
        vector_dimension: int,
    ) -> None:
        if vector_dimension <= 0:
            raise ValueError("Vector dimension must be greater than 0")

        self._model_name = model_name
        self._vector_dimension = vector_dimension
        self._embeddings: dict[str, ChunkEmbedding] = {}

    def upsert(
        self,
        embeddings: Sequence[ChunkEmbedding],
    ) -> None:
        for embedding in embeddings:
            if embedding.model != self._model_name:
                raise ValueError(
                    f"Embedding model {embedding.model!r} does not match "
                    f"vector store model {self._model_name!r}"
                )

            if len(embedding.vector) != self._vector_dimension:
                raise ValueError(
                    f"Embedding vector dimension {len(embedding.vector)} "
                    f"does not match vector store dimension "
                    f"{self._vector_dimension}"
                )

            if all(value == 0.0 for value in embedding.vector):
                raise ValueError("Embedding vector must not be a zero vector")

        for embedding in embeddings:
            self._embeddings[embedding.chunk_id] = embedding

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int,
    ) -> list[VectorSearchResult]:
        if query_embedding.model != self._model_name:
            raise ValueError("model is not the same")

        if top_k <= 0:
            raise ValueError("top k must be greater than 0")

        results: list[VectorSearchResult] = []

        if len(query_embedding.vector) != self._vector_dimension:
            raise ValueError(
                f"Query vector dimension {len(query_embedding.vector)} "
                f"does not match vector store dimension {self._vector_dimension}"
            )

        if len(query_embedding.vector) != self._vector_dimension:
            raise ValueError(
                f"Embedding vector dimension {len(query_embedding.vector)} "
                f"does not match vector store dimension "
                f"{self._vector_dimension}"
            )

        for embedding in self._embeddings.values():
            score = _cosine_similarity(query_embedding.vector, embedding.vector)

            results.append(
                VectorSearchResult(
                    chunk_id=embedding.chunk_id,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: (-result.score, result.chunk_id),
        )

        return results[:top_k]
