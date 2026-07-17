from collections.abc import Sequence
from typing import Protocol

from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding
from graph_policy_rag.vector_store.search_result import VectorSearchResult
from graph_policy_rag.domain.query_embedding import QueryEmbedding


class VectorStore(Protocol):
    def upsert(
        self,
        embeddings: Sequence[ChunkEmbedding],
    ) -> None: ...

    def search(
        self,
        query_embedding: QueryEmbedding,
        top_k: int,
    ) -> list[VectorSearchResult]: ...
