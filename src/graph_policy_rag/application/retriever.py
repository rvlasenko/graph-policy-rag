from collections.abc import Sequence
from typing import Protocol

from graph_policy_rag.application.retrieval_hit import RetrievalHit


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievalHit]: ...
