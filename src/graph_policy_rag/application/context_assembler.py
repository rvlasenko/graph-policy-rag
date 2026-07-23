from collections.abc import Sequence
from typing import Protocol

from graph_policy_rag.application.retrieval_context import RetrievalContext
from graph_policy_rag.application.retrieval_hit import RetrievalHit


class ContextAssembler(Protocol):
    def assemble(
        self,
        hits: Sequence[RetrievalHit],
    ) -> RetrievalContext: ...
