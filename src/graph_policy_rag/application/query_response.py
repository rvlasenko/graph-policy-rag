from dataclasses import dataclass

from graph_policy_rag.application.citation import Citation
from graph_policy_rag.application.retrieval_context import RetrievalContext


@dataclass(frozen=True)
class QueryResponse:
    context: RetrievalContext
    citations: tuple[Citation, ...]
