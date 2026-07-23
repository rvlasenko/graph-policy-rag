from dataclasses import dataclass

from graph_policy_rag.application.context_item import ContextItem


@dataclass(frozen=True)
class RetrievalContext:
    items: tuple[ContextItem, ...]
