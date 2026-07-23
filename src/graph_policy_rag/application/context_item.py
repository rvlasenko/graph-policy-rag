from dataclasses import dataclass

from graph_policy_rag.domain.chunk import Chunk


@dataclass(frozen=True)
class ContextItem:
    chunk: Chunk
    score: float
