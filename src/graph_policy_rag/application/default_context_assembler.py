from collections.abc import Sequence

from graph_policy_rag.application.chunk_repository import ChunkRepository
from graph_policy_rag.application.context_item import ContextItem
from graph_policy_rag.application.retrieval_context import RetrievalContext
from graph_policy_rag.application.retrieval_hit import RetrievalHit


class DefaultContextAssembler:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
        max_characters: int,
    ) -> None:
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")

        self._chunk_repository = chunk_repository
        self._max_characters = max_characters

    def assemble(
        self,
        hits: Sequence[RetrievalHit],
    ) -> RetrievalContext:
        items: list[ContextItem] = []
        seen_chunk_ids: set[str] = set()
        total_characters = 0

        for hit in hits:
            if hit.chunk_id in seen_chunk_ids:
                continue

            chunk = self._chunk_repository.get(hit.chunk_id)

            if chunk is None:
                continue

            chunk_size = len(chunk.text)

            if total_characters + chunk_size > self._max_characters:
                continue

            seen_chunk_ids.add(hit.chunk_id)
            total_characters += chunk_size

            items.append(
                ContextItem(
                    chunk=chunk,
                    score=hit.score,
                )
            )

        return RetrievalContext(items=tuple(items))
