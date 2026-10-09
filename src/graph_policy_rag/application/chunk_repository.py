from collections.abc import Sequence
from typing import Protocol

from graph_policy_rag.domain.chunk import Chunk


class ChunkRepository(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    def get(self, chunk_id: str) -> Chunk | None: ...

    def list_by_document(self, document_id: str) -> list[Chunk]: ...

    def delete(self, chunk_ids: Sequence[str]) -> None: ...
