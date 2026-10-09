from collections.abc import Sequence

from graph_policy_rag.domain.chunk import Chunk


class InMemoryChunkRepository:
    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def list_by_document(self, document_id: str) -> list[Chunk]:
        return [
            chunk
            for chunk in self._chunks.values()
            if chunk.document_id == document_id
        ]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)
