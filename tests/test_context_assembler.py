from typing import Sequence

from graph_policy_rag.application.default_context_assembler import (
    DefaultContextAssembler,
)
from graph_policy_rag.application.context_item import ContextItem
from graph_policy_rag.application.retrieval_hit import RetrievalHit
from graph_policy_rag.application.retrieval_context import RetrievalContext
from graph_policy_rag.domain.chunk import Chunk


class FakeChunkRepository:
    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def upsert(self, chunks: Sequence[Chunk]) -> None: ...


def test_assembles_context_items_from_retrieval_hits():
    chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        text="Sellers must accept returns within fourteen days.",
        source_uri="file://seller_onboarding.md",
        index=0,
        version="version-1",
    )

    assembler = DefaultContextAssembler(
        chunk_repository=FakeChunkRepository(
            chunks={"chunk-1": chunk},
        ),
        max_characters=1_000,
    )

    context = assembler.assemble(
        [
            RetrievalHit(
                chunk_id="chunk-1",
                score=0.9,
            )
        ]
    )

    assert context == RetrievalContext(
        items=(
            ContextItem(
                chunk=chunk,
                score=0.9,
            ),
        )
    )


def test_assemble_deduplicates_hits_by_chunk_id():
    chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        index=0,
        text="Return policy",
        version="version-1",
        source_uri="file://returns.md",
    )

    repository = FakeChunkRepository(
        chunks={
            chunk.id: chunk,
        }
    )

    assembler = DefaultContextAssembler(
        chunk_repository=repository,
        max_characters=1_000,
    )

    context = assembler.assemble(
        [
            RetrievalHit(chunk_id="chunk-1", score=0.9),
            RetrievalHit(chunk_id="chunk-1", score=0.8),
        ]
    )

    assert context.items == (
        ContextItem(
            chunk=chunk,
            score=0.9,
        ),
    )


def test_assemble_respects_character_budget():
    first_chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        index=0,
        text="12345",
        version="version-1",
        source_uri="file://policy.md",
    )
    second_chunk = Chunk(
        id="chunk-2",
        document_id="document-1",
        index=1,
        text="67890",
        version="version-1",
        source_uri="file://policy.md",
    )

    repository = FakeChunkRepository(
        chunks={
            first_chunk.id: first_chunk,
            second_chunk.id: second_chunk,
        }
    )

    assembler = DefaultContextAssembler(
        chunk_repository=repository,
        max_characters=7,
    )

    context = assembler.assemble(
        [
            RetrievalHit(chunk_id=first_chunk.id, score=0.9),
            RetrievalHit(chunk_id=second_chunk.id, score=0.8),
        ]
    )

    assert context.items == (
        ContextItem(
            chunk=first_chunk,
            score=0.9,
        ),
    )
