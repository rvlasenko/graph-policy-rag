from graph_policy_rag.chunk_repository.in_memory import InMemoryChunkRepository
from graph_policy_rag.domain.chunk import Chunk


def test_upsert_stores_chunk_by_id():
    repository = InMemoryChunkRepository()

    chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        text="Return policy",
        source_uri="file://seller_onboarding.md",
        index=0,
        version="version-1",
    )

    repository.upsert([chunk])

    assert repository.get("chunk-1") == chunk


def test_upsert_replaces_chunk_with_same_id():
    repository = InMemoryChunkRepository()

    old_chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        text="Old policy",
        source_uri="file://seller_onboarding.md",
        index=0,
        version="version-1",
    )

    new_chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        text="Updated policy",
        source_uri="file://seller_onboarding.md",
        index=0,
        version="version-2",
    )

    repository.upsert([old_chunk])
    repository.upsert([new_chunk])

    assert repository.get("chunk-1") == new_chunk
