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


def test_list_by_document_returns_only_chunks_of_that_document():
    repository = InMemoryChunkRepository()

    returns_chunk = Chunk(
        id="chunk-1",
        document_id="returns",
        text="Return policy",
        source_uri="file://returns.md",
        index=0,
        version="version-1",
    )
    shipping_chunk = Chunk(
        id="chunk-2",
        document_id="shipping",
        text="Shipping policy",
        source_uri="file://shipping.md",
        index=0,
        version="version-1",
    )

    repository.upsert([returns_chunk, shipping_chunk])

    assert repository.list_by_document("returns") == [returns_chunk]


def test_delete_removes_chunks_and_ignores_unknown_ids():
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
    repository.delete(["chunk-1", "missing-chunk"])

    assert repository.get("chunk-1") is None
