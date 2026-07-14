import pytest

from graph_policy_rag.domain.document import Document
from graph_policy_rag.chunking.chunker import TextChunker


def test_chunks_document_with_overlap():
    document = Document(
        id="doc-1",
        title="Seller onboarding",
        content="one two three four five six",
        source_uri="file://seller_onboarding.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    chunks = chunker.chunk(document)

    assert [chunk.text for chunk in chunks] == [
        "one two three",
        "three four five",
        "five six",
    ]


def test_returns_empty_list_for_empty_document():
    document = Document(
        id="doc-1",
        title="Seller onboarding",
        content="      ",
        source_uri="file://seller_onboarding.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    chunks = chunker.chunk(document)

    assert chunks == []


def test_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size_words=0,
            overlap_words=1,
        )


@pytest.mark.parametrize("overlap_words", [-1, 3])
def test_rejects_invalid_overlap(overlap_words):
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size_words=3,
            overlap_words=overlap_words,
        )


def test_sets_chunk_indexes():
    document = Document(
        id="doc-1",
        title="Seller onboarding",
        content="one two three four five six",
        source_uri="file://seller_onboarding.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    chunks = chunker.chunk(document)

    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_chunk_has_same_document_id():
    document = Document(
        id="doc-1",
        title="Seller onboarding",
        content="one two three four five six",
        source_uri="file://seller_onboarding.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    chunks = chunker.chunk(document)

    assert all(chunk.document_id == document.id for chunk in chunks)


def test_chunk_id_is_stable():
    document = Document(
        id="doc-1",
        title="Seller onboarding",
        content="one two three four five six",
        source_uri="file://seller_onboarding.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    first_chunks = chunker.chunk(document)
    second_chunks = chunker.chunk(document)

    assert [chunk.id for chunk in first_chunks] == [chunk.id for chunk in second_chunks]


def test_changes_chunk_version_when_text_changes():
    first_document = Document(
        id="doc-1",
        title="Policy",
        content="one two three",
        source_uri="file://policy.md",
        metadata={},
    )
    second_document = Document(
        id="doc-1",
        title="Policy",
        content="one changed three",
        source_uri="file://policy.md",
        metadata={},
    )

    chunker = TextChunker(chunk_size_words=3, overlap_words=1)

    first_chunks = chunker.chunk(first_document)
    second_chunks = chunker.chunk(second_document)

    assert first_chunks[0].id == second_chunks[0].id
    assert first_chunks[0].version != second_chunks[0].version
