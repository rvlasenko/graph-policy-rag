import pytest

from collections.abc import Sequence

from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.domain.chunk import Chunk
from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding


class FakeEmbeddingModel:
    model_name = "fake-model"

    def __init__(self) -> None:
        self.received_texts: list[str] = []
        self.call_count = 0

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        self.call_count += 1
        self.received_texts = list(texts)

        return [
            [0.1, 0.2],
            [0.3, 0.4],
        ]


def test_passes_chunk_texts_to_model():
    model = FakeEmbeddingModel()
    embedder = ChunkEmbedder(model=model)

    chunks = [
        Chunk(
            id="1",
            document_id="doc-1",
            text="first chunk",
            source_uri="file://seller_onboarding.md",
            index=0,
            version="hash",
        ),
        Chunk(
            id="2",
            document_id="doc-1",
            text="second chunk",
            source_uri="file://seller_onboarding.md",
            index=1,
            version="hash",
        ),
    ]

    embedder.embed(chunks)

    assert model.received_texts == [
        "first chunk",
        "second chunk",
    ]


def test_creates_embeddings_for_chunks():
    model = FakeEmbeddingModel()
    embedder = ChunkEmbedder(model=model)

    chunks = [
        Chunk(
            id="1",
            document_id="doc-1",
            text="first chunk",
            source_uri="file://seller_onboarding.md",
            index=0,
            version="hash 1",
        ),
        Chunk(
            id="2",
            document_id="doc-1",
            text="second chunk",
            source_uri="file://seller_onboarding.md",
            index=1,
            version="hash 2",
        ),
    ]

    embeddings = embedder.embed(chunks)

    assert embeddings == [
        ChunkEmbedding(
            chunk_id="1", chunk_version="hash 1", vector=(0.1, 0.2), model="fake-model"
        ),
        ChunkEmbedding(
            chunk_id="2", chunk_version="hash 2", vector=(0.3, 0.4), model="fake-model"
        ),
    ]


def test_returns_empty_list_without_calling_model():
    model = FakeEmbeddingModel()
    embedder = ChunkEmbedder(model=model)

    embeddings = embedder.embed([])

    assert embeddings == []
    assert model.call_count == 0


def test_raises_when_model_returns_wrong_number_of_vectors():
    class IncompleteEmbeddingModel:
        model_name = "incomplete-model"

        def embed_texts(
            self,
            texts: Sequence[str],
        ) -> Sequence[Sequence[float]]:
            return [[0.1, 0.2]]

    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="first chunk",
            source_uri="file://policy.md",
            index=0,
            version="version-1",
        ),
        Chunk(
            id="chunk-2",
            document_id="doc-1",
            text="second chunk",
            source_uri="file://policy.md",
            index=1,
            version="version-2",
        ),
    ]

    embedder = ChunkEmbedder(model=IncompleteEmbeddingModel())

    with pytest.raises(ValueError):
        embedder.embed(chunks)
