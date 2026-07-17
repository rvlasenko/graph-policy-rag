import pytest

from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding
from graph_policy_rag.domain.query_embedding import QueryEmbedding
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore
from graph_policy_rag.vector_store.search_result import VectorSearchResult


def test_search_returns_most_similar_embedding():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            ),
            ChunkEmbedding(
                chunk_id="chunk-2",
                chunk_version="version-2",
                vector=(0.0, 1.0),
                model="test-model",
            ),
        ]
    )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=1,
    )

    assert results == [
        VectorSearchResult(
            chunk_id="chunk-1",
            score=1.0,
        )
    ]


def test_search_returns_top_k_results_ordered_by_score():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            ),
            ChunkEmbedding(
                chunk_id="chunk-2",
                chunk_version="version-2",
                vector=(1.0, 1.0),
                model="test-model",
            ),
            ChunkEmbedding(
                chunk_id="chunk-3",
                chunk_version="version-3",
                vector=(0.0, 1.0),
                model="test-model",
            ),
        ]
    )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=2,
    )

    assert [result.chunk_id for result in results] == [
        "chunk-1",
        "chunk-2",
    ]


def test_upsert_replaces_embedding_with_same_chunk_id():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            ),
        ]
    )

    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-2",
                vector=(0.0, 1.0),
                model="test-model",
            ),
        ]
    )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(0.0, 1.0),
            model="test-model",
        ),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 1.0


def test_upsert_does_not_modify_store_when_batch_contains_wrong_model():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            )
        ]
    )

    with pytest.raises(ValueError):
        store.upsert(
            [
                ChunkEmbedding(
                    chunk_id="chunk-1",
                    chunk_version="version-2",
                    vector=(0.0, 1.0),
                    model="test-model",
                ),
                ChunkEmbedding(
                    chunk_id="chunk-2",
                    chunk_version="version-1",
                    vector=(1.0, 1.0),
                    model="wrong-model",
                ),
            ]
        )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=10,
    )

    assert results == [
        VectorSearchResult(
            chunk_id="chunk-1",
            score=1.0,
        )
    ]


def test_upsert_does_not_modify_store_when_batch_contains_zero_vector():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            )
        ]
    )

    with pytest.raises(ValueError):
        store.upsert(
            [
                ChunkEmbedding(
                    chunk_id="chunk-1",
                    chunk_version="version-2",
                    vector=(0.0, 1.0),
                    model="test-model",
                ),
                ChunkEmbedding(
                    chunk_id="chunk-2",
                    chunk_version="version-1",
                    vector=(0.0, 0.0),
                    model="test-model",
                ),
            ]
        )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=10,
    )

    assert results == [
        VectorSearchResult(
            chunk_id="chunk-1",
            score=1.0,
        )
    ]


def test_upsert_does_not_modify_store_when_batch_contains_wrong_dimension():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            )
        ]
    )

    with pytest.raises(ValueError):
        store.upsert(
            [
                ChunkEmbedding(
                    chunk_id="chunk-1",
                    chunk_version="version-2",
                    vector=(0.0, 1.0),
                    model="test-model",
                ),
                ChunkEmbedding(
                    chunk_id="chunk-2",
                    chunk_version="version-1",
                    vector=(1.0, 1.0, 1.0),
                    model="test-model",
                ),
            ]
        )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=10,
    )

    assert results == [
        VectorSearchResult(
            chunk_id="chunk-1",
            score=1.0,
        )
    ]


def test_search_rejects_query_with_wrong_dimension():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            )
        ]
    )

    with pytest.raises(ValueError):
        store.search(
            query_embedding=QueryEmbedding(
                vector=(1.0, 0.0, 0.0),
                model="test-model",
            ),
            top_k=1,
        )


def test_search_rejects_zero_query_vector():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-1",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            )
        ]
    )

    with pytest.raises(ValueError):
        store.search(
            query_embedding=QueryEmbedding(
                vector=(0.0, 0.0),
                model="test-model",
            ),
            top_k=1,
        )


def test_search_orders_equal_scores_by_chunk_id():
    store = InMemoryVectorStore(
        model_name="test-model",
        vector_dimension=2,
    )
    store.upsert(
        [
            ChunkEmbedding(
                chunk_id="chunk-b",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            ),
            ChunkEmbedding(
                chunk_id="chunk-a",
                chunk_version="version-1",
                vector=(1.0, 0.0),
                model="test-model",
            ),
        ]
    )

    results = store.search(
        query_embedding=QueryEmbedding(
            vector=(1.0, 0.0),
            model="test-model",
        ),
        top_k=2,
    )

    assert results == [
        VectorSearchResult(chunk_id="chunk-a", score=1.0),
        VectorSearchResult(chunk_id="chunk-b", score=1.0),
    ]


def test_rejects_non_positive_vector_dimension():
    with pytest.raises(ValueError):
        InMemoryVectorStore(
            model_name="test-model",
            vector_dimension=0,
        )
