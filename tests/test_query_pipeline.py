from collections.abc import Sequence

import pytest

from graph_policy_rag.application.citation import Citation
from graph_policy_rag.application.context_item import ContextItem
from graph_policy_rag.application.errors import EmptyRetrievalError
from graph_policy_rag.application.query_pipeline import QueryPipeline
from graph_policy_rag.application.query_request import QueryRequest
from graph_policy_rag.application.query_response import QueryResponse
from graph_policy_rag.application.retrieval_context import RetrievalContext
from graph_policy_rag.application.retrieval_hit import RetrievalHit
from graph_policy_rag.domain.chunk import Chunk


class FakeRetriever:
    def __init__(
        self,
        hits: Sequence[RetrievalHit],
    ) -> None:
        self._hits = hits
        self.received_query: str | None = None
        self.received_top_k: int | None = None

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> Sequence[RetrievalHit]:
        self.received_query = query
        self.received_top_k = top_k
        return self._hits


class FakeContextAssembler:
    def __init__(self, context: RetrievalContext) -> None:
        self._context = context
        self.received_hits: Sequence[RetrievalHit] | None = None

    def assemble(
        self,
        hits: Sequence[RetrievalHit],
    ) -> RetrievalContext:
        self.received_hits = hits
        return self._context


def test_runs_retrieval_for_query_request():
    expected_hits = [
        RetrievalHit(
            chunk_id="chunk-1",
            score=0.9,
        )
    ]
    chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        index=0,
        text="Return policy",
        version="version-1",
        source_uri="file://returns.md",
    )

    retriever = FakeRetriever(hits=expected_hits)

    expected_context = RetrievalContext(
        items=(
            ContextItem(
                chunk=chunk,
                score=0.9,
            ),
        )
    )
    context_assembler = FakeContextAssembler(expected_context)

    pipeline = QueryPipeline(
        retriever=retriever,
        context_assembler=context_assembler,
    )

    result = pipeline.run(
        QueryRequest(
            text="return policy",
            top_k=3,
        )
    )

    assert retriever.received_query == "return policy"
    assert retriever.received_top_k == 3
    assert context_assembler.received_hits == expected_hits

    assert result == QueryResponse(
        context=expected_context,
        citations=(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_uri=chunk.source_uri,
            ),
        ),
    )


def test_raises_when_retrieval_context_is_empty():
    retriever = FakeRetriever(hits=[])
    context_assembler = FakeContextAssembler(RetrievalContext(items=()))

    pipeline = QueryPipeline(
        retriever=retriever,
        context_assembler=context_assembler,
    )

    with pytest.raises(EmptyRetrievalError):
        pipeline.run(
            QueryRequest(
                text="unknown policy",
                top_k=3,
            )
        )
