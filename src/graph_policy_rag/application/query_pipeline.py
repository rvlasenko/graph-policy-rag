from graph_policy_rag.application.context_assembler import ContextAssembler
from graph_policy_rag.application.query_request import QueryRequest
from graph_policy_rag.application.retriever import Retriever
from graph_policy_rag.application.citation import Citation
from graph_policy_rag.application.query_response import QueryResponse
from graph_policy_rag.application.errors import EmptyRetrievalError


class QueryPipeline:
    def __init__(
        self,
        retriever: Retriever,
        context_assembler: ContextAssembler,
    ) -> None:
        self._retriever = retriever
        self._context_assembler = context_assembler

    def run(self, request: QueryRequest) -> QueryResponse:
        hits = self._retriever.retrieve(
            query=request.text,
            top_k=request.top_k,
        )

        context = self._context_assembler.assemble(hits)

        if not context.items:
            raise EmptyRetrievalError("No relevant context found")

        citations = tuple(
            Citation(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                source_uri=item.chunk.source_uri,
            )
            for item in context.items
        )

        return QueryResponse(
            context=context,
            citations=citations,
        )
