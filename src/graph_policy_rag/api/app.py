from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from graph_policy_rag.api.dependencies import (
    get_application_container,
    get_query_pipeline,
)
from graph_policy_rag.api.schemas import QueryApiRequest
from graph_policy_rag.application.errors import EmptyRetrievalError
from graph_policy_rag.application.query_pipeline import QueryPipeline
from graph_policy_rag.application.query_request import QueryRequest
from graph_policy_rag.application.query_response import QueryResponse
from graph_policy_rag.demo_data import build_demo_document


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    container = get_application_container()

    container.ingestion_service.ingest(build_demo_document())

    yield


app = FastAPI(
    title="Graph Policy RAG",
    lifespan=lifespan,
)


@app.post("/query")
def query(
    request: QueryApiRequest,
    pipeline: Annotated[QueryPipeline, Depends(get_query_pipeline)],
) -> QueryResponse:
    try:
        return pipeline.run(
            QueryRequest(
                text=request.text,
                top_k=request.top_k,
            )
        )
    except EmptyRetrievalError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error
