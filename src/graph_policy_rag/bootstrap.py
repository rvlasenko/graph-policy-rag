from graph_policy_rag.application.default_context_assembler import (
    DefaultContextAssembler,
)
from graph_policy_rag.application.query_pipeline import QueryPipeline
from graph_policy_rag.chunk_repository.in_memory import InMemoryChunkRepository
from graph_policy_rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from graph_policy_rag.retrieval.vector_store_retriever import VectorStoreRetriever
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore


def build_query_pipeline() -> QueryPipeline:
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
    )

    vector_store = InMemoryVectorStore(
        model_name=embedding_model.model_name,
        vector_dimension=embedding_model.vector_dimension,
    )

    chunk_repository = InMemoryChunkRepository()

    retriever = VectorStoreRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    context_assembler = DefaultContextAssembler(
        chunk_repository=chunk_repository,
        max_characters=5_000,
    )

    return QueryPipeline(
        retriever=retriever,
        context_assembler=context_assembler,
    )
