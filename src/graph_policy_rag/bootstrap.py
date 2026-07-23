from dataclasses import dataclass

from graph_policy_rag.application.default_context_assembler import (
    DefaultContextAssembler,
)
from graph_policy_rag.application.ingestion_service import IngestionService
from graph_policy_rag.application.query_pipeline import QueryPipeline
from graph_policy_rag.chunk_repository.in_memory import (
    InMemoryChunkRepository,
)
from graph_policy_rag.chunking.chunker import TextChunker
from graph_policy_rag.embeddings.chunk_embedder import ChunkEmbedder
from graph_policy_rag.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)
from graph_policy_rag.retrieval.vector_store_retriever import (
    VectorStoreRetriever,
)
from graph_policy_rag.vector_store.in_memory import InMemoryVectorStore


@dataclass(frozen=True)
class ApplicationContainer:
    query_pipeline: QueryPipeline
    ingestion_service: IngestionService
    chunk_repository: InMemoryChunkRepository
    vector_store: InMemoryVectorStore


def build_application_container() -> ApplicationContainer:
    embedding_model = SentenceTransformerEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
    )

    vector_store = InMemoryVectorStore(
        model_name=embedding_model.model_name,
        vector_dimension=embedding_model.vector_dimension,
    )

    chunk_repository = InMemoryChunkRepository()

    chunk_embedder = ChunkEmbedder(
        model=embedding_model,
    )

    chunker = TextChunker(
        chunk_size_words=500,
        overlap_words=50,
    )

    retriever = VectorStoreRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    context_assembler = DefaultContextAssembler(
        chunk_repository=chunk_repository,
        max_characters=5_000,
    )

    query_pipeline = QueryPipeline(
        retriever=retriever,
        context_assembler=context_assembler,
    )

    ingestion_service = IngestionService(
        chunker=chunker,
        chunk_embedder=chunk_embedder,
        chunk_repository=chunk_repository,
        vector_store=vector_store,
    )

    return ApplicationContainer(
        query_pipeline=query_pipeline,
        chunk_repository=chunk_repository,
        vector_store=vector_store,
        ingestion_service=ingestion_service,
    )
