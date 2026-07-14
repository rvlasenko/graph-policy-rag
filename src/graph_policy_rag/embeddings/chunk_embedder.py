from collections.abc import Sequence

from graph_policy_rag.domain.chunk import Chunk
from graph_policy_rag.domain.chunk_embedding import ChunkEmbedding
from graph_policy_rag.embeddings.model import EmbeddingModel


class ChunkEmbedder:
    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    def embed(
        self,
        chunks: Sequence[Chunk],
    ) -> list[ChunkEmbedding]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        vectors = self._model.embed_texts(texts=texts)

        embeddings: list[ChunkEmbedding] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            embeddings.append(
                ChunkEmbedding(
                    chunk_id=chunk.id,
                    chunk_version=chunk.version,
                    vector=tuple(vector),
                    model=self._model.model_name,
                )
            )

        return embeddings
