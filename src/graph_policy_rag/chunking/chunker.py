from graph_policy_rag.domain.document import Document
from graph_policy_rag.domain.chunk import Chunk
from graph_policy_rag.common.hashing import make_hash


class TextChunker:
    def __init__(self, chunk_size_words: int, overlap_words: int) -> None:
        if chunk_size_words <= 0:
            raise ValueError("Chunk size must be greater than 0")

        if not (0 <= overlap_words < chunk_size_words):
            raise ValueError("Overlap must not be greater than chunk size")

        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk(self, document: Document) -> list[Chunk]:
        words = document.content.split()

        if not words:
            return []

        step = self.chunk_size_words - self.overlap_words
        start = 0
        chunks: list[Chunk] = []

        for index, start in enumerate(range(0, len(words), step)):
            end = start + self.chunk_size_words
            chunk_words = words[start:end]
            chunk_id = make_hash(document.id + ":" + str(index))

            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    text=" ".join(chunk_words),
                    source_uri=document.source_uri,
                    index=index,
                )
            )

        return chunks
