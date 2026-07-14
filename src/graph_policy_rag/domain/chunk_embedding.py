from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkEmbedding:
    chunk_id: str
    chunk_version: str
    vector: tuple[float, ...]
    model: str
