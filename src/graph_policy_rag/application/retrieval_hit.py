from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
