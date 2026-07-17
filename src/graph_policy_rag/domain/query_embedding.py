from dataclasses import dataclass


@dataclass(frozen=True)
class QueryEmbedding:
    vector: tuple[float, ...]
    model: str
