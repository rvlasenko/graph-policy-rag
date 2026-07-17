from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRequest:
    text: str
    top_k: int = 5

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Query text must not be empty")

        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0")
