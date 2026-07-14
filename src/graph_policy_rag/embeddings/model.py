from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    model_name: str

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]: ...
