from collections.abc import Sequence
from typing import Protocol


class EmbeddingModel(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def vector_dimension(self) -> int: ...

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]: ...
