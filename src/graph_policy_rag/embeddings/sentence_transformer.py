from collections.abc import Sequence

from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(list(texts))

        return [[float(value) for value in vector] for vector in vectors]
