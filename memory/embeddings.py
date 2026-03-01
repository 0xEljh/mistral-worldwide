from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL_NAME = "unsloth/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Lazy sentence-transformer embedding wrapper."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._dimension: int | None = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        model = self._ensure_model()
        self._dimension = int(model.get_sentence_embedding_dimension())
        return self._dimension

    def encode(self, text: str) -> np.ndarray:
        text_value = text.strip()
        if not text_value:
            raise ValueError("text cannot be empty")

        embeddings = self.encode_batch([text_value])
        return embeddings[0]

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts cannot be empty")

        normalized_texts: list[str] = []
        for index, text in enumerate(texts):
            value = text.strip()
            if not value:
                raise ValueError(f"texts[{index}] cannot be empty")
            normalized_texts.append(value)

        model = self._ensure_model()
        embeddings = model.encode(
            normalized_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            return matrix.reshape(1, -1)
        return matrix

    def _ensure_model(self) -> Any:
        model = self._model
        if model is not None:
            return model

        with self._lock:
            model = self._model
            if model is not None:
                return model

            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for semantic memory support. "
                    "Install project dependencies to enable embeddings."
                ) from exc

            model = SentenceTransformer(self._model_name)
            self._model = model
            self._dimension = int(model.get_sentence_embedding_dimension())
            return model
