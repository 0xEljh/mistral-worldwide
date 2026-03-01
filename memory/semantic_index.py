from __future__ import annotations

from importlib import import_module
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from memory.embeddings import EmbeddingModel


def _load_faiss_module() -> Any:
    try:
        return import_module("faiss")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "faiss-cpu is required for semantic memory support. "
            "Install project dependencies to enable vector search."
        ) from exc


@dataclass(frozen=True)
class SemanticIndexEntry:
    track_id: int
    object_type: str
    description: str
    indexed_world_version: int
    described_at: float | None = None


@dataclass(frozen=True)
class SemanticSearchResult:
    entry: SemanticIndexEntry
    similarity: float


class SemanticIndex:
    """Thread-safe FAISS index over entity descriptions."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self._embedding_model = embedding_model
        self._faiss = _load_faiss_module()
        self._index: Any | None = None
        self._entries: list[SemanticIndexEntry] = []
        self._index_by_track_id: dict[int, int] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._index_by_track_id

    def entries_snapshot(self) -> list[SemanticIndexEntry]:
        with self._lock:
            return list(self._entries)

    def add(
        self,
        *,
        track_id: int,
        object_type: str,
        description: str,
        indexed_world_version: int,
        described_at: float | None = None,
    ) -> bool:
        normalized_description = description.strip()
        if not normalized_description:
            return False

        with self._lock:
            if track_id in self._index_by_track_id:
                return False

        embedding = self._embedding_model.encode(normalized_description)
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        with self._lock:
            if track_id in self._index_by_track_id:
                return False

            self._ensure_index(dimension=vector.shape[1])
            index = self._index
            if index is None:
                raise RuntimeError("semantic index failed to initialize")
            index.add(vector)

            entry = SemanticIndexEntry(
                track_id=track_id,
                object_type=object_type,
                description=normalized_description,
                indexed_world_version=indexed_world_version,
                described_at=described_at,
            )
            self._entries.append(entry)
            self._index_by_track_id[track_id] = len(self._entries) - 1
            return True

    def search(self, query: str, top_k: int = 5) -> list[SemanticSearchResult]:
        query_text = query.strip()
        if not query_text or top_k <= 0:
            return []

        with self._lock:
            if self._index is None or not self._entries:
                return []

        query_vector = self._embedding_model.encode(query_text)
        query_matrix = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        with self._lock:
            if self._index is None or not self._entries:
                return []

            search_k = min(top_k, len(self._entries))
            distances, indices = self._index.search(query_matrix, search_k)
            results: list[SemanticSearchResult] = []

            for score, index in zip(distances[0], indices[0], strict=False):
                entry_index = int(index)
                if entry_index < 0 or entry_index >= len(self._entries):
                    continue

                results.append(
                    SemanticSearchResult(
                        entry=self._entries[entry_index],
                        similarity=float(score),
                    )
                )

            return results

    def _ensure_index(self, *, dimension: int) -> None:
        index = self._index
        if index is None:
            self._index = self._faiss.IndexFlatIP(dimension)
            return

        if int(index.d) != dimension:
            raise RuntimeError(
                "Embedding dimension changed unexpectedly for semantic index: "
                f"existing={index.d} new={dimension}"
            )
