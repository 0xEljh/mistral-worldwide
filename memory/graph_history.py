from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random
import threading
from typing import Any, Mapping

_EVICTION_WEIGHT_EPSILON = 1e-6


@dataclass(frozen=True)
class GraphHistorySnapshot:
    world_version: int
    timestamp: int
    graph: dict[str, Any]
    visible_track_ids: frozenset[int]
    visible_entity_count: int
    description_length_rms: float
    visible_entity_time_in_frame_rms: float
    static_score_component: float


class GraphHistoryStore:
    """Stores sparse world graph snapshots with score-aware retention."""

    def __init__(
        self,
        *,
        save_interval_world_versions: int = 200,
        max_snapshots: int = 50,
        protected_recent_snapshots: int | None = None,
        age_decay_frames: float = 5000.0,
        reference_visible_entity_count: float = 10.0,
        reference_description_length: float = 80.0,
        reference_time_in_frame: float = 1000.0,
        weight_age: float = 0.3,
        weight_visible_entities: float = 0.2,
        weight_description_length: float = 0.25,
        weight_time_in_frame: float = 0.25,
        random_seed: int | None = None,
    ) -> None:
        if save_interval_world_versions <= 0:
            raise ValueError("save_interval_world_versions must be > 0")
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be > 0")

        if protected_recent_snapshots is None:
            protected_recent_snapshots = min(5, max_snapshots - 1)

        if protected_recent_snapshots < 0:
            raise ValueError("protected_recent_snapshots must be >= 0")
        if protected_recent_snapshots >= max_snapshots:
            raise ValueError("protected_recent_snapshots must be < max_snapshots")
        if age_decay_frames <= 0:
            raise ValueError("age_decay_frames must be > 0")
        if reference_visible_entity_count <= 0:
            raise ValueError("reference_visible_entity_count must be > 0")
        if reference_description_length <= 0:
            raise ValueError("reference_description_length must be > 0")
        if reference_time_in_frame <= 0:
            raise ValueError("reference_time_in_frame must be > 0")

        weights = (
            weight_age,
            weight_visible_entities,
            weight_description_length,
            weight_time_in_frame,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("score weights must be >= 0")

        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError("at least one score weight must be > 0")

        self._save_interval_world_versions = save_interval_world_versions
        self._max_snapshots = max_snapshots
        self._protected_recent_snapshots = protected_recent_snapshots

        self._age_decay_frames = float(age_decay_frames)
        self._reference_visible_entity_count = float(reference_visible_entity_count)
        self._reference_description_length = float(reference_description_length)
        self._reference_time_in_frame = float(reference_time_in_frame)

        self._weight_age = float(weight_age)
        self._weight_visible_entities = float(weight_visible_entities)
        self._weight_description_length = float(weight_description_length)
        self._weight_time_in_frame = float(weight_time_in_frame)
        self._weight_sum = float(weight_sum)

        self._history: list[GraphHistorySnapshot] = []
        self._last_saved_world_version = -1
        self._rng = random.Random(random_seed)
        self._lock = threading.Lock()

    @property
    def save_interval_world_versions(self) -> int:
        return self._save_interval_world_versions

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)

    def maybe_save(
        self,
        snapshot: Mapping[str, Any],
        *,
        text_by_track_id: Mapping[int, str] | None = None,
    ) -> bool:
        try:
            world_version = int(snapshot.get("world_version", 0))
        except (TypeError, ValueError):
            return False
        if world_version <= 0:
            return False
        if world_version % self._save_interval_world_versions != 0:
            return False

        with self._lock:
            if world_version == self._last_saved_world_version:
                return False

            graph_copy = deepcopy(dict(snapshot))
            try:
                timestamp = int(graph_copy.get("timestamp", -1))
            except (TypeError, ValueError):
                timestamp = -1

            visible_track_ids = self._extract_visible_track_ids(graph_copy)
            pruned_graph = self._prune_graph(graph_copy, visible_track_ids)

            description_length_rms = self._description_length_rms(
                visible_track_ids,
                text_by_track_id,
            )
            visible_entity_time_in_frame_rms = self._visible_entity_time_in_frame_rms(
                pruned_graph.get("objects"),
                visible_track_ids,
            )
            static_score_component = self._static_score_component(
                visible_entity_count=len(visible_track_ids),
                description_length_rms=description_length_rms,
                visible_entity_time_in_frame_rms=visible_entity_time_in_frame_rms,
            )

            history_snapshot = GraphHistorySnapshot(
                world_version=world_version,
                timestamp=timestamp,
                graph=pruned_graph,
                visible_track_ids=visible_track_ids,
                visible_entity_count=len(visible_track_ids),
                description_length_rms=description_length_rms,
                visible_entity_time_in_frame_rms=visible_entity_time_in_frame_rms,
                static_score_component=static_score_component,
            )

            self._append_snapshot_locked(
                history_snapshot,
                current_timestamp=timestamp,
                current_world_version=world_version,
            )
            self._last_saved_world_version = world_version
            return True

    def latest_snapshot_for_track(self, track_id: int) -> GraphHistorySnapshot | None:
        with self._lock:
            for snapshot in reversed(self._history):
                if track_id in snapshot.visible_track_ids:
                    return snapshot
        return None

    def snapshots(self) -> list[GraphHistorySnapshot]:
        with self._lock:
            return list(self._history)

    def _append_snapshot_locked(
        self,
        snapshot: GraphHistorySnapshot,
        *,
        current_timestamp: int,
        current_world_version: int,
    ) -> None:
        while len(self._history) >= self._max_snapshots:
            self._evict_one_snapshot_locked(
                current_timestamp=current_timestamp,
                current_world_version=current_world_version,
            )
        self._history.append(snapshot)

    def _evict_one_snapshot_locked(
        self,
        *,
        current_timestamp: int,
        current_world_version: int,
    ) -> None:
        history_size = len(self._history)
        if history_size == 0:
            return

        candidate_end = history_size - self._protected_recent_snapshots
        if candidate_end <= 0:
            candidate_indices = list(range(history_size))
        else:
            candidate_indices = list(range(candidate_end))

        eviction_weights = [
            self._eviction_weight(
                self._score_snapshot(
                    self._history[index],
                    current_timestamp=current_timestamp,
                    current_world_version=current_world_version,
                )
            )
            for index in candidate_indices
        ]

        evicted_index = self._rng.choices(
            candidate_indices,
            weights=eviction_weights,
            k=1,
        )[0]
        del self._history[evicted_index]

    @staticmethod
    def _eviction_weight(score: float) -> float:
        bounded_score = max(score, 0.0)
        return 1.0 / (bounded_score + _EVICTION_WEIGHT_EPSILON)

    def _score_snapshot(
        self,
        snapshot: GraphHistorySnapshot,
        *,
        current_timestamp: int,
        current_world_version: int,
    ) -> float:
        age_delta = self._age_delta(
            snapshot,
            current_timestamp=current_timestamp,
            current_world_version=current_world_version,
        )
        age_score = math.exp(-age_delta / self._age_decay_frames)

        weighted_sum = (self._weight_age * age_score) + snapshot.static_score_component
        return weighted_sum / self._weight_sum

    @staticmethod
    def _age_delta(
        snapshot: GraphHistorySnapshot,
        *,
        current_timestamp: int,
        current_world_version: int,
    ) -> float:
        current_reference = (
            current_timestamp if current_timestamp >= 0 else current_world_version
        )
        snapshot_reference = (
            snapshot.timestamp if snapshot.timestamp >= 0 else snapshot.world_version
        )
        return float(max(current_reference - snapshot_reference, 0))

    def _static_score_component(
        self,
        *,
        visible_entity_count: int,
        description_length_rms: float,
        visible_entity_time_in_frame_rms: float,
    ) -> float:
        visible_entity_count_score = self._normalize_ratio(
            float(visible_entity_count),
            self._reference_visible_entity_count,
        )
        description_length_score = self._normalize_ratio(
            description_length_rms,
            self._reference_description_length,
        )
        visible_entity_time_in_frame_score = self._normalize_ratio(
            visible_entity_time_in_frame_rms,
            self._reference_time_in_frame,
        )

        return (
            self._weight_visible_entities * visible_entity_count_score
            + self._weight_description_length * description_length_score
            + self._weight_time_in_frame * visible_entity_time_in_frame_score
        )

    @staticmethod
    def _normalize_ratio(value: float, reference_value: float) -> float:
        if reference_value <= 0:
            return 0.0
        return min(max(value, 0.0) / reference_value, 1.0)

    @staticmethod
    def _description_length_rms(
        visible_track_ids: frozenset[int],
        text_by_track_id: Mapping[int, str] | None,
    ) -> float:
        if not visible_track_ids:
            return 0.0

        squared_length_sum = 0.0
        described_count = 0
        for track_id in visible_track_ids:
            description = text_by_track_id.get(track_id) if text_by_track_id else ""
            description_length = (
                float(len(description)) if isinstance(description, str) else 0.0
            )

            squared_length_sum += description_length * description_length
            described_count += 1

        if described_count == 0:
            return 0.0

        return math.sqrt(squared_length_sum / described_count)

    @classmethod
    def _visible_entity_time_in_frame_rms(
        cls,
        objects: Any,
        visible_track_ids: frozenset[int],
    ) -> float:
        if not visible_track_ids or not isinstance(objects, list):
            return 0.0

        squared_age_sum = 0.0
        measured_count = 0
        for item in objects:
            if not isinstance(item, Mapping):
                continue

            track_id = cls._coerce_int(item.get("id"))
            if track_id is None or track_id not in visible_track_ids:
                continue

            first_seen = cls._coerce_int(item.get("first_seen"))
            last_seen = cls._coerce_int(item.get("last_seen"))
            if first_seen is None or last_seen is None:
                continue

            visible_age = max(last_seen - first_seen, 0)
            squared_age_sum += float(visible_age * visible_age)
            measured_count += 1

        if measured_count == 0:
            return 0.0

        return math.sqrt(squared_age_sum / measured_count)

    @classmethod
    def _prune_graph(
        cls,
        graph: dict[str, Any],
        visible_track_ids: frozenset[int],
    ) -> dict[str, Any]:
        objects = graph.get("objects")
        if not isinstance(objects, list):
            graph["objects"] = []
            if isinstance(graph.get("relations"), list):
                graph["relations"] = []
            return graph

        indexed_objects: list[tuple[int, Mapping[str, Any]]] = []
        object_track_ids: set[int] = set()

        for item in objects:
            if not isinstance(item, Mapping):
                continue
            track_id = cls._coerce_int(item.get("id"))
            if track_id is None:
                continue

            indexed_objects.append((track_id, item))
            object_track_ids.add(track_id)

        relations = graph.get("relations")
        kept_relations: list[Mapping[str, Any]] = []
        related_track_ids: set[int] = set()

        if isinstance(relations, list):
            for relation in relations:
                if not isinstance(relation, Mapping):
                    continue

                subject_track_id = cls._track_id_from_relation_entity(
                    relation.get("subject")
                )
                object_track_id = cls._track_id_from_relation_entity(
                    relation.get("object")
                )
                if subject_track_id is None or object_track_id is None:
                    continue
                if (
                    subject_track_id not in object_track_ids
                    or object_track_id not in object_track_ids
                ):
                    continue

                if (
                    subject_track_id in visible_track_ids
                    or object_track_id in visible_track_ids
                ):
                    kept_relations.append(relation)
                    related_track_ids.add(subject_track_id)
                    related_track_ids.add(object_track_id)

        keep_track_ids = set(visible_track_ids)
        keep_track_ids.update(related_track_ids)

        graph["objects"] = [
            item for track_id, item in indexed_objects if track_id in keep_track_ids
        ]

        if isinstance(relations, list):
            graph["relations"] = kept_relations

        return graph

    @staticmethod
    def _track_id_from_relation_entity(value: Any) -> int | None:
        if not isinstance(value, str):
            return None

        _, _, track_id_text = value.rpartition("_")
        if not track_id_text:
            return None
        try:
            return int(track_id_text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_visible_track_ids(snapshot: Mapping[str, Any]) -> frozenset[int]:
        objects = snapshot.get("objects")
        if not isinstance(objects, list):
            return frozenset()

        visible_ids: set[int] = set()
        for item in objects:
            if not isinstance(item, Mapping):
                continue

            state = item.get("state")
            if isinstance(state, Mapping):
                visible_flag = state.get("visible")
                if isinstance(visible_flag, bool) and not visible_flag:
                    continue

            raw_track_id = item.get("id")
            if raw_track_id is None:
                continue

            try:
                visible_ids.add(int(raw_track_id))
            except (TypeError, ValueError):
                continue

        return frozenset(visible_ids)
