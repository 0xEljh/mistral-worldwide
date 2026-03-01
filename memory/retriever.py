from __future__ import annotations

from typing import Any, Mapping

from memory.graph_history import GraphHistorySnapshot, GraphHistoryStore
from memory.semantic_index import SemanticIndex, SemanticIndexEntry


class MemoryRetriever:
    """Builds retrieval context from semantic matches and graph history."""

    def __init__(
        self,
        *,
        semantic_index: SemanticIndex,
        graph_history: GraphHistoryStore,
    ) -> None:
        self._semantic_index = semantic_index
        self._graph_history = graph_history

    def query_context(
        self,
        query: str,
        *,
        top_k: int,
        current_visible_track_ids: set[int],
    ) -> dict[str, Any]:
        query_text = query.strip()
        if not query_text:
            return {
                "query": "",
                "count": 0,
                "items": [],
            }

        matches = self._semantic_index.search(query_text, top_k=top_k)
        items = [
            self._build_item(
                entry=match.entry,
                current_visible_track_ids=current_visible_track_ids,
                similarity=match.similarity,
            )
            for match in matches
        ]
        return {
            "query": query_text,
            "count": len(items),
            "items": items,
        }

    def all_entity_memory_context(
        self,
        *,
        current_visible_track_ids: set[int],
    ) -> dict[str, Any]:
        entries = self._semantic_index.entries_snapshot()
        sorted_entries = sorted(
            entries,
            key=lambda entry: (entry.indexed_world_version, entry.track_id),
            reverse=True,
        )

        items = [
            self._build_item(
                entry=entry,
                current_visible_track_ids=current_visible_track_ids,
                similarity=None,
            )
            for entry in sorted_entries
        ]

        return {
            "count": len(items),
            "items": items,
        }

    def lookup_entities(
        self,
        *,
        current_visible_track_ids: set[int],
        track_id: int | None = None,
        class_name: str | None = None,
        include_graph_context: bool = True,
    ) -> dict[str, Any]:
        normalized_class_name = class_name.strip().lower() if class_name else None
        if track_id is None and not normalized_class_name:
            raise ValueError("Provide track_id or class_name")

        matches: list[SemanticIndexEntry] = []
        for entry in self._semantic_index.entries_snapshot():
            if track_id is not None and entry.track_id != track_id:
                continue

            if (
                normalized_class_name is not None
                and entry.object_type.strip().lower() != normalized_class_name
            ):
                continue

            matches.append(entry)

        sorted_matches = sorted(
            matches,
            key=lambda entry: (entry.indexed_world_version, entry.track_id),
            reverse=True,
        )

        items = [
            self._build_item(
                entry=entry,
                current_visible_track_ids=current_visible_track_ids,
                similarity=None,
                include_graph_context=include_graph_context,
            )
            for entry in sorted_matches
        ]
        return {
            "count": len(items),
            "items": items,
        }

    def _build_item(
        self,
        *,
        entry: SemanticIndexEntry,
        current_visible_track_ids: set[int],
        similarity: float | None,
        include_graph_context: bool = True,
    ) -> dict[str, Any]:
        graph_snapshot = (
            self._graph_history.latest_snapshot_for_track(entry.track_id)
            if include_graph_context
            else None
        )

        return {
            "track_id": entry.track_id,
            "object_type": entry.object_type,
            "description": entry.description,
            "indexed_world_version": entry.indexed_world_version,
            "described_at": entry.described_at,
            "similarity": similarity,
            "currently_visible": entry.track_id in current_visible_track_ids,
            "last_visible_graph": self._graph_summary(graph_snapshot),
        }

    @staticmethod
    def _graph_summary(snapshot: GraphHistorySnapshot | None) -> dict[str, Any] | None:
        if snapshot is None:
            return None

        objects = snapshot.graph.get("objects")
        relations = snapshot.graph.get("relations")
        recent_events = snapshot.graph.get("recent_events")

        object_labels = _extract_object_labels(objects)
        relation_examples = _extract_relation_examples(relations)

        relation_count = len(relations) if isinstance(relations, list) else 0
        recent_event_list = (
            [str(item) for item in recent_events if isinstance(item, str)]
            if isinstance(recent_events, list)
            else []
        )

        return {
            "world_version": snapshot.world_version,
            "timestamp": snapshot.timestamp,
            "object_labels": object_labels,
            "relation_count": relation_count,
            "relation_examples": relation_examples,
            "recent_events": recent_event_list,
        }


def _extract_object_labels(raw_objects: Any) -> list[str]:
    if not isinstance(raw_objects, list):
        return []

    labels: list[str] = []
    for item in raw_objects:
        if not isinstance(item, Mapping):
            continue
        object_type = item.get("type")
        track_id = item.get("id")

        if not isinstance(object_type, str):
            object_type = "object"
        if track_id is None:
            continue

        labels.append(f"{object_type}_{track_id}")

    return labels


def _extract_relation_examples(
    raw_relations: Any, *, max_examples: int = 3
) -> list[str]:
    if not isinstance(raw_relations, list):
        return []

    examples: list[str] = []
    for relation in raw_relations:
        if not isinstance(relation, Mapping):
            continue

        subject = relation.get("subject")
        relation_text = relation.get("relation")
        object_id = relation.get("object")

        if not (
            isinstance(subject, str)
            and isinstance(relation_text, str)
            and isinstance(object_id, str)
        ):
            continue

        examples.append(f"{subject} is {relation_text} {object_id}")
        if len(examples) >= max_examples:
            break

    return examples
