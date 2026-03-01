from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping

from memory.retriever import MemoryRetriever

logger = logging.getLogger(__name__)

_STALE_DESCRIPTION_MARKERS = (
    "description unavailable",
    "invalid crop data",
    "inference error",
)


class ToolDispatcher:
    """Simple callable registry for tool execution."""

    def __init__(
        self,
        *,
        retriever: MemoryRetriever,
        scene_state_provider: Callable[[], Mapping[str, Any]],
        text_by_track_id_provider: Callable[[], Mapping[int, str]],
    ) -> None:
        self._retriever = retriever
        self._scene_state_provider = scene_state_provider
        self._text_by_track_id_provider = text_by_track_id_provider
        self._handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "lookup_entity": self._handle_lookup_entity,
            "describe_scene": self._handle_describe_scene,
        }

    def dispatch(self, tool_name: str, arguments: str | dict[str, Any]) -> str:
        parsed_arguments: dict[str, Any]
        if isinstance(arguments, str):
            try:
                loaded = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return json.dumps({"error": f"Invalid tool argument JSON: {exc}"})
            if not isinstance(loaded, dict):
                return json.dumps({"error": "Tool arguments must decode to an object"})
            parsed_arguments = loaded
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        else:
            return json.dumps({"error": "Tool arguments must be a JSON object"})

        handler = self._handlers.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = handler(**parsed_arguments)
            return json.dumps(result, ensure_ascii=True, default=str)
        except Exception as exc:
            logger.error(
                "Tool dispatch failed for %s: %s", tool_name, exc, exc_info=True
            )
            return json.dumps({"error": str(exc)})

    def _handle_lookup_entity(
        self,
        *,
        track_id: int | None = None,
        class_name: str | None = None,
        include_graph_context: bool = True,
    ) -> dict[str, Any]:
        scene_state = self._scene_state_provider()
        visible_track_ids = _extract_visible_track_ids(scene_state)
        retrieved = self._retriever.lookup_entities(
            current_visible_track_ids=visible_track_ids,
            track_id=track_id,
            class_name=class_name,
            include_graph_context=include_graph_context,
        )
        return {
            "count": retrieved["count"],
            "entities": retrieved["items"],
        }

    def _handle_describe_scene(
        self,
        *,
        region: str = "full",
        max_entities: int = 20,
        include_stale: bool = False,
    ) -> dict[str, Any]:
        scene_state = self._scene_state_provider()
        text_by_track_id = self._text_by_track_id_provider()

        objects = scene_state.get("objects")
        if not isinstance(objects, list):
            return {
                "region": region,
                "entity_count": 0,
                "entities": [],
            }

        entities = [
            _build_scene_entity(item, text_by_track_id=text_by_track_id)
            for item in objects
        ]
        entities = [entity for entity in entities if entity is not None]

        if not include_stale:
            entities = [
                entity
                for entity in entities
                if not bool(entity.get("description_stale", False))
            ]

        filtered_entities = self._filter_by_region(entities, region, scene_state)
        filtered_entities.sort(
            key=lambda entity: float(entity.get("confidence") or 0.0),
            reverse=True,
        )

        if max_entities < 0:
            max_entities = 0
        selected_entities = filtered_entities[:max_entities]
        selected_track_ids: set[int] = set()
        for item in selected_entities:
            track_id = _coerce_int(item.get("track_id"))
            if track_id is not None:
                selected_track_ids.add(track_id)
        relation_subset = _filter_relations(scene_state, selected_track_ids)

        return {
            "region": region,
            "entity_count": len(selected_entities),
            "relation_count": len(relation_subset),
            "relations": relation_subset,
            "entities": selected_entities,
        }

    def _filter_by_region(
        self,
        entities: list[dict[str, Any]],
        region: str,
        scene_state: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        region_bounds = {
            "left": (0.0, 0.0, 0.33, 1.0),
            "center": (0.33, 0.0, 0.66, 1.0),
            "right": (0.66, 0.0, 1.0, 1.0),
            "top": (0.0, 0.0, 1.0, 0.5),
            "bottom": (0.0, 0.5, 1.0, 1.0),
        }

        if region == "full":
            return entities

        bounds = region_bounds.get(region)
        if bounds is None:
            return entities

        frame_width, frame_height = _extract_frame_size(scene_state)
        left, top, right, bottom = bounds

        filtered: list[dict[str, Any]] = []
        for entity in entities:
            center = _normalized_center(
                entity,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            if center is None:
                continue

            center_x, center_y = center
            if left <= center_x <= right and top <= center_y <= bottom:
                filtered.append(entity)

        return filtered


def _extract_visible_track_ids(scene_state: Mapping[str, Any]) -> set[int]:
    objects = scene_state.get("objects")
    if not isinstance(objects, list):
        return set()

    visible_track_ids: set[int] = set()
    for object_entry in objects:
        if not isinstance(object_entry, Mapping):
            continue

        state = object_entry.get("state")
        if isinstance(state, Mapping):
            visible_flag = state.get("visible")
            if isinstance(visible_flag, bool) and not visible_flag:
                continue

        raw_track_id = object_entry.get("id")
        if raw_track_id is None:
            continue

        try:
            visible_track_ids.add(int(raw_track_id))
        except (TypeError, ValueError):
            continue

    return visible_track_ids


def _build_scene_entity(
    raw_object: Any,
    *,
    text_by_track_id: Mapping[int, str],
) -> dict[str, Any] | None:
    if not isinstance(raw_object, Mapping):
        return None

    track_id = _coerce_int(raw_object.get("id"))
    if track_id is None:
        return None

    object_type = raw_object.get("type")
    object_type_text = (
        object_type.strip()
        if isinstance(object_type, str) and object_type.strip()
        else "object"
    )

    description = text_by_track_id.get(track_id)
    description_text = description.strip() if isinstance(description, str) else ""
    description_stale = _is_stale_description(description_text)
    if not description_text:
        description_text = object_type_text

    confidence = raw_object.get("confidence")
    confidence_value = (
        float(confidence) if isinstance(confidence, (int, float)) else None
    )

    state = raw_object.get("state")
    visible = True
    if isinstance(state, Mapping):
        visible_flag = state.get("visible")
        if isinstance(visible_flag, bool):
            visible = visible_flag

    return {
        "track_id": track_id,
        "object_type": object_type_text,
        "description": description_text,
        "description_stale": description_stale,
        "confidence": confidence_value,
        "position": raw_object.get("position"),
        "bbox": raw_object.get("bbox"),
        "visible": visible,
    }


def _is_stale_description(description: str) -> bool:
    normalized = description.strip().lower()
    if not normalized:
        return False
    return any(marker in normalized for marker in _STALE_DESCRIPTION_MARKERS)


def _extract_frame_size(
    scene_state: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    frame_size = scene_state.get("frame_size")
    if isinstance(frame_size, Mapping):
        width = _coerce_positive_float(frame_size.get("width"))
        height = _coerce_positive_float(frame_size.get("height"))
        if width is not None and height is not None:
            return width, height

    width = _coerce_positive_float(scene_state.get("frame_width"))
    height = _coerce_positive_float(scene_state.get("frame_height"))
    return width, height


def _normalized_center(
    entity: Mapping[str, Any],
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[float, float] | None:
    bbox_center = _center_from_bbox(
        entity.get("bbox"),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if bbox_center is not None:
        return bbox_center

    position = entity.get("position")
    if not isinstance(position, Mapping):
        return None

    center_x = _coerce_float(position.get("x"))
    center_y = _coerce_float(position.get("y"))
    if center_x is None or center_y is None:
        return None

    if 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0:
        return center_x, center_y

    if frame_width is None or frame_height is None:
        return None

    if frame_width <= 0.0 or frame_height <= 0.0:
        return None

    return center_x / frame_width, center_y / frame_height


def _center_from_bbox(
    raw_bbox: Any,
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> tuple[float, float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None

    x1 = _coerce_float(raw_bbox[0])
    y1 = _coerce_float(raw_bbox[1])
    x2 = _coerce_float(raw_bbox[2])
    y2 = _coerce_float(raw_bbox[3])
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    if 0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0:
        return center_x, center_y

    if frame_width is None or frame_height is None:
        return None
    if frame_width <= 0.0 or frame_height <= 0.0:
        return None
    return center_x / frame_width, center_y / frame_height


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_float(value: Any) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None or parsed <= 0.0:
        return None
    return parsed


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _filter_relations(
    scene_state: Mapping[str, Any],
    included_track_ids: set[int],
) -> list[dict[str, Any]]:
    if not included_track_ids:
        return []

    raw_relations = scene_state.get("relations")
    if not isinstance(raw_relations, list):
        return []

    filtered: list[dict[str, Any]] = []
    for relation in raw_relations:
        if not isinstance(relation, Mapping):
            continue

        subject_track_id = _track_id_from_relation_label(relation.get("subject"))
        object_track_id = _track_id_from_relation_label(relation.get("object"))
        if subject_track_id is None or object_track_id is None:
            continue

        if (
            subject_track_id not in included_track_ids
            and object_track_id not in included_track_ids
        ):
            continue

        filtered.append(dict(relation))

    return filtered


def _track_id_from_relation_label(value: Any) -> int | None:
    if not isinstance(value, str):
        return None

    _, _, track_id_text = value.rpartition("_")
    if not track_id_text:
        return None

    try:
        return int(track_id_text)
    except (TypeError, ValueError):
        return None
