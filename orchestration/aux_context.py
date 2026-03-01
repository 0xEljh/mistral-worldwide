from __future__ import annotations

from typing import Any, Callable, Mapping


def _extract_visible_track_ids(scene_state: Mapping[str, Any]) -> set[int]:
    objects = scene_state.get("objects")
    if not isinstance(objects, list):
        return set()

    visible_track_ids: set[int] = set()
    for object_entry in objects:
        if not isinstance(object_entry, Mapping):
            continue

        raw_track_id = object_entry.get("id")
        if raw_track_id is None:
            continue

        try:
            visible_track_ids.add(int(raw_track_id))
        except (TypeError, ValueError):
            continue

    return visible_track_ids


def _emit_warning_once(
    *,
    key: str,
    message: str,
    warning_state: dict[str, bool],
    warning_reporter: Callable[[str], None] | None,
) -> None:
    if warning_state.get(key, False):
        return
    warning_state[key] = True
    if warning_reporter is not None:
        warning_reporter(message)


def _build_auxiliary_context_with_fallback(
    *,
    scene_state: Mapping[str, Any],
    query_text: str,
    descriptions_snapshot: Mapping[str, Any],
    memory_retriever: Any,
    semantic_search_top_k: int,
    index_descriptions: Callable[[], None],
    warning_state: dict[str, bool] | None = None,
    warning_reporter: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    active_warning_state = warning_state if warning_state is not None else {}

    try:
        index_descriptions()
    except Exception as exc:
        _emit_warning_once(
            key="semantic_indexing_unavailable",
            message=(
                "[memory] Semantic indexing unavailable; continuing without new "
                f"vector entries. Reason: {exc}"
            ),
            warning_state=active_warning_state,
            warning_reporter=warning_reporter,
        )

    visible_track_ids = _extract_visible_track_ids(scene_state)
    entity_memory_context = memory_retriever.all_entity_memory_context(
        current_visible_track_ids=visible_track_ids,
    )

    context: dict[str, Any] = {
        "object_descriptions": dict(descriptions_snapshot),
        "entity_memory": entity_memory_context,
    }

    normalized_query_text = query_text.strip()
    if normalized_query_text:
        try:
            semantic_matches = memory_retriever.query_context(
                normalized_query_text,
                top_k=semantic_search_top_k,
                current_visible_track_ids=visible_track_ids,
            )
        except Exception as exc:
            _emit_warning_once(
                key="semantic_search_unavailable",
                message=(
                    "[memory] Semantic search unavailable; continuing with scene "
                    f"memory only. Reason: {exc}"
                ),
                warning_state=active_warning_state,
                warning_reporter=warning_reporter,
            )
            semantic_matches = {
                "query": normalized_query_text,
                "count": 0,
                "items": [],
            }

        context["semantic_entity_matches"] = semantic_matches

    return context
