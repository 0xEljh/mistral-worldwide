from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

# DEFAULT_SYSTEM_PROMPT = (
#    "You are an embodied scene analyst. "
#    "You receive structured scene memory from a perception system and must reason only from that data. "
#    "If information is missing, say so explicitly. "
#    "Keep responses concise and actionable."
# )

DEFAULT_SYSTEM_PROMPT = (
    "You are an embodied visual agent observing a physical scene. "
    "Use natural, human language. Avoid robotic phrasing. DO NOT USE () or **"
    "You operate in 2 modes, a user triggered mode and a event triggered mode"
    "In the event triggered mode, you receive: "
    "- A structured world state snapshot "
    "- Recent event history "
    "- A user query "
    "User mode Constraints: "
    "- Reason ONLY from the provided data. "
    "- Do not hallucinate unseen objects or actions. "
    "- If information is missing, state what is missing. "
    "- Answer only the user’s question. "
    "- Do not narrate the entire scene unless asked. "
    "- Keep responses under 3 sentences. "
    "- Prefer spatial relationships (e.g., left of the laptop) over raw coordinates. "
    "- If uncertain, state uncertainty briefly. "
    "In the event triggered mode, you receive: "
    "- A structured world state snapshot "
    "- Recent event history "
    "Event mode Constraints: "
    "- Describe the events in 1 line\n"
    "DO NOT INCLUDE X/Y COORDINATES OR NUMERICAL VALUES IN YOUR REPLIES.\n"
    "You should be making inferences about the scene/query based on the data presented to you."
    "For example, if a banana *disappears* after overlapping a bag,"
    "you can infer that it was occluded or that it was put into another object"
)


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    messages: list[dict[str, str]]


def _normalize_conversation_history(
    conversation_history: list[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    if not conversation_history:
        return []

    normalized_messages: list[dict[str, str]] = []
    for message in conversation_history:
        if not isinstance(message, Mapping):
            continue

        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue

        normalized_content = content.strip()
        if not normalized_content:
            continue

        normalized_messages.append(
            {
                "role": role,
                "content": normalized_content,
            }
        )

    return normalized_messages


def _format_object_descriptions_context(context: Any) -> str:
    if not isinstance(context, Mapping):
        return _safe_json_dump(context)

    items = context.get("items")
    if not isinstance(items, list) or not items:
        text_by_track_id = context.get("text_by_track_id")
        if isinstance(text_by_track_id, Mapping) and text_by_track_id:
            compact_lines = [
                f"object_{track_id}: {description.strip()}"
                for track_id, description in text_by_track_id.items()
                if isinstance(description, str) and description.strip()
            ]
            if compact_lines:
                return "\n".join(compact_lines)
        return "No object crop descriptions available yet."

    lines: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue

        description = item.get("description")
        if not isinstance(description, str) or not description.strip():
            continue

        object_type = item.get("type")
        track_id = item.get("id")
        object_type_text = (
            object_type.strip()
            if isinstance(object_type, str) and object_type.strip()
            else "object"
        )
        track_id_text = str(track_id) if track_id is not None else "unknown"
        lines.append(f"{object_type_text}_{track_id_text}: {description.strip()}")

    if not lines:
        return "No object crop descriptions available yet."
    return "\n".join(lines)


def _format_last_visible_graph(context: Any) -> str:
    if not isinstance(context, Mapping):
        return "last graph unavailable"

    world_version = context.get("world_version")
    timestamp = context.get("timestamp")
    object_labels = context.get("object_labels")
    relation_count = context.get("relation_count")
    relation_examples = context.get("relation_examples")

    object_text = "none"
    if isinstance(object_labels, list):
        normalized_labels = [
            label.strip() for label in object_labels if isinstance(label, str) and label
        ]
        if normalized_labels:
            object_text = ", ".join(normalized_labels)

    relation_count_text = (
        str(int(relation_count))
        if isinstance(relation_count, (int, float))
        else "unknown"
    )

    summary = (
        f"last_seen world_version={world_version} timestamp={timestamp} "
        f"objects=[{object_text}] relations={relation_count_text}"
    )

    if isinstance(relation_examples, list):
        examples = [
            item.strip()
            for item in relation_examples
            if isinstance(item, str) and item.strip()
        ]
        if examples:
            summary += "; relation_examples=" + " | ".join(examples)

    return summary


def _format_entity_memory_item(
    item: Mapping[str, Any],
    *,
    include_similarity: bool,
) -> str | None:
    track_id = item.get("track_id", item.get("id"))
    object_type = item.get("object_type", item.get("type"))
    description = item.get("description")
    currently_visible = item.get("currently_visible")
    similarity = item.get("similarity")
    graph_context = item.get("last_visible_graph")

    if not isinstance(description, str) or not description.strip():
        return None

    if isinstance(object_type, str) and object_type.strip():
        object_label = object_type.strip()
    else:
        object_label = "object"

    track_text = str(track_id) if track_id is not None else "unknown"
    visibility_text = (
        "visible"
        if isinstance(currently_visible, bool) and currently_visible
        else "not visible"
    )
    line = f"{object_label}_{track_text} ({visibility_text})"

    if include_similarity and isinstance(similarity, (int, float)):
        line += f" sim={float(similarity):.3f}"

    line += f": {description.strip()}"
    line += f" | {_format_last_visible_graph(graph_context)}"
    return line


def _format_semantic_entity_matches_context(context: Any) -> str:
    if not isinstance(context, Mapping):
        return _safe_json_dump(context)

    query = context.get("query")
    items = context.get("items")

    lines: list[str] = []
    if isinstance(query, str) and query.strip():
        lines.append(f'query="{query.strip()}"')

    if not isinstance(items, list) or not items:
        if lines:
            lines.append("No semantic entity matches found.")
            return "\n".join(lines)
        return "No semantic entity matches available yet."

    for item in items:
        if not isinstance(item, Mapping):
            continue
        formatted = _format_entity_memory_item(item, include_similarity=True)
        if formatted is not None:
            lines.append(formatted)

    if not lines:
        return "No semantic entity matches available yet."
    return "\n".join(lines)


def _format_entity_memory_context(context: Any) -> str:
    if not isinstance(context, Mapping):
        return _safe_json_dump(context)

    items = context.get("items")
    if not isinstance(items, list) or not items:
        return "No entity memory entries available yet."

    lines: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        formatted = _format_entity_memory_item(item, include_similarity=False)
        if formatted is not None:
            lines.append(formatted)

    if not lines:
        return "No entity memory entries available yet."
    return "\n".join(lines)


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def _format_auxiliary_context_sections(
    auxiliary_context: Mapping[str, Any],
) -> list[str]:
    sections: list[str] = []

    key_priority = {
        "object_descriptions": 0,
        "semantic_entity_matches": 1,
        "entity_memory": 2,
    }

    for key in sorted(
        auxiliary_context, key=lambda item: (key_priority.get(item, 99), item)
    ):
        value = auxiliary_context[key]
        if key == "object_descriptions":
            title = "Object descriptions (crop inference):"
            body = _format_object_descriptions_context(value)
        elif key == "semantic_entity_matches":
            title = "Semantic entity matches:"
            body = _format_semantic_entity_matches_context(value)
        elif key == "entity_memory":
            title = "Entity memory (last visible graph):"
            body = _format_entity_memory_context(value)
        else:
            title = f"{key.replace('_', ' ').title()}:"
            body = _safe_json_dump(value)

        sections.extend([title, body])

    return sections


def build_prompt(
    scene_state: Mapping[str, Any],
    user_prompt: str = "",
    auxiliary_context: Mapping[str, Any] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    conversation_history: list[Mapping[str, Any]] | None = None,
) -> PromptBundle:
    normalized_user_prompt = user_prompt.strip()
    scene_memory_json = _safe_json_dump(scene_state)

    task_text = normalized_user_prompt or (
        "No explicit user request. "
        "Summarize the current scene, then highlight the most important recent events."
    )

    prompt_sections = [
        "Task:",
        task_text,
        "Scene memory (JSON):",
        scene_memory_json,
    ]
    if auxiliary_context:
        prompt_sections.extend(_format_auxiliary_context_sections(auxiliary_context))

    composed_user_prompt = "\n\n".join(prompt_sections)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_normalize_conversation_history(conversation_history))
    messages.append({"role": "user", "content": composed_user_prompt})

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=composed_user_prompt,
        messages=messages,
    )
