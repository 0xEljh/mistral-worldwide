from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_SYSTEM_PROMPT = (
    "You are an embodied scene analyst. "
    "You receive structured scene memory from a perception system and must reason only from that data. "
    "If information is missing, say so explicitly. "
    "Keep responses concise and actionable."
)


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    messages: list[dict[str, str]]


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


def _safe_json_dump(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def _format_auxiliary_context_sections(
    auxiliary_context: Mapping[str, Any],
) -> list[str]:
    sections: list[str] = []

    for key in sorted(auxiliary_context):
        value = auxiliary_context[key]
        if key == "object_descriptions":
            title = "Object descriptions (crop inference):"
            body = _format_object_descriptions_context(value)
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": composed_user_prompt},
    ]

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=composed_user_prompt,
        messages=messages,
    )
