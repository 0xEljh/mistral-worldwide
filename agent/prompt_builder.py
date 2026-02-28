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


def build_prompt(
    scene_state: Mapping[str, Any],
    user_prompt: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> PromptBundle:
    normalized_user_prompt = user_prompt.strip()
    scene_memory_json = json.dumps(
        scene_state, indent=2, sort_keys=True, ensure_ascii=True
    )

    task_text = normalized_user_prompt or (
        "No explicit user request. "
        "Summarize the current scene, then highlight the most important recent events."
    )

    composed_user_prompt = "\n\n".join(
        [
            "Task:",
            task_text,
            "Scene memory (JSON):",
            scene_memory_json,
        ]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": composed_user_prompt},
    ]

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=composed_user_prompt,
        messages=messages,
    )
