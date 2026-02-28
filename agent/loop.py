from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.prompt_builder import PromptBundle, build_prompt

SceneState = Mapping[str, Any]
SceneStateProvider = Callable[[], SceneState]
PromptBuilder = Callable[[SceneState, str], PromptBundle]


@dataclass(frozen=True)
class AgentTurn:
    world_version: int
    scene_timestamp: int
    response: str
    prompt: PromptBundle


class AgentLoop:
    def __init__(
        self,
        inference: Any,
        prompt_builder: PromptBuilder = build_prompt,
    ) -> None:
        self._inference = inference
        self._prompt_builder = prompt_builder

    def step(self, scene_state: SceneState, user_prompt: str = "") -> AgentTurn:
        prompt = self._prompt_builder(scene_state, user_prompt)
        response = self._inference.generate(prompt)

        return AgentTurn(
            world_version=int(scene_state.get("world_version", -1)),
            scene_timestamp=int(scene_state.get("timestamp", -1)),
            response=response,
            prompt=prompt,
        )

    def run(
        self,
        state_provider: SceneStateProvider,
        user_prompt: str = "",
        poll_interval_seconds: float = 1.0,
        only_on_scene_change: bool = True,
        require_initialized_state: bool = True,
        max_steps: int | None = None,
        on_turn: Callable[[AgentTurn], None] | None = None,
    ) -> None:
        steps = 0
        last_seen_world_version: int | None = None

        while max_steps is None or steps < max_steps:
            scene_state = state_provider()
            current_world_version = int(scene_state.get("world_version", 0))

            if require_initialized_state and current_world_version <= 0:
                time.sleep(poll_interval_seconds)
                continue

            if (
                only_on_scene_change
                and last_seen_world_version == current_world_version
            ):
                time.sleep(poll_interval_seconds)
                continue

            turn = self.step(scene_state, user_prompt=user_prompt)
            steps += 1
            last_seen_world_version = current_world_version

            if on_turn is not None:
                on_turn(turn)

            time.sleep(poll_interval_seconds)
