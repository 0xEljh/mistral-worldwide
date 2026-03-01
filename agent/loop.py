from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.prompt_builder import PromptBundle, build_prompt

SceneState = Mapping[str, Any]
SceneStateProvider = Callable[[], SceneState]
AuxiliaryContext = Mapping[str, Any]
AuxiliaryContextProvider = Callable[[], AuxiliaryContext]
PromptBuilder = Callable[..., PromptBundle]
StreamChunkHandler = Callable[[str], None]


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

    def step(
        self,
        scene_state: SceneState,
        user_prompt: str = "",
        auxiliary_context: AuxiliaryContext | None = None,
        conversation_history: list[Mapping[str, Any]] | None = None,
        on_model_stdout: StreamChunkHandler | None = None,
        on_model_stderr: StreamChunkHandler | None = None,
    ) -> AgentTurn:
        prompt = self._prompt_builder(
            scene_state,
            user_prompt,
            auxiliary_context,
            conversation_history=conversation_history,
        )
        response = self._inference.generate(
            prompt,
            on_stdout=on_model_stdout,
            on_stderr=on_model_stderr,
        )

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
        require_initialized_state: bool = True,
        max_steps: int | None = None,
        auxiliary_context_provider: AuxiliaryContextProvider | None = None,
        on_turn: Callable[[AgentTurn], None] | None = None,
        on_model_stdout: StreamChunkHandler | None = None,
        on_model_stderr: StreamChunkHandler | None = None,
    ) -> None:
        steps = 0

        while max_steps is None or steps < max_steps:
            scene_state = state_provider()
            current_world_version = int(scene_state.get("world_version", 0))

            if require_initialized_state and current_world_version <= 0:
                time.sleep(poll_interval_seconds)
                continue

            auxiliary_context = (
                auxiliary_context_provider()
                if auxiliary_context_provider is not None
                else None
            )

            turn = self.step(
                scene_state,
                user_prompt=user_prompt,
                auxiliary_context=auxiliary_context,
                on_model_stdout=on_model_stdout,
                on_model_stderr=on_model_stderr,
            )
            steps += 1

            if on_turn is not None:
                on_turn(turn)

            time.sleep(poll_interval_seconds)
