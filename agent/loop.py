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
ToolCallHandler = Callable[[str, str, str], None]


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
        tool_dispatcher: Any | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
    ) -> None:
        self._inference = inference
        self._prompt_builder = prompt_builder
        self._tool_dispatcher = tool_dispatcher
        self._tool_schemas = list(tool_schemas or [])

    def step(
        self,
        scene_state: SceneState,
        user_prompt: str = "",
        auxiliary_context: AuxiliaryContext | None = None,
        conversation_history: list[Mapping[str, Any]] | None = None,
        on_model_stdout: StreamChunkHandler | None = None,
        on_model_stderr: StreamChunkHandler | None = None,
        on_tool_call: ToolCallHandler | None = None,
    ) -> AgentTurn:
        prompt = self._prompt_builder(
            scene_state,
            user_prompt,
            auxiliary_context,
            conversation_history=conversation_history,
        )

        response = self._run_inference(
            prompt,
            on_model_stdout=on_model_stdout,
            on_model_stderr=on_model_stderr,
            on_tool_call=on_tool_call,
        )

        return AgentTurn(
            world_version=int(scene_state.get("world_version", -1)),
            scene_timestamp=int(scene_state.get("timestamp", -1)),
            response=response,
            prompt=prompt,
        )

    def _run_inference(
        self,
        prompt: PromptBundle,
        *,
        on_model_stdout: StreamChunkHandler | None,
        on_model_stderr: StreamChunkHandler | None,
        on_tool_call: ToolCallHandler | None,
    ) -> str:
        if (
            self._tool_dispatcher is None
            or not self._tool_schemas
            or not hasattr(self._inference, "generate_with_tools")
        ):
            return self._inference.generate(
                prompt,
                on_stdout=on_model_stdout,
                on_stderr=on_model_stderr,
            )

        tool_result = self._inference.generate_with_tools(
            prompt,
            tools=self._tool_schemas,
            on_stderr=on_model_stderr,
        )

        if not tool_result.tool_calls:
            if isinstance(tool_result.content, str) and tool_result.content.strip():
                return tool_result.content.strip()
            return self._inference.generate(
                prompt,
                on_stdout=on_model_stdout,
                on_stderr=on_model_stderr,
            )

        if not hasattr(self._inference, "complete"):
            raise RuntimeError("Inference backend does not support tool completion")

        tool_messages = list(prompt.messages)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": tool_result.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in tool_result.tool_calls
            ],
        }
        tool_messages.append(assistant_message)

        for tool_call in tool_result.tool_calls:
            tool_output = self._tool_dispatcher.dispatch(
                tool_call.name,
                tool_call.arguments,
            )
            if on_tool_call is not None:
                on_tool_call(tool_call.name, tool_call.arguments, tool_output)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": tool_output,
                }
            )

        response = self._inference.complete(
            tool_messages,
            on_stdout=on_model_stdout,
            on_stderr=on_model_stderr,
        )
        return response

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
