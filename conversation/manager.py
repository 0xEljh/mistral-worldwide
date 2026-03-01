from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.loop import AgentTurn

SceneState = Mapping[str, Any]
StateProvider = Callable[[], SceneState]
AuxiliaryContextBuilder = Callable[[SceneState, str], Mapping[str, Any]]
TurnChunkCallback = Callable[[str], None]
TurnStartCallback = Callable[["ConversationTrigger", SceneState], None]
TurnCompleteCallback = Callable[["ConversationTrigger", AgentTurn], None]
TriggerCallback = Callable[["ConversationTrigger"], None]
MessageCallback = Callable[[str], None]
ToolCallCallback = Callable[[str, str, str], None]


@dataclass(frozen=True)
class ConversationTrigger:
    source: str
    text: str
    created_at: float
    world_version: int | None = None
    frame_index: int | None = None


class ConversationManager:
    def __init__(
        self,
        *,
        agent_loop: Any,
        state_provider: StateProvider,
        auxiliary_context_builder: AuxiliaryContextBuilder,
        history_window_turns: int = 6,
        max_steps: int | None = None,
        stream_llm_output: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        if history_window_turns <= 0:
            raise ValueError("history_window_turns must be > 0")
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps must be > 0 when provided")

        self._agent_loop = agent_loop
        self._state_provider = state_provider
        self._auxiliary_context_builder = auxiliary_context_builder
        self._history_window_turns = history_window_turns
        self._max_steps = max_steps
        self._stream_llm_output = stream_llm_output
        self._logger = logger

        self._user_queue: asyncio.Queue[ConversationTrigger] = asyncio.Queue()
        self._vision_lock = threading.Lock()
        self._pending_vision_trigger: ConversationTrigger | None = None
        self._history_messages: list[dict[str, str]] = []

        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake_event: asyncio.Event | None = None
        self._stop_requested = False
        self._turns_completed = 0
        self._is_processing = False

        self.on_vision_trigger: TriggerCallback | None = None
        self.on_turn_start: TurnStartCallback | None = None
        self.on_turn_chunk: TurnChunkCallback | None = None
        self.on_turn_complete: TurnCompleteCallback | None = None
        self.on_notice: MessageCallback | None = None
        self.on_error: MessageCallback | None = None
        self.on_tool_call: ToolCallCallback | None = None

    @property
    def turns_completed(self) -> int:
        return self._turns_completed

    async def submit_user_message(self, text: str) -> None:
        normalized_text = text.strip()
        if not normalized_text:
            return

        await self._user_queue.put(
            ConversationTrigger(
                source="user",
                text=normalized_text,
                created_at=time.time(),
            )
        )
        self._set_wake_event_threadsafe()

    def notify_critical_vision_event(
        self,
        *,
        description: str,
        world_version: int | None = None,
        frame_index: int | None = None,
    ) -> None:
        normalized_description = description.strip()
        if not normalized_description:
            return

        trigger = ConversationTrigger(
            source="vision",
            text=normalized_description,
            created_at=time.time(),
            world_version=world_version,
            frame_index=frame_index,
        )

        with self._vision_lock:
            self._pending_vision_trigger = trigger

        self._set_wake_event_threadsafe()

    def request_stop(self) -> None:
        self._stop_requested = True
        self._set_wake_event_threadsafe()

    def status_snapshot(self) -> dict[str, Any]:
        with self._vision_lock:
            pending_vision = self._pending_vision_trigger is not None

        return {
            "turns_completed": self._turns_completed,
            "pending_user_messages": self._user_queue.qsize(),
            "vision_event_pending": pending_vision,
            "is_processing": self._is_processing,
        }

    async def run(self) -> None:
        if self._loop is not None:
            raise RuntimeError("ConversationManager.run() is already active")

        self._loop = asyncio.get_running_loop()
        self._wake_event = asyncio.Event()
        self._log_info("ConversationManager loop started")

        try:
            while not self._stop_requested:
                if (
                    self._max_steps is not None
                    and self._turns_completed >= self._max_steps
                ):
                    self._stop_requested = True
                    self._emit_notice("Reached max interactive turns; ending session.")
                    break

                trigger = await self._next_trigger()
                if trigger is None:
                    continue

                await self._run_turn(trigger)
        finally:
            self._log_info("ConversationManager loop stopped")
            self._loop = None
            self._wake_event = None

    async def _next_trigger(self) -> ConversationTrigger | None:
        while not self._stop_requested:
            user_trigger = self._try_get_user_trigger_nowait()
            if user_trigger is not None:
                return user_trigger

            vision_trigger = self._consume_pending_vision_trigger()
            if vision_trigger is not None:
                self._emit_vision_trigger(vision_trigger)
                return vision_trigger

            wake_event = self._wake_event
            if wake_event is None:
                await asyncio.sleep(0)
                continue

            wake_event.clear()

            user_trigger = self._try_get_user_trigger_nowait()
            if user_trigger is not None:
                return user_trigger

            vision_trigger = self._consume_pending_vision_trigger()
            if vision_trigger is not None:
                self._emit_vision_trigger(vision_trigger)
                return vision_trigger

            await wake_event.wait()

        return None

    def _try_get_user_trigger_nowait(self) -> ConversationTrigger | None:
        try:
            return self._user_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _consume_pending_vision_trigger(self) -> ConversationTrigger | None:
        with self._vision_lock:
            trigger = self._pending_vision_trigger
            self._pending_vision_trigger = None
        return trigger

    async def _run_turn(self, trigger: ConversationTrigger) -> None:
        history_messages = self._history_for_prompt()
        self._is_processing = True

        try:
            turn = await asyncio.to_thread(
                self._run_turn_sync,
                trigger,
                history_messages,
            )
        except Exception as exc:
            self._log_exception("Failed to run conversation turn", exc)
            self._emit_error(f"Failed to run conversation turn: {exc}")
            return
        finally:
            self._is_processing = False

        self._turns_completed += 1
        self._history_messages.extend(
            [
                {"role": "user", "content": trigger.text},
                {"role": "assistant", "content": turn.response},
            ]
        )
        self._trim_history_messages()
        self._emit_turn_complete(trigger, turn)

    def _run_turn_sync(
        self,
        trigger: ConversationTrigger,
        history_messages: list[dict[str, str]],
    ) -> AgentTurn:
        scene_state = self._state_provider()
        scene_state_dict = dict(scene_state)
        self._emit_turn_start_threadsafe(trigger, scene_state_dict)

        auxiliary_context = self._auxiliary_context_builder(scene_state, trigger.text)

        return self._agent_loop.step(
            scene_state,
            user_prompt=trigger.text,
            auxiliary_context=auxiliary_context,
            conversation_history=history_messages,
            on_model_stdout=(
                self._on_model_stdout_chunk if self._stream_llm_output else None
            ),
            on_model_stderr=self._on_model_stderr_chunk,
            on_tool_call=self._emit_tool_call_threadsafe,
        )

    def _history_for_prompt(self) -> list[dict[str, str]]:
        max_messages = self._history_window_turns * 2
        if len(self._history_messages) <= max_messages:
            return list(self._history_messages)
        return list(self._history_messages[-max_messages:])

    def _trim_history_messages(self) -> None:
        max_messages = self._history_window_turns * 2
        if len(self._history_messages) > max_messages:
            self._history_messages = self._history_messages[-max_messages:]

    def _on_model_stdout_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._emit_turn_chunk_threadsafe(chunk)

    def _on_model_stderr_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._emit_error_threadsafe(chunk)

    def _set_wake_event_threadsafe(self) -> None:
        loop = self._loop
        wake_event = self._wake_event
        if loop is None or wake_event is None:
            return
        self._call_soon_threadsafe(loop, wake_event.set)

    def _emit_vision_trigger(self, trigger: ConversationTrigger) -> None:
        callback = self.on_vision_trigger
        if callback is not None:
            self._safe_callback(callback, trigger)

    def _emit_turn_start_threadsafe(
        self,
        trigger: ConversationTrigger,
        scene_state: SceneState,
    ) -> None:
        callback = self.on_turn_start
        if callback is None:
            return

        loop = self._loop
        if loop is None:
            self._safe_callback(callback, trigger, scene_state)
            return

        self._call_soon_threadsafe(
            loop,
            self._safe_callback,
            callback,
            trigger,
            scene_state,
        )

    def _emit_turn_chunk_threadsafe(self, chunk: str) -> None:
        callback = self.on_turn_chunk
        if callback is None:
            return

        loop = self._loop
        if loop is None:
            self._safe_callback(callback, chunk)
            return

        self._call_soon_threadsafe(
            loop,
            self._safe_callback,
            callback,
            chunk,
        )

    def _emit_turn_complete(
        self, trigger: ConversationTrigger, turn: AgentTurn
    ) -> None:
        callback = self.on_turn_complete
        if callback is not None:
            self._safe_callback(callback, trigger, turn)

    def _emit_tool_call_threadsafe(
        self,
        tool_name: str,
        arguments: str,
        result: str,
    ) -> None:
        callback = self.on_tool_call
        if callback is None:
            return

        loop = self._loop
        if loop is None:
            self._safe_callback(callback, tool_name, arguments, result)
            return

        self._call_soon_threadsafe(
            loop,
            self._safe_callback,
            callback,
            tool_name,
            arguments,
            result,
        )

    def _emit_notice(self, message: str) -> None:
        callback = self.on_notice
        if callback is not None:
            self._safe_callback(callback, message)

    def _emit_error(self, message: str) -> None:
        callback = self.on_error
        if callback is not None:
            self._safe_callback(callback, message)

    def _emit_error_threadsafe(self, message: str) -> None:
        callback = self.on_error
        if callback is None:
            return

        loop = self._loop
        if loop is None:
            self._safe_callback(callback, message)
            return

        self._call_soon_threadsafe(
            loop,
            self._safe_callback,
            callback,
            message,
        )

    def _call_soon_threadsafe(
        self,
        loop: asyncio.AbstractEventLoop,
        callback: Callable[..., None],
        *args: object,
    ) -> None:
        try:
            loop.call_soon_threadsafe(callback, *args)
        except RuntimeError as exc:
            self._log_warning(
                "Dropping callback while event loop is closing: %s",
                exc,
            )

    def _log_info(self, message: str, *args: object) -> None:
        logger = self._logger
        if logger is not None:
            logger.info(message, *args)

    def _log_warning(self, message: str, *args: object) -> None:
        logger = self._logger
        if logger is not None:
            logger.warning(message, *args)

    def _log_exception(self, message: str, exc: BaseException) -> None:
        logger = self._logger
        if logger is not None:
            logger.error(message, exc_info=exc)

    def _safe_callback(self, callback: Callable[..., None], *args: object) -> None:
        try:
            callback(*args)
        except Exception as exc:
            self._log_exception("Conversation callback failed", exc)
            return
