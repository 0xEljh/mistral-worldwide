from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, Callable, Mapping

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Log, Static

from conversation.manager import ConversationManager, ConversationTrigger

SceneStateProvider = Callable[[], Mapping[str, Any]]


def _log_supports_wrap_parameter() -> bool:
    try:
        return "wrap" in inspect.signature(Log.__init__).parameters
    except (TypeError, ValueError):
        return False


class InteractiveConversationApp(App[None]):
    CSS_PATH = "styles.tcss"
    TOOL_LOG_PREVIEW_LIMIT = 200
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_transcript", "Clear"),
    ]

    def __init__(
        self,
        *,
        conversation_manager: ConversationManager,
        state_provider: SceneStateProvider,
        initial_user_prompt: str = "",
        status_refresh_interval_seconds: float = 1.0,
        transcript_max_lines: int = 2000,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__()
        if status_refresh_interval_seconds <= 0.0:
            raise ValueError("status_refresh_interval_seconds must be > 0")
        if transcript_max_lines <= 0:
            raise ValueError("transcript_max_lines must be > 0")

        self._conversation_manager = conversation_manager
        self._state_provider = state_provider
        self._initial_user_prompt = initial_user_prompt.strip()
        self._status_refresh_interval_seconds = status_refresh_interval_seconds
        self._transcript_max_lines = transcript_max_lines
        self._event_logger = logger

        self._manager_task: asyncio.Task[None] | None = None
        self._current_turn_streamed_output = False
        self._last_status_text = ""
        self._closing = False
        self._log_supports_wrap = _log_supports_wrap_parameter()
        self._fallback_stream_buffer = ""

    def compose(self) -> ComposeResult:
        transcript_kwargs: dict[str, Any] = {
            "id": "transcript",
            "highlight": False,
            "auto_scroll": True,
            "max_lines": self._transcript_max_lines,
        }
        if self._log_supports_wrap:
            transcript_kwargs["wrap"] = True

        yield Header(show_clock=True)
        yield Static("world_version=- frame=-", id="status")
        yield Log(**transcript_kwargs)
        yield Input(placeholder="Type a message and press Enter", id="prompt-input")
        yield Footer()

    async def on_mount(self) -> None:
        self._conversation_manager.on_vision_trigger = self._on_vision_trigger
        self._conversation_manager.on_turn_start = self._on_turn_start
        self._conversation_manager.on_turn_chunk = self._on_turn_chunk
        self._conversation_manager.on_turn_complete = self._on_turn_complete
        self._conversation_manager.on_notice = self._on_notice
        self._conversation_manager.on_error = self._on_error
        self._conversation_manager.on_tool_call = self._on_tool_call

        self._manager_task = asyncio.create_task(self._conversation_manager.run())
        self._manager_task.add_done_callback(self._on_manager_task_done)
        self.set_interval(self._status_refresh_interval_seconds, self._refresh_status)

        input_widget = self.query_one("#prompt-input", Input)
        input_widget.focus()

        if self._initial_user_prompt:
            self._write_user_line(self._initial_user_prompt)
            await self._conversation_manager.submit_user_message(
                self._initial_user_prompt
            )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query_text = event.value.strip()
        event.input.value = ""
        if not query_text:
            return

        if query_text.lower() in {"exit", "quit", "q"}:
            await self.action_quit()
            return

        self._write_user_line(query_text)
        await self._conversation_manager.submit_user_message(query_text)

    def on_unmount(self, _: object = None) -> None:
        self._closing = True
        if self._event_logger is not None:
            self._event_logger.info(
                "TUI unmount requested; stopping conversation manager"
            )
        self._conversation_manager.request_stop()
        self._cancel_manager_task()

    def action_clear_transcript(self) -> None:
        transcript = self._transcript()
        if transcript is None:
            return
        try:
            transcript.clear()
            self._fallback_stream_buffer = ""
        except Exception:
            return

    async def action_quit(self) -> None:
        self._closing = True
        if self._event_logger is not None:
            self._event_logger.info("Quit action triggered; stopping interactive app")
        self._conversation_manager.request_stop()
        self._cancel_manager_task()
        self._safe_exit()

    def _on_manager_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            if self._event_logger is not None:
                self._event_logger.info("Conversation manager task cancelled")
            self._safe_exit()
            return

        exception: BaseException | None = None
        try:
            exception = task.exception()
        except asyncio.CancelledError:
            if self._event_logger is not None:
                self._event_logger.info(
                    "Conversation manager task cancellation acknowledged"
                )
            self._safe_exit()
            return

        if exception is not None and not self._closing:
            if self._event_logger is not None:
                self._event_logger.error(
                    "Conversation manager task crashed",
                    exc_info=exception,
                )
            try:
                self._write_line(f"[error] Conversation loop crashed: {exception}")
            except Exception:
                pass
        self._safe_exit()

    def _refresh_status(self) -> None:
        if self._closing:
            return

        try:
            scene_state = self._state_provider()
            world_version = int(scene_state.get("world_version", -1))
            frame_index = int(scene_state.get("timestamp", -1))
            manager_status = self._conversation_manager.status_snapshot()

            status_text = (
                f"world_version={world_version} frame={frame_index} "
                f"turns={manager_status['turns_completed']} "
                f"queued_users={manager_status['pending_user_messages']} "
                f"pending_vision={manager_status['vision_event_pending']} "
                f"processing={manager_status['is_processing']}"
            )
        except Exception as exc:
            status_text = f"status unavailable: {exc}"

        if status_text == self._last_status_text:
            return

        self._last_status_text = status_text
        try:
            self.query_one("#status", Static).update(status_text)
        except Exception:
            return

    def _on_vision_trigger(self, trigger: ConversationTrigger) -> None:
        self._write_line(f"[vision] {trigger.text}")

    def _on_turn_start(
        self,
        trigger: ConversationTrigger,
        scene_state: Mapping[str, Any],
    ) -> None:
        if self._closing:
            return

        world_version = int(scene_state.get("world_version", -1))
        frame_index = int(scene_state.get("timestamp", -1))
        self._current_turn_streamed_output = False
        transcript = self._transcript()
        if transcript is None:
            return
        try:
            self._write_transcript_text(
                transcript,
                f"[agent] world_version={world_version} frame={frame_index}: ",
            )
        except Exception:
            return

    def _on_turn_chunk(self, chunk: str) -> None:
        if self._closing or not chunk:
            return
        self._current_turn_streamed_output = True
        transcript = self._transcript()
        if transcript is None:
            return
        try:
            self._write_transcript_text(transcript, chunk)
        except Exception:
            return

    def _on_turn_complete(self, trigger: ConversationTrigger, turn: Any) -> None:
        if self._closing:
            return

        transcript = self._transcript()
        if transcript is None:
            return

        if not self._current_turn_streamed_output:
            response_text = str(getattr(turn, "response", ""))
            try:
                self._write_transcript_text(transcript, response_text)
            except Exception:
                return
        try:
            self._finish_stream_output(transcript)
        except Exception:
            return

    def _on_notice(self, message: str) -> None:
        if self._closing:
            return
        self._write_line(f"[system] {message}")

    def _on_error(self, message: str) -> None:
        if self._closing:
            return
        if self._event_logger is not None:
            self._event_logger.error("Conversation error callback: %s", message)
        self._write_line(f"[error] {message}")

    def _on_tool_call(self, tool_name: str, arguments: str, result: str) -> None:
        if self._closing:
            return

        argument_preview = self._truncate_tool_log_value(arguments)
        result_preview = self._truncate_tool_log_value(result)
        self._write_line(
            f"[tool] {tool_name} args={argument_preview} result={result_preview}"
        )

    def _write_user_line(self, text: str) -> None:
        self._write_line(f"[user] {text}")

    def _truncate_tool_log_value(self, value: str) -> str:
        single_line = str(value).strip().replace("\r", "\\r").replace("\n", "\\n")
        if len(single_line) <= self.TOOL_LOG_PREVIEW_LIMIT:
            return single_line

        if self.TOOL_LOG_PREVIEW_LIMIT <= 3:
            return "..."

        cutoff = self.TOOL_LOG_PREVIEW_LIMIT - 3
        return f"{single_line[:cutoff]}..."

    def _write_line(self, text: str) -> None:
        if self._closing:
            return
        try:
            transcript = self._transcript()
            if transcript is None:
                return
            self._write_transcript_line(transcript, text)
        except Exception:
            return

    def _write_transcript_text(self, transcript: Log, text: str) -> None:
        if self._log_supports_wrap:
            transcript.write(text)
            return

        self._consume_text_fallback(transcript, text)

    def _write_transcript_line(self, transcript: Log, text: str) -> None:
        if self._log_supports_wrap:
            transcript.write_line(text)
            return

        self._flush_stream_buffer_fallback(transcript)
        self._consume_text_fallback(transcript, text)
        if self._fallback_stream_buffer:
            transcript.write_line(self._fallback_stream_buffer)
            self._fallback_stream_buffer = ""
        else:
            transcript.write_line("")

    def _finish_stream_output(self, transcript: Log) -> None:
        if self._log_supports_wrap:
            transcript.write_line("")
            return

        self._flush_stream_buffer_fallback(transcript)
        transcript.write_line("")

    def _consume_text_fallback(self, transcript: Log, text: str) -> None:
        if not text:
            return

        parts = text.split("\n")
        last_index = len(parts) - 1
        for index, part in enumerate(parts):
            self._fallback_stream_buffer += part
            self._emit_wrapped_buffer_lines(transcript)
            if index != last_index:
                transcript.write_line(self._fallback_stream_buffer)
                self._fallback_stream_buffer = ""

    def _emit_wrapped_buffer_lines(self, transcript: Log) -> None:
        wrap_width = self._fallback_wrap_width(transcript)
        while len(self._fallback_stream_buffer) > wrap_width:
            split_index = self._fallback_stream_buffer.rfind(" ", 0, wrap_width + 1)
            if split_index <= 0:
                split_index = wrap_width

            line = self._fallback_stream_buffer[:split_index].rstrip()
            if not line:
                line = self._fallback_stream_buffer[:split_index]

            transcript.write_line(line)
            self._fallback_stream_buffer = self._fallback_stream_buffer[
                split_index:
            ].lstrip()

    def _flush_stream_buffer_fallback(self, transcript: Log) -> None:
        if not self._fallback_stream_buffer:
            return
        transcript.write_line(self._fallback_stream_buffer)
        self._fallback_stream_buffer = ""

    def _fallback_wrap_width(self, transcript: Log) -> int:
        width = 0
        try:
            content_size = getattr(transcript, "content_size")
            width = int(content_size.width)
        except Exception:
            width = 0

        if width <= 0:
            try:
                size = getattr(transcript, "size")
                width = int(size.width)
            except Exception:
                width = 0

        if width <= 0:
            width = 80
        return max(width, 20)

    def _cancel_manager_task(self) -> None:
        manager_task = self._manager_task
        if manager_task is None or manager_task.done():
            return
        manager_task.cancel()

    def _safe_exit(self) -> None:
        try:
            self.exit()
        except Exception:
            return

    def _transcript(self) -> Log | None:
        try:
            return self.query_one("#transcript", Log)
        except Exception:
            return None
