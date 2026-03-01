from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Input, Log, Static

from conversation.manager import ConversationManager, ConversationTrigger

SceneStateProvider = Callable[[], Mapping[str, Any]]


class InteractiveConversationApp(App[None]):
    CSS_PATH = "styles.tcss"
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
    ) -> None:
        super().__init__()
        if status_refresh_interval_seconds <= 0.0:
            raise ValueError("status_refresh_interval_seconds must be > 0")

        self._conversation_manager = conversation_manager
        self._state_provider = state_provider
        self._initial_user_prompt = initial_user_prompt.strip()
        self._status_refresh_interval_seconds = status_refresh_interval_seconds

        self._manager_task: asyncio.Task[None] | None = None
        self._current_turn_streamed_output = False
        self._last_status_text = ""
        self._closing = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("world_version=- frame=-", id="status")
        yield Log(id="transcript", highlight=False, auto_scroll=True)
        yield Input(placeholder="Type a message and press Enter", id="prompt-input")
        yield Footer()

    async def on_mount(self) -> None:
        self._conversation_manager.on_vision_trigger = self._on_vision_trigger
        self._conversation_manager.on_turn_start = self._on_turn_start
        self._conversation_manager.on_turn_chunk = self._on_turn_chunk
        self._conversation_manager.on_turn_complete = self._on_turn_complete
        self._conversation_manager.on_notice = self._on_notice
        self._conversation_manager.on_error = self._on_error

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
        self._conversation_manager.request_stop()

    def action_clear_transcript(self) -> None:
        self._transcript().clear()

    async def action_quit(self) -> None:
        self._closing = True
        self._conversation_manager.request_stop()
        self.exit()

    def _on_manager_task_done(self, task: asyncio.Task[None]) -> None:
        if self._closing:
            return

        if task.cancelled():
            self.exit()
            return

        exception = task.exception()
        if exception is not None:
            self._write_line(f"[error] Conversation loop crashed: {exception}")
        self.exit()

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
        self.query_one("#status", Static).update(status_text)

    def _on_vision_trigger(self, trigger: ConversationTrigger) -> None:
        self._write_line(f"[vision] {trigger.text}")

    def _on_turn_start(
        self,
        trigger: ConversationTrigger,
        scene_state: Mapping[str, Any],
    ) -> None:
        world_version = int(scene_state.get("world_version", -1))
        frame_index = int(scene_state.get("timestamp", -1))
        self._current_turn_streamed_output = False
        self._transcript().write(
            f"[agent] world_version={world_version} frame={frame_index}: "
        )

    def _on_turn_chunk(self, chunk: str) -> None:
        if not chunk:
            return
        self._current_turn_streamed_output = True
        self._transcript().write(chunk)

    def _on_turn_complete(self, trigger: ConversationTrigger, turn: Any) -> None:
        if not self._current_turn_streamed_output:
            response_text = str(getattr(turn, "response", ""))
            self._transcript().write(response_text)
        self._transcript().write_line("")

    def _on_notice(self, message: str) -> None:
        self._write_line(f"[system] {message}")

    def _on_error(self, message: str) -> None:
        self._write_line(f"[error] {message}")

    def _write_user_line(self, text: str) -> None:
        self._write_line(f"[user] {text}")

    def _write_line(self, text: str) -> None:
        self._transcript().write_line(text)

    def _transcript(self) -> Log:
        return self.query_one("#transcript", Log)
