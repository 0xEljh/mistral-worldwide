from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, cast

from api.embedded_server import EmbeddedApiServer
from agent.crop_describer import CropDescriptionStore, run_crop_description_loop
from agent.inference import LlamaCppServerConfig, LlamaCppServerInference
from agent.loop import AgentLoop, AgentTurn
from agent.tools import TOOL_SCHEMAS, ToolDispatcher
from memory import EmbeddingModel, GraphHistoryStore, MemoryRetriever, SemanticIndex
from observability.vizgraph import GraphSnapshotRecorder
from orchestration.aux_context import _build_auxiliary_context_with_fallback
from perception.frame_provider import FrameSourceMode
from perception.yoloworldstate import WorldState, run_world_state_tracking_loop


def _default_turn_handler(turn: AgentTurn) -> None:
    print(
        f"[agent] world_version={turn.world_version} frame={turn.scene_timestamp}\n"
        f"{turn.response}\n"
    )


def _streaming_turn_handler(turn: AgentTurn) -> None:
    print(
        f"\n[agent] world_version={turn.world_version} frame={turn.scene_timestamp}",
        flush=True,
    )


def _is_indexable_description(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False

    unavailable_markers = (
        "description unavailable",
        "invalid crop data",
        "inference error",
    )
    return not any(marker in normalized for marker in unavailable_markers)


def _resolve_interactive_log_path(log_file: str | None) -> Path:
    if log_file is not None and log_file.strip():
        return Path(log_file).expanduser().resolve()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "logs"
        / f"interactive-{timestamp}.log"
    )


def _build_interactive_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pipeline.interactive.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


def run_pipeline(
    user_prompt: str = "",
    quantization: str = "Q4_K_M",
    poll_interval_seconds: float = 1.0,
    crop_description_poll_interval_seconds: float = 0.5,
    max_steps: int | None = None,
    stream_llm_output: bool = True,
    perception_startup_timeout_seconds: float = 5.0,
    camera_index: int = 0,
    display_perception: bool = True,
    model_path: str = "yolo26x.pt",
    tracker_path: str = "perception/botsort.yaml",
    frame_source_mode: FrameSourceMode = "auto",
    api_stale_after_seconds: float = 1.0,
    switch_to_api_after_consecutive: int = 5,
    switch_cooldown_seconds: float = 2.0,
    api_ingest_host: str = "0.0.0.0",
    api_ingest_port: int = 8000,
    api_ingest_startup_timeout_seconds: float = 5.0,
    llm_server_port: int = 8081,
    llm_gpu_layers: str = "auto",
    llm_cpu_only: bool = False,
    llm_cpu_fallback: bool = True,
    semantic_search_top_k: int = 5,
    interactive_user_input: bool = False,
    interactive_history_window_turns: int = 6,
    interactive_status_refresh_interval_seconds: float = 1.0,
    interactive_transcript_max_lines: int = 2000,
    interactive_log_file: str | None = None,
    graph_history_interval_world_versions: int = 50,
    graph_history_max_snapshots: int = 100,
    graph_snapshot_interval: int = 100,
    offline: bool = False,
) -> None:
    if graph_snapshot_interval < 0:
        raise ValueError("graph_snapshot_interval must be >= 0")
    if semantic_search_top_k <= 0:
        raise ValueError("semantic_search_top_k must be > 0")
    if interactive_history_window_turns <= 0:
        raise ValueError("interactive_history_window_turns must be > 0")
    if interactive_status_refresh_interval_seconds <= 0.0:
        raise ValueError("interactive_status_refresh_interval_seconds must be > 0")
    if interactive_transcript_max_lines <= 0:
        raise ValueError("interactive_transcript_max_lines must be > 0")
    if graph_history_interval_world_versions <= 0:
        raise ValueError("graph_history_interval_world_versions must be > 0")
    if graph_history_max_snapshots <= 0:
        raise ValueError("graph_history_max_snapshots must be > 0")

    shared_state = WorldState()
    crop_description_store = CropDescriptionStore()
    embedding_model = EmbeddingModel(offline=offline)
    semantic_index = SemanticIndex(embedding_model)
    graph_history_store = GraphHistoryStore(
        save_interval_world_versions=graph_history_interval_world_versions,
        max_snapshots=graph_history_max_snapshots,
    )
    memory_retriever = MemoryRetriever(
        semantic_index=semantic_index,
        graph_history=graph_history_store,
    )
    stop_event = threading.Event()
    failure_box: dict[str, Exception | None] = {"error": None}
    embedded_api_server: EmbeddedApiServer | None = None
    perception_thread: threading.Thread | None = None
    crop_description_thread: threading.Thread | None = None
    inference: LlamaCppServerInference | None = None
    graph_recorder: GraphSnapshotRecorder | None = None
    conversation_manager: Any | None = None

    if graph_snapshot_interval > 0:
        graph_run_id = time.strftime("%Y%m%d-%H%M%S")
        graph_output_dir = (
            Path(__file__).resolve().parents[1] / "artifacts" / "graphs" / graph_run_id
        )
        graph_recorder = GraphSnapshotRecorder(
            output_dir=graph_output_dir,
            interval=graph_snapshot_interval,
        )
        print(
            "[observability] Graph snapshots enabled "
            f"every {graph_snapshot_interval} world versions at {graph_output_dir}"
        )

    print(
        "[memory] Semantic memory enabled "
        f"(faiss, model={embedding_model.model_name}, "
        f"graph_history every {graph_history_interval_world_versions} versions, "
        f"max {graph_history_max_snapshots} snapshots)"
    )
    if offline:
        print("[pipeline] Offline mode enabled (cache-only model loading).")

    try:
        graph_snapshot_error_reported = False

        def _critical_event_description(snapshot: Mapping[str, Any]) -> str:
            recent_events = snapshot.get("recent_events")
            if isinstance(recent_events, list):
                for raw_event in reversed(recent_events):
                    if isinstance(raw_event, str) and raw_event.strip():
                        return raw_event.strip()

            world_version = int(snapshot.get("world_version", -1))
            frame_index = int(snapshot.get("timestamp", -1))
            return (
                "Critical world-state change detected "
                f"(world_version={world_version}, frame={frame_index})."
            )

        def _on_world_snapshot(snapshot: dict) -> None:
            nonlocal graph_snapshot_error_reported

            try:
                text_by_track_id: Mapping[int, str] | None = None
                try:
                    world_version = int(snapshot.get("world_version", 0))
                except (TypeError, ValueError):
                    world_version = 0

                if (
                    world_version > 0
                    and world_version % graph_history_store.save_interval_world_versions
                    == 0
                ):
                    text_by_track_id = crop_description_store.text_by_track_id()

                graph_history_store.maybe_save(
                    snapshot,
                    text_by_track_id=text_by_track_id,
                )
            except Exception as exc:
                failure_box["error"] = exc
                print(f"[memory] Failed to save graph history snapshot: {exc}")
                return

            if conversation_manager is not None and bool(
                snapshot.get("critical", False)
            ):
                world_version = int(snapshot.get("world_version", -1))
                frame_index = int(snapshot.get("timestamp", -1))
                conversation_manager.notify_critical_vision_event(
                    description=_critical_event_description(snapshot),
                    world_version=world_version,
                    frame_index=frame_index,
                )

            if graph_recorder is None:
                return

            try:
                saved_path = graph_recorder.maybe_record(snapshot)
            except Exception as exc:  # pragma: no cover - best effort observability
                if not graph_snapshot_error_reported:
                    print(
                        "[observability] Failed to write graph snapshot; "
                        f"continuing without graph exports. Reason: {exc}"
                    )
                    graph_snapshot_error_reported = True
                return

            if saved_path is not None and not interactive_user_input:
                print(f"[observability] Saved graph snapshot: {saved_path}")

        if frame_source_mode in {"auto", "api"}:
            candidate_api_server = EmbeddedApiServer(
                host=api_ingest_host,
                port=api_ingest_port,
            )
            try:
                websocket_url = candidate_api_server.start(
                    startup_timeout_seconds=api_ingest_startup_timeout_seconds
                )
            except Exception as exc:
                candidate_api_server.stop()
                if frame_source_mode == "auto":
                    print(
                        "[api] Embedded ingest server unavailable in auto mode; "
                        "continuing with local-only fallback. "
                        f"Reason: {exc}"
                    )
                else:
                    raise RuntimeError(
                        "Failed to start embedded API ingest server in strict api mode. "
                        "Use a free --api-ingest-port or stop conflicting servers."
                    ) from exc
            else:
                embedded_api_server = candidate_api_server
                print(f"[api] Embedded ingest server listening on {websocket_url}")

        def _perception_target() -> None:
            try:
                run_world_state_tracking_loop(
                    state=shared_state,
                    camera_index=camera_index,
                    model_path=model_path,
                    tracker_path=tracker_path,
                    display=display_perception,
                    on_snapshot=_on_world_snapshot,
                    stop_event=stop_event,
                    frame_source_mode=frame_source_mode,
                    api_stale_after_seconds=api_stale_after_seconds,
                    switch_to_api_after_consecutive=switch_to_api_after_consecutive,
                    switch_cooldown_seconds=switch_cooldown_seconds,
                )
            except Exception as exc:  # pragma: no cover - best effort crash forwarding
                failure_box["error"] = exc

        perception_thread = threading.Thread(target=_perception_target, daemon=True)
        perception_thread.start()

        # When the embedded API server is running, frames only arrive once a
        # remote client connects and starts streaming.  There is no way to know
        # how long that will take, so we disable the startup deadline.  The
        # perception loop already logs "Waiting for API frames..." periodically,
        # giving the operator feedback.  In local-only mode (no embedded server)
        # we keep the deadline so a missing camera fails fast.
        if (
            embedded_api_server is not None
            or frame_source_mode == "api"
            or perception_startup_timeout_seconds <= 0.0
        ):
            startup_deadline: float | None = None
            if embedded_api_server is not None:
                print(
                    "[pipeline] Waiting for frames. "
                    f"Stream to {embedded_api_server.websocket_url} to begin."
                )
        else:
            startup_deadline = time.monotonic() + perception_startup_timeout_seconds

        while True:
            if failure_box["error"] is not None:
                raise RuntimeError(
                    "Perception failed during startup. "
                    "Check camera index/permissions or run with a different input source."
                ) from failure_box["error"]

            if not perception_thread.is_alive() and not stop_event.is_set():
                raise RuntimeError("Perception loop exited unexpectedly during startup")

            if shared_state.snapshot().get("timestamp", 0) > 0:
                break

            if startup_deadline is not None and time.monotonic() >= startup_deadline:
                if embedded_api_server is not None:
                    source_hint = (
                        "Check local camera availability or stream frames to "
                        f"{embedded_api_server.websocket_url}."
                    )
                else:
                    source_hint = (
                        "Check camera availability or pass a valid --camera-index."
                    )

                raise RuntimeError(
                    "Perception did not produce frames before startup timeout. "
                    + source_hint
                )

            time.sleep(0.05)

        def _on_model_stdout(chunk: str) -> None:
            print(chunk, end="", flush=True)

        def _on_model_stderr(chunk: str) -> None:
            print(chunk, end="", flush=True, file=sys.stderr)

        print("[pipeline] Loading LLM inference engine...")
        inference = LlamaCppServerInference(
            LlamaCppServerConfig(
                quantization=quantization,
                port=llm_server_port,
                gpu_layers=llm_gpu_layers,
                cpu_only=llm_cpu_only,
                cpu_fallback=llm_cpu_fallback,
                offline=offline,
            )
        )
        inference.start(
            on_stderr=_on_model_stderr if stream_llm_output else None,
        )

        def _crop_description_target() -> None:
            active_inference = cast(LlamaCppServerInference, inference)
            try:
                run_crop_description_loop(
                    inference=active_inference,
                    crops_provider=shared_state.crop_snapshot,
                    object_type_provider=shared_state.object_types_snapshot,
                    store=crop_description_store,
                    stop_event=stop_event,
                    poll_interval_seconds=crop_description_poll_interval_seconds,
                    on_error=(
                        _on_model_stderr
                        if stream_llm_output and not interactive_user_input
                        else None
                    ),
                )
            except Exception as exc:  # pragma: no cover - best effort crash forwarding
                failure_box["error"] = exc

        crop_description_thread = threading.Thread(
            target=_crop_description_target,
            daemon=True,
        )
        crop_description_thread.start()

        tool_dispatcher = ToolDispatcher(
            retriever=memory_retriever,
            scene_state_provider=shared_state.snapshot,
            text_by_track_id_provider=crop_description_store.text_by_track_id,
        )
        agent_loop = AgentLoop(
            inference=inference,
            tool_dispatcher=tool_dispatcher,
            tool_schemas=TOOL_SCHEMAS,
        )
        latest_scene_state: dict[str, Any] | None = None

        def _ensure_background_health() -> None:
            if failure_box["error"] is not None:
                raise RuntimeError("Pipeline background loop failed") from failure_box[
                    "error"
                ]

            if not perception_thread.is_alive() and not stop_event.is_set():
                raise RuntimeError("Perception loop exited unexpectedly")

            if (
                crop_description_thread is not None
                and not crop_description_thread.is_alive()
                and not stop_event.is_set()
            ):
                raise RuntimeError("Crop description loop exited unexpectedly")

        def _index_new_descriptions(
            descriptions_snapshot: Mapping[str, Any],
            *,
            world_version: int,
        ) -> None:
            raw_items = descriptions_snapshot.get("items")
            if not isinstance(raw_items, list):
                return

            for raw_item in raw_items:
                if not isinstance(raw_item, Mapping):
                    continue

                raw_track_id = raw_item.get("id")
                raw_description = raw_item.get("description")
                if raw_track_id is None or not isinstance(raw_description, str):
                    continue

                if not _is_indexable_description(raw_description):
                    continue

                try:
                    track_id = int(raw_track_id)
                except (TypeError, ValueError):
                    continue

                object_type = raw_item.get("type")
                object_type_text = (
                    object_type.strip()
                    if isinstance(object_type, str) and object_type.strip()
                    else "object"
                )

                raw_described_at = raw_item.get("described_at")
                described_at = (
                    float(raw_described_at)
                    if isinstance(raw_described_at, (int, float))
                    else None
                )

                semantic_index.add(
                    track_id=track_id,
                    object_type=object_type_text,
                    description=raw_description,
                    indexed_world_version=world_version,
                    described_at=described_at,
                )

        memory_warning_state: dict[str, bool] = {}

        def _report_memory_warning(message: str) -> None:
            print(message)

        def _build_auxiliary_context(
            *,
            scene_state: Mapping[str, Any],
            query_text: str,
        ) -> dict[str, Any]:
            descriptions_snapshot = crop_description_store.snapshot()
            world_version = int(scene_state.get("world_version", 0))

            def _index_descriptions() -> None:
                _index_new_descriptions(
                    descriptions_snapshot,
                    world_version=world_version,
                )

            return _build_auxiliary_context_with_fallback(
                scene_state=scene_state,
                query_text=query_text,
                descriptions_snapshot=descriptions_snapshot,
                memory_retriever=memory_retriever,
                semantic_search_top_k=semantic_search_top_k,
                index_descriptions=_index_descriptions,
                warning_state=memory_warning_state,
                warning_reporter=_report_memory_warning,
            )

        def _state_provider() -> dict[str, Any]:
            nonlocal latest_scene_state
            _ensure_background_health()
            latest_scene_state = shared_state.snapshot()
            return latest_scene_state

        def _auxiliary_context_provider() -> dict[str, Any]:
            _ensure_background_health()
            scene_state = latest_scene_state or shared_state.snapshot()
            return _build_auxiliary_context(
                scene_state=scene_state,
                query_text=user_prompt,
            )

        if interactive_user_input:
            try:
                from conversation.manager import ConversationManager
                from ui.app import InteractiveConversationApp
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Interactive mode now requires the Textual UI dependency. "
                    "Run 'uv sync' and retry."
                ) from exc

            interactive_log_path = _resolve_interactive_log_path(interactive_log_file)
            interactive_logger = _build_interactive_logger(interactive_log_path)

            print(
                "[pipeline] Interactive conversation UI enabled. Press Ctrl+C to quit."
            )
            print(f"[pipeline] Interactive logs: {interactive_log_path}")
            interactive_logger.info("Interactive mode started")

            conversation_manager = ConversationManager(
                agent_loop=agent_loop,
                state_provider=_state_provider,
                auxiliary_context_builder=(
                    lambda scene_state, query_text: _build_auxiliary_context(
                        scene_state=scene_state,
                        query_text=query_text,
                    )
                ),
                history_window_turns=interactive_history_window_turns,
                max_steps=max_steps,
                stream_llm_output=stream_llm_output,
                logger=interactive_logger,
            )

            app = InteractiveConversationApp(
                conversation_manager=conversation_manager,
                state_provider=_state_provider,
                initial_user_prompt=user_prompt,
                status_refresh_interval_seconds=(
                    interactive_status_refresh_interval_seconds
                ),
                transcript_max_lines=interactive_transcript_max_lines,
                logger=interactive_logger,
            )
            try:
                app.run()
            except KeyboardInterrupt:
                interactive_logger.info("KeyboardInterrupt received; closing app")
                print("\n[pipeline] Interrupted. Closing interactive mode.")
            except Exception as exc:
                interactive_logger.exception(
                    "Interactive UI crashed",
                    exc_info=exc,
                )
                raise
            finally:
                conversation_manager.request_stop()
                interactive_logger.info("Interactive mode stopped")
        else:
            agent_loop.run(
                state_provider=_state_provider,
                user_prompt=user_prompt,
                poll_interval_seconds=poll_interval_seconds,
                require_initialized_state=False,
                max_steps=max_steps,
                auxiliary_context_provider=_auxiliary_context_provider,
                on_turn=(
                    _streaming_turn_handler
                    if stream_llm_output
                    else _default_turn_handler
                ),
                on_model_stdout=_on_model_stdout if stream_llm_output else None,
                on_model_stderr=_on_model_stderr if stream_llm_output else None,
            )
    finally:
        stop_event.set()
        if crop_description_thread is not None:
            crop_description_thread.join(timeout=5.0)
        if inference is not None:
            inference.stop()
        if perception_thread is not None:
            perception_thread.join(timeout=5.0)
        if embedded_api_server is not None:
            embedded_api_server.stop()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the perception -> local LLM MVP pipeline"
    )
    parser.add_argument(
        "--user-prompt",
        default="",
        help="Optional user instruction (or first query in interactive mode).",
    )
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        help="GGUF quantization (e.g. Q4_K_M for 4-bit or Q2_K for 2-bit)",
    )
    parser.add_argument(
        "--llm-server-port",
        type=int,
        default=8081,
        help="Port for persistent embedded llama-server process.",
    )
    parser.add_argument(
        "--llm-gpu-layers",
        default="auto",
        help="llama.cpp --n-gpu-layers value (default: auto).",
    )
    parser.add_argument(
        "--llm-cpu-only",
        action="store_true",
        help="Force llama-server to run without GPU offload.",
    )
    parser.add_argument(
        "--no-llm-cpu-fallback",
        action="store_true",
        help="Disable automatic CPU fallback if GPU startup fails.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cache-only model loading and disable Hugging Face downloads.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Agent polling interval in seconds.",
    )
    parser.add_argument(
        "--crop-description-poll-interval-seconds",
        type=float,
        default=0.5,
        help="Crop-to-text polling interval in seconds.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max number of agent turns",
    )
    parser.add_argument(
        "--interactive-user-input",
        action="store_true",
        help="Run the interactive Textual conversation UI.",
    )
    parser.add_argument(
        "--interactive-history-window-turns",
        type=int,
        default=6,
        help="Number of prior user/assistant turns to include in interactive mode.",
    )
    parser.add_argument(
        "--interactive-status-refresh-interval-seconds",
        type=float,
        default=1.0,
        help="How often the interactive status bar refreshes scene counters.",
    )
    parser.add_argument(
        "--interactive-transcript-max-lines",
        type=int,
        default=2000,
        help="Maximum transcript lines retained in the TUI before trimming.",
    )
    parser.add_argument(
        "--interactive-log-file",
        default="",
        help=(
            "Optional file path for interactive-mode logs. "
            "Defaults to artifacts/logs/interactive-<timestamp>.log"
        ),
    )
    parser.add_argument(
        "--semantic-search-top-k",
        type=int,
        default=5,
        help="Top-k semantic entity matches to inject per query.",
    )
    parser.add_argument(
        "--graph-history-interval-world-versions",
        type=int,
        default=50,
        help="Persist world graph history every N world versions.",
    )
    parser.add_argument(
        "--graph-history-max-snapshots",
        type=int,
        default=100,
        help="Maximum number of world graph history snapshots to retain.",
    )
    parser.add_argument(
        "--no-stream-llm-output",
        action="store_true",
        help="Disable live LLM stdout/stderr streaming and print one response per turn.",
    )
    parser.add_argument(
        "--perception-startup-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait for first perception frame before failing (<=0 disables)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index"
    )
    parser.add_argument(
        "--no-display-perception",
        action="store_true",
        help="Disable OpenCV display window for perception loop",
    )
    parser.add_argument("--model-path", default="yolo26x.pt", help="YOLO model path")
    parser.add_argument(
        "--tracker-path",
        default="perception/botsort.yaml",
        help="BoT-SORT tracker config path",
    )
    parser.add_argument(
        "--frame-source-mode",
        "--frame-source",
        choices=["auto", "api", "local"],
        default="auto",
        help="Frame source selection strategy for perception loop.",
    )
    parser.add_argument(
        "--api-stale-after-seconds",
        type=float,
        default=1.0,
        help="How old API frames can be before treated as stale.",
    )
    parser.add_argument(
        "--switch-to-api-after-consecutive",
        type=int,
        default=5,
        help="Fresh API frame streak required before switching local -> API.",
    )
    parser.add_argument(
        "--switch-cooldown-seconds",
        type=float,
        default=2.0,
        help="Minimum time between source switches in auto mode.",
    )
    parser.add_argument(
        "--api-ingest-host",
        default="0.0.0.0",
        help="Host for embedded API ingest server (used in auto/api modes).",
    )
    parser.add_argument(
        "--api-ingest-port",
        type=int,
        default=8000,
        help="Port for embedded API ingest server (used in auto/api modes).",
    )
    parser.add_argument(
        "--api-ingest-startup-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait for embedded API ingest server startup.",
    )
    parser.add_argument(
        "--graph-snapshot-interval",
        type=int,
        default=100,
        help="Save world graph PNG snapshots every N world versions (0 disables).",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        run_pipeline(
            user_prompt=args.user_prompt,
            quantization=args.quantization,
            poll_interval_seconds=args.poll_interval_seconds,
            crop_description_poll_interval_seconds=(
                args.crop_description_poll_interval_seconds
            ),
            max_steps=args.max_steps,
            interactive_user_input=args.interactive_user_input,
            interactive_history_window_turns=args.interactive_history_window_turns,
            interactive_status_refresh_interval_seconds=(
                args.interactive_status_refresh_interval_seconds
            ),
            interactive_transcript_max_lines=args.interactive_transcript_max_lines,
            interactive_log_file=args.interactive_log_file,
            semantic_search_top_k=args.semantic_search_top_k,
            stream_llm_output=not args.no_stream_llm_output,
            perception_startup_timeout_seconds=args.perception_startup_timeout_seconds,
            camera_index=args.camera_index,
            display_perception=not args.no_display_perception,
            model_path=args.model_path,
            tracker_path=args.tracker_path,
            frame_source_mode=cast(FrameSourceMode, args.frame_source_mode),
            api_stale_after_seconds=args.api_stale_after_seconds,
            switch_to_api_after_consecutive=args.switch_to_api_after_consecutive,
            switch_cooldown_seconds=args.switch_cooldown_seconds,
            api_ingest_host=args.api_ingest_host,
            api_ingest_port=args.api_ingest_port,
            api_ingest_startup_timeout_seconds=(
                args.api_ingest_startup_timeout_seconds
            ),
            llm_server_port=args.llm_server_port,
            llm_gpu_layers=args.llm_gpu_layers,
            llm_cpu_only=args.llm_cpu_only,
            llm_cpu_fallback=not args.no_llm_cpu_fallback,
            offline=args.offline,
            graph_history_interval_world_versions=(
                args.graph_history_interval_world_versions
            ),
            graph_history_max_snapshots=args.graph_history_max_snapshots,
            graph_snapshot_interval=args.graph_snapshot_interval,
        )
    except KeyboardInterrupt:
        print("\n[pipeline] Interrupted.")


if __name__ == "__main__":
    main()
