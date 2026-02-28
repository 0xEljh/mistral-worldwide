from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, cast

from api.embedded_server import EmbeddedApiServer
from agent.inference import LlamaCppServerConfig, LlamaCppServerInference
from agent.loop import AgentLoop, AgentTurn
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


def run_pipeline(
    user_prompt: str = "",
    quantization: str = "Q4_K_M",
    poll_interval_seconds: float = 1.0,
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
) -> None:
    shared_state = WorldState()
    stop_event = threading.Event()
    failure_box: dict[str, Exception | None] = {"error": None}
    embedded_api_server: EmbeddedApiServer | None = None
    perception_thread: threading.Thread | None = None
    inference: LlamaCppServerInference | None = None

    try:
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
            )
        )
        inference.start(
            on_stderr=_on_model_stderr if stream_llm_output else None,
        )
        agent_loop = AgentLoop(inference=inference)

        def _state_provider() -> dict[str, Any]:
            if failure_box["error"] is not None:
                raise RuntimeError("Perception loop failed") from failure_box["error"]

            if not perception_thread.is_alive() and not stop_event.is_set():
                raise RuntimeError("Perception loop exited unexpectedly")

            return shared_state.snapshot()

        agent_loop.run(
            state_provider=_state_provider,
            user_prompt=user_prompt,
            poll_interval_seconds=poll_interval_seconds,
            require_initialized_state=False,
            max_steps=max_steps,
            on_turn=(
                _streaming_turn_handler if stream_llm_output else _default_turn_handler
            ),
            on_model_stdout=_on_model_stdout if stream_llm_output else None,
            on_model_stderr=_on_model_stderr if stream_llm_output else None,
        )
    finally:
        stop_event.set()
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
    parser.add_argument("--user-prompt", default="", help="Optional user instruction")
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
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Agent polling interval",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max number of agent turns",
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
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        user_prompt=args.user_prompt,
        quantization=args.quantization,
        poll_interval_seconds=args.poll_interval_seconds,
        max_steps=args.max_steps,
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
        api_ingest_startup_timeout_seconds=args.api_ingest_startup_timeout_seconds,
        llm_server_port=args.llm_server_port,
        llm_gpu_layers=args.llm_gpu_layers,
        llm_cpu_only=args.llm_cpu_only,
        llm_cpu_fallback=not args.no_llm_cpu_fallback,
    )


if __name__ == "__main__":
    main()
