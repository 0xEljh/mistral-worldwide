from __future__ import annotations

import argparse
import threading
import time
from typing import Any, cast

from agent.inference import LlamaCppConfig, LlamaCppInference
from agent.loop import AgentLoop, AgentTurn
from perception.frame_provider import FrameSourceMode
from perception.yoloworldstate import WorldState, run_world_state_tracking_loop


def _default_turn_handler(turn: AgentTurn) -> None:
    print(
        f"[agent] world_version={turn.world_version} frame={turn.scene_timestamp}\n"
        f"{turn.response}\n"
    )


def run_pipeline(
    user_prompt: str = "",
    quantization: str = "Q4_K_M",
    poll_interval_seconds: float = 1.0,
    max_steps: int | None = None,
    perception_startup_timeout_seconds: float = 5.0,
    camera_index: int = 0,
    display_perception: bool = True,
    model_path: str = "yolo26x.pt",
    tracker_path: str = "perception/botsort.yaml",
    frame_source_mode: FrameSourceMode = "auto",
    api_stale_after_seconds: float = 1.0,
    switch_to_api_after_consecutive: int = 5,
    switch_cooldown_seconds: float = 2.0,
) -> None:
    shared_state = WorldState()
    stop_event = threading.Event()
    failure_box: dict[str, Exception | None] = {"error": None}

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

    try:
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

            if time.monotonic() >= startup_deadline:
                raise RuntimeError(
                    "Perception did not produce frames before startup timeout. "
                    "Check camera availability or pass a valid --camera-index."
                )

            time.sleep(0.05)

        inference = LlamaCppInference(
            LlamaCppConfig(
                quantization=quantization,
            )
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
            only_on_scene_change=True,
            require_initialized_state=False,
            max_steps=max_steps,
            on_turn=_default_turn_handler,
        )
    finally:
        stop_event.set()
        perception_thread.join(timeout=5.0)


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
        "--perception-startup-timeout-seconds",
        type=float,
        default=5.0,
        help="How long to wait for first perception frame before failing",
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
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        user_prompt=args.user_prompt,
        quantization=args.quantization,
        poll_interval_seconds=args.poll_interval_seconds,
        max_steps=args.max_steps,
        perception_startup_timeout_seconds=args.perception_startup_timeout_seconds,
        camera_index=args.camera_index,
        display_perception=not args.no_display_perception,
        model_path=args.model_path,
        tracker_path=args.tracker_path,
        frame_source_mode=cast(FrameSourceMode, args.frame_source_mode),
        api_stale_after_seconds=args.api_stale_after_seconds,
        switch_to_api_after_consecutive=args.switch_to_api_after_consecutive,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
    )


if __name__ == "__main__":
    main()
