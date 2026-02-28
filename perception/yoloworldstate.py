from __future__ import annotations

import argparse
import math
import threading
import time
import json
from collections import deque
from typing import Callable, cast

import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.utils.metrics import bbox_iou

try:
    from perception.frame_provider import (
        FrameSourceMode,
        FrameSourceSelector,
        frame_provider,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from frame_provider import FrameSourceMode, FrameSourceSelector, frame_provider

IOU_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 100
MOVEMENT_THRESHOLD = 8
MOTION_START_CONTINUOUS_FRAME_THRESHOLD = 4
MOTION_STOP_CONTINUOUS_FRAME_THRESHOLD = 4
MAX_EVENTS = 20
DISAPPEARANCE_THRESHOLD = 15
IDLE_SLEEP_SECONDS = 0.01
WAIT_LOG_INTERVAL_SECONDS = 2.0

DIRECTION_MAPPING = {
    -4: "left",
    -3: "bottom left",
    -2: "below",
    -1: "bottom right",
    0: "right",
    1: "upper right",
    2: "above",
    3: "upper left",
    4: "left",
}


class WorldState:
    """
    Scene memory generated from tracked detections.
    Attributes:
        objects: Dict[int, WorldObject]: A dictionary of objects keyed by tracking id from YOLO.
        relations: List[Relation]: A list of spatial relations between pairs of objects
        events: Deque[str]: Last MAX_EVENTS number of events
        version: int: How many updates to the WorldState
        frame_index: int: How many frames have been processed

    """

    def __init__(self, max_events: int = MAX_EVENTS):
        self.objects: dict[int, WorldObject] = {}
        self.relations: dict[
            set, Relation
        ] = {}  # key: frozenset({id_a, id_b}), value: relation object
        self.events: deque[str] = deque(maxlen=max_events)

        self.version = 0
        self.frame_index = 0
        self._lock = threading.Lock()

    def update_from_detections(self, names, boxes) -> None:
        with self._lock:
            self.frame_index += 1
            if boxes.id is None:
                return

            self.version += 1
            seen_ids = {int(item) for item in boxes.id.tolist()}
            num_detections = boxes.shape[0]

            for i in range(num_detections):
                track_id = int(boxes.id[i].item())
                xyxy = boxes.xyxy[i]
                center = (
                    (boxes.xyxy[i, 0] + boxes.xyxy[i, 2]).item() / 2,
                    (boxes.xyxy[i, 1] + boxes.xyxy[i, 3]).item() / 2,
                )
                class_name = names[boxes.cls[i].int().item()]
                conf = boxes.conf[i].item()

                if track_id in self.objects:
                    obj = self.objects[track_id]
                    prev_moving_state = obj.moving
                    obj.update(center, conf, self.frame_index, xyxy)

                    if not prev_moving_state and obj.moving:
                        direction = DIRECTION_MAPPING[
                            round(
                                4
                                * math.atan2(-obj.velocity[1], obj.velocity[0])
                                / math.pi
                            )
                        ]
                        self.events.append(f"{obj.type}_{track_id} moved {direction}")
                    elif prev_moving_state and not obj.moving:
                        self.events.append(f"{obj.type}_{track_id} stopped")
                else:
                    self.objects[track_id] = WorldObject(
                        track_id,
                        class_name,
                        center,
                        conf,
                        self.frame_index,
                        xyxy,
                    )
                    self.events.append(f"{class_name}_{track_id} appeared")

            for track_id, obj in self.objects.items():
                is_missing = track_id not in seen_ids
                disappearance_age = self.frame_index - obj.last_seen

                if (
                    is_missing
                    and disappearance_age > DISAPPEARANCE_THRESHOLD
                    and obj.visible
                ):
                    obj.mark_missing()
                    self.events.append(f"{obj.type}_{track_id} disappeared")

            self._update_relations_delta(seen_ids)

    def _update_relations_delta(self, seen_ids):
        objs = self.objects
        visible_ids = [tid for tid in seen_ids if objs[tid].visible]

        for i in range(len(visible_ids)):
            for j in range(i + 1, len(visible_ids)):
                id_a = visible_ids[i]
                id_b = visible_ids[j]

                a = objs[id_a]
                b = objs[id_b]

                ax, ay = a.center
                bx, by = b.center

                dx = ax - bx
                dy = by - ay  # y-axis flipped

                dist = math.dist(a.center, b.center)

                index = int(round(4 * math.atan2(dy, dx) / math.pi))
                index = max(-4, min(4, index))

                direction = DIRECTION_MAPPING[index]
                overlapping = bbox_iou(a.xyxy, b.xyxy) > IOU_THRESHOLD

                near = dist < DISTANCE_THRESHOLD

                key = frozenset({id_a, id_b})

                self.relations[key] = Relation(
                    subject_id=f"{a.type}_{a.track_id}",
                    direction=direction,
                    near=near,
                    overlapping=overlapping,
                    object_id=f"{b.type}_{b.track_id}",
                    last_updated=self.frame_index,
                )

    # Deprecated, useful for getting current graph later
    def _compute_relations(self) -> None:
        self.relations = []

        objs = list(self.objects.values())
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if not a.visible or not b.visible:
                    continue

                ax, ay = a.center
                bx, by = b.center

                dx = ax - bx
                dy = by - ay

                dist = math.dist(a.center, b.center)
                direction = DIRECTION_MAPPING[round(4 * math.atan2(dy, dx) / math.pi)]
                overlap = bool((bbox_iou(a.xyxy, b.xyxy) > IOU_THRESHOLD).item())

                id_a = f"{a.type}_{a.track_id}"
                id_b = f"{b.type}_{b.track_id}"

                self.relations.append(
                    Relation(
                        subject_id=id_a,
                        direction=direction,
                        near=dist < DISTANCE_THRESHOLD,
                        overlapping=overlap,
                        object_id=id_b,
                    )
                )

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "world_version": self.version,
                "timestamp": self.frame_index,
                "objects": [
                    obj.to_dict() for obj in self.objects.values() if obj.visible
                ],
                "relations": [rel.to_dict() for rel in self.relations.values()],
                "recent_events": list(self.events),
            }


class Relation:
    def __init__(
        self,
        subject_id: str,
        direction: str,
        near: bool,
        overlapping: bool,
        object_id: str,
        last_updated,
    ):
        self.subject_id = subject_id
        self.direction = direction
        self.near = near
        self.overlapping = overlapping
        self.object_id = object_id
        self.last_updated = last_updated

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject_id,
            "relation": (
                self.direction
                + ", "
                + ("near" if self.near else "far")
                + ", "
                + ("overlapping" if self.overlapping else "not overlapping")
            ),
            "object": self.object_id,
            "last_updated": self.last_updated,
        }


class WorldObject:
    def __init__(
        self,
        track_id: int,
        class_name: str,
        center,
        confidence: float,
        frame_idx: int,
        xyxy,
    ):
        self.track_id = track_id
        self.type = class_name

        self.xyxy = xyxy
        self.center = center
        self.prev_center = center
        self.velocity = (0.0, 0.0)

        self.visible = True
        self.moving = False
        self.motion_counter = 0
        self.stationary_counter = 0

        self.confidence = confidence
        self.first_seen = frame_idx
        self.last_seen = frame_idx

    def update(self, center, confidence: float, frame_idx: int, xyxy) -> None:
        vx = center[0] - self.center[0]
        vy = center[1] - self.center[1]

        self.xyxy = xyxy
        self.prev_center = self.center
        self.center = center
        self.velocity = (vx, vy)
        speed = math.hypot(vx,vy)

        if speed > MOVEMENT_THRESHOLD:
            self.motion_counter += 1
            self.stationary_counter = 0
        else:
            self.stationary_counter += 1
            self.motion_counter = 0
        
        if (not self.moving) and (self.motion_counter >= MOTION_START_CONTINUOUS_FRAME_THRESHOLD):
            self.moving = True
        elif (self.moving) and (self.stationary_counter >= MOTION_STOP_CONTINUOUS_FRAME_THRESHOLD):
            self.moving = False
        
        self.confidence = confidence
        self.last_seen = frame_idx
        self.visible = True

    def mark_missing(self) -> None:
        self.visible = False

    def to_dict(self) -> dict:
        return {
            "id": self.track_id,
            "type": self.type,
            "position": {"x": self.center[0], "y": self.center[1]},
            "velocity": {"x": self.velocity[0], "y": self.velocity[1]},
            "state": {
                "visible": self.visible,
                "moving": self.moving,
            },
            "confidence": self.confidence,
        }


def run_world_state_tracking_loop(
    state: WorldState | None = None,
    camera_index: int = 0,
    model_path: str = "yolo26x.pt",
    tracker_path: str = "perception/botsort.yaml",
    display: bool = True,
    on_snapshot: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    frame_source_mode: FrameSourceMode = "auto",
    api_stale_after_seconds: float = 1.0,
    switch_to_api_after_consecutive: int = 5,
    switch_cooldown_seconds: float = 2.0,
) -> WorldState:
    world_state = state or WorldState()

    selector = FrameSourceSelector(
        mode=frame_source_mode,
        api_stale_after_seconds=api_stale_after_seconds,
        switch_to_api_after_consecutive=switch_to_api_after_consecutive,
        switch_cooldown_seconds=switch_cooldown_seconds,
    )

    local_cap: cv2.VideoCapture | None = None
    if frame_source_mode in {"auto", "local"}:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            local_cap = cap
            if display:
                frame_width = int(local_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(frame_width, frame_height)
        elif frame_source_mode == "local":
            cap.release()
            raise RuntimeError(
                "Cannot open video device. Check camera index or connection."
            )
        else:
            cap.release()
            print(
                "Local camera unavailable. Waiting for API stream frames in auto mode."
            )

    if frame_source_mode == "api":
        print("Frame source mode: api (strict)")
    elif frame_source_mode == "local":
        print("Frame source mode: local (strict)")
    else:
        print("Frame source mode: auto (prefer API, fallback local)")

    model = YOLO(model_path)
    last_wait_log_at = 0.0
    display_enabled = display
    display_error_reported = False

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            now = time.time()
            packet = frame_provider.get_frame()
            decision = selector.select_source(
                packet,
                local_available=local_cap is not None,
                now=now,
            )

            if decision.switched:
                reason = decision.reason or "unknown"
                print(f"Frame source switched to {decision.source} ({reason})")

            frame: np.ndarray | None = None
            wait_reason: str | None = None
            if decision.source == "api":
                if selector.is_api_packet_fresh(packet, now=now):
                    frame = packet.frame
                else:
                    wait_reason = "api"
            else:
                if local_cap is None:
                    wait_reason = "local"
                else:
                    has_frame, local_frame = local_cap.read()
                    if has_frame:
                        frame = local_frame
                    else:
                        wait_reason = "local"

            if frame is None:
                if (now - last_wait_log_at) >= WAIT_LOG_INTERVAL_SECONDS:
                    if wait_reason == "api":
                        frame_age = frame_provider.frame_age_seconds(now=now)
                        if frame_age is None:
                            print("Waiting for API frames...")
                        else:
                            print(
                                f"Waiting for fresh API frame (latest age: {frame_age:.2f}s)"
                            )
                    else:
                        print("Waiting for local camera frame...")
                    last_wait_log_at = now

                if display_enabled:
                    try:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        if not display_error_reported:
                            print(
                                "OpenCV display backend unavailable. Disabling display; "
                                "use --no-display to suppress this warning."
                            )
                            display_error_reported = True
                        display_enabled = False

                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                verbose=False,
            )
            if not results:
                continue

            world_state.update_from_detections(results[0].names, results[0].boxes)

            if on_snapshot is not None:
                on_snapshot(world_state.snapshot())

            if display_enabled:
                try:
                    annotated_frame = results[0].plot()
                    cv2.imshow("YOLO26 Tracking", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    if not display_error_reported:
                        print(
                            "OpenCV display backend unavailable. Disabling display; "
                            "use --no-display to suppress this warning."
                        )
                        display_error_reported = True
                    display_enabled = False
    finally:
        if local_cap is not None:
            local_cap.release()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    return world_state


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO world-state tracking with configurable frame source mode."
    )
    parser.add_argument(
        "--camera-index", type=int, default=0, help="OpenCV camera index"
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
        help="Frame source mode: auto prefers API, api/local are strict.",
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
        help="Fresh API streak required before switching local -> API.",
    )
    parser.add_argument(
        "--switch-cooldown-seconds",
        type=float,
        default=2.0,
        help="Minimum time between source switches in auto mode.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window.",
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="output json to be sent to LLM for debugging",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.frame_source_mode == "api":
        print(
            "[api] Standalone perception CLI does not start embedded ingestion. "
            "Use 'python -m orchestration.pipeline' for embedded API ingest."
        )

    final_state = run_world_state_tracking_loop(
        camera_index=args.camera_index,
        model_path=args.model_path,
        tracker_path=args.tracker_path,
        display=not args.no_display,
        frame_source_mode=cast(FrameSourceMode, args.frame_source_mode),
        api_stale_after_seconds=args.api_stale_after_seconds,
        switch_to_api_after_consecutive=args.switch_to_api_after_consecutive,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
    )

    print(final_state.snapshot())
    if args.output_json:
        with open("output.json", "w") as json_file:
            json.dump(final_state.snapshot(), json_file, indent=4)


if __name__ == "__main__":
    main()
