from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment]

DEFAULT_CROP_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a vision assistant that describes a single tracked object crop. "
    "Keep descriptions grounded in visible evidence, concise, and useful for scene reasoning."
)
DEFAULT_CROP_DESCRIPTION_TASK = (
    "Describe the main object in this crop in 1-2 sentences. "
    "Mention clear visual attributes (color, pose, shape, material, approximate size) "
    "and obvious state if visible. If details are unclear, say that briefly."
)

LogHandler = Callable[[str], None]
CropValue = Any
CropProvider = Callable[[], Mapping[int, CropValue]]
ObjectTypeProvider = Callable[[], Mapping[int, str]]

CROP_DESCRIPTION_JPEG_QUALITY = 85


class CropInferenceClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_stdout: LogHandler | None = None,
        on_stderr: LogHandler | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class CropDescription:
    track_id: int
    object_type: str
    description: str
    described_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.track_id,
            "type": self.object_type,
            "description": self.description,
            "described_at": self.described_at,
        }


class CropDescriptionStore:
    def __init__(self) -> None:
        self._descriptions: dict[int, CropDescription] = {}
        self._lock = threading.Lock()

    def update(self, entry: CropDescription) -> None:
        with self._lock:
            self._descriptions[entry.track_id] = entry

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = [
                entry.to_dict()
                for entry in sorted(
                    self._descriptions.values(), key=lambda item: item.track_id
                )
            ]
            text_by_track_id = {
                entry.track_id: entry.description
                for entry in self._descriptions.values()
            }

        return {
            "count": len(entries),
            "items": entries,
            "text_by_track_id": text_by_track_id,
        }

    def text_by_track_id(self) -> dict[int, str]:
        with self._lock:
            return {
                track_id: entry.description
                for track_id, entry in self._descriptions.items()
            }


def describe_crop(
    inference: CropInferenceClient,
    crop_bytes: bytes,
    *,
    object_type: str,
    mime_type: str | None = None,
    max_tokens: int = 96,
    temperature: float = 0.1,
) -> str:
    resolved_mime_type = mime_type or _infer_image_mime_type(crop_bytes)
    image_base64 = base64.b64encode(crop_bytes).decode("ascii")
    image_data_uri = f"data:{resolved_mime_type};base64,{image_base64}"

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": DEFAULT_CROP_DESCRIPTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": DEFAULT_CROP_DESCRIPTION_TASK,
                },
                {
                    "type": "text",
                    "text": f"Detector label hint: {object_type}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri,
                    },
                },
            ],
        },
    ]

    description = inference.complete(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
    ).strip()
    if not description:
        raise RuntimeError("Empty crop description response")
    return description


def run_crop_description_loop(
    *,
    inference: CropInferenceClient,
    crops_provider: CropProvider,
    object_type_provider: ObjectTypeProvider,
    store: CropDescriptionStore,
    stop_event: threading.Event,
    poll_interval_seconds: float = 0.5,
    max_tokens: int = 96,
    temperature: float = 0.1,
    on_error: LogHandler | None = None,
) -> None:
    described_track_ids: set[int] = set()
    sleep_seconds = max(poll_interval_seconds, 0.01)

    while not stop_event.is_set():
        try:
            crops = dict(crops_provider())
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            _emit_error(
                on_error,
                f"[crop-describer] Failed to snapshot crops: {exc}\n",
            )
            stop_event.wait(timeout=sleep_seconds)
            continue

        new_track_ids = sorted(
            track_id for track_id in crops if track_id not in described_track_ids
        )
        if not new_track_ids:
            stop_event.wait(timeout=sleep_seconds)
            continue

        try:
            object_types = dict(object_type_provider())
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            _emit_error(
                on_error,
                f"[crop-describer] Failed to snapshot object types: {exc}\n",
            )
            object_types = {}

        for track_id in new_track_ids:
            if stop_event.is_set():
                break

            crop_value = crops.get(track_id)
            object_type = str(object_types.get(track_id, "object"))
            crop_bytes = _normalize_crop_to_bytes(crop_value)

            if crop_bytes is None:
                store.update(
                    CropDescription(
                        track_id=track_id,
                        object_type=object_type,
                        description="Description unavailable (invalid crop data).",
                        described_at=time.monotonic(),
                    )
                )
                described_track_ids.add(track_id)
                _emit_error(
                    on_error,
                    (
                        "[crop-describer] Skipping "
                        f"track_id={track_id}: invalid crop payload.\n"
                    ),
                )
                continue

            try:
                description = describe_crop(
                    inference,
                    crop_bytes,
                    object_type=object_type,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                description = "Description unavailable due to inference error."
                _emit_error(
                    on_error,
                    (
                        "[crop-describer] Failed to describe "
                        f"track_id={track_id}: {exc}\n"
                    ),
                )

            store.update(
                CropDescription(
                    track_id=track_id,
                    object_type=object_type,
                    description=description,
                    described_at=time.monotonic(),
                )
            )
            described_track_ids.add(track_id)


def _emit_error(handler: LogHandler | None, message: str) -> None:
    if handler is None:
        return
    try:
        handler(message)
    except Exception:
        return


def _infer_image_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/jpeg"


def _normalize_crop_to_bytes(crop_value: Any) -> bytes | None:
    if isinstance(crop_value, (bytes, bytearray)):
        if len(crop_value) == 0:
            return None
        return bytes(crop_value)

    if cv2 is None:
        return None

    if not hasattr(crop_value, "shape") or not hasattr(crop_value, "size"):
        return None

    if int(getattr(crop_value, "size")) == 0:
        return None

    try:
        success, encoded = cv2.imencode(
            ".jpg",
            crop_value,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(CROP_DESCRIPTION_JPEG_QUALITY)],
        )
    except Exception:
        return None

    if not success:
        return None
    return encoded.tobytes()
