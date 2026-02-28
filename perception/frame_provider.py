from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame: Optional[np.ndarray]
    timestamp: float


class FrameProvider:
    """Lock-free latest-frame provider for inference pipelines.

    The provider stores only the most recent frame. Reads and writes are atomic
    tuple/dataclass reference swaps under the CPython GIL.
    """

    def __init__(self) -> None:
        self._current: FramePacket = FramePacket(frame=None, timestamp=0.0)

    def update(self, frame: np.ndarray, timestamp: float) -> None:
        self._current = FramePacket(frame=frame, timestamp=timestamp)

    def get_frame(self) -> FramePacket:
        return self._current

    def get_frame_copy(self) -> FramePacket:
        packet = self._current
        if packet.frame is None:
            return packet
        return FramePacket(frame=packet.frame.copy(), timestamp=packet.timestamp)

    def has_frame(self) -> bool:
        return self._current.frame is not None

    def frame_age_seconds(self, now: Optional[float] = None) -> Optional[float]:
        packet = self._current
        if packet.frame is None:
            return None
        current_time = time.time() if now is None else now
        return max(0.0, current_time - packet.timestamp)


frame_provider = FrameProvider()
