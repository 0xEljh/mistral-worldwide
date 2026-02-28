from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


FrameSourceMode = Literal["auto", "api", "local"]
FrameSource = Literal["api", "local"]


@dataclass(frozen=True)
class FramePacket:
    frame: Optional[np.ndarray]
    timestamp: float


@dataclass(frozen=True)
class SourceDecision:
    source: FrameSource
    switched: bool
    reason: Optional[str]


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


class FrameSourceSelector:
    """Selects a stable frame source with API-preferred auto fallback.

    Modes:
      - "api": strict API-only.
      - "local": strict local-only.
      - "auto": prefer API when fresh, fallback to local when API is stale.
    """

    def __init__(
        self,
        mode: FrameSourceMode = "auto",
        api_stale_after_seconds: float = 1.0,
        switch_to_api_after_consecutive: int = 5,
        switch_cooldown_seconds: float = 2.0,
    ) -> None:
        if mode not in {"auto", "api", "local"}:
            raise ValueError("mode must be one of: auto, api, local")
        if api_stale_after_seconds <= 0.0:
            raise ValueError("api_stale_after_seconds must be > 0")
        if switch_to_api_after_consecutive < 1:
            raise ValueError("switch_to_api_after_consecutive must be >= 1")
        if switch_cooldown_seconds < 0.0:
            raise ValueError("switch_cooldown_seconds must be >= 0")

        self._mode: FrameSourceMode = mode
        self._api_stale_after_seconds = api_stale_after_seconds
        self._switch_to_api_after_consecutive = switch_to_api_after_consecutive
        self._switch_cooldown_seconds = switch_cooldown_seconds

        self._active_source: Optional[FrameSource] = None
        self._last_switch_timestamp = 0.0
        self._api_fresh_streak = 0
        self._last_counted_api_timestamp: Optional[float] = None

    @property
    def mode(self) -> FrameSourceMode:
        return self._mode

    @property
    def active_source(self) -> Optional[FrameSource]:
        return self._active_source

    def is_api_packet_fresh(
        self, packet: FramePacket, now: Optional[float] = None
    ) -> bool:
        if packet.frame is None:
            return False
        current_time = time.time() if now is None else now
        return (current_time - packet.timestamp) <= self._api_stale_after_seconds

    def select_source(
        self,
        packet: FramePacket,
        *,
        local_available: bool,
        now: Optional[float] = None,
    ) -> SourceDecision:
        current_time = time.time() if now is None else now

        if self._mode == "api":
            return self._switch_to("api", current_time, "forced_api")

        if self._mode == "local":
            return self._switch_to("local", current_time, "forced_local")

        api_fresh = self.is_api_packet_fresh(packet, now=current_time)

        if self._active_source is None:
            if api_fresh:
                self._api_fresh_streak = self._switch_to_api_after_consecutive
                self._last_counted_api_timestamp = packet.timestamp
                return self._switch_to("api", current_time, "initial_api")

            self._api_fresh_streak = 0
            if local_available:
                return self._switch_to("local", current_time, "initial_local")

            return SourceDecision(source="api", switched=False, reason="waiting_api")

        if self._active_source == "api":
            if api_fresh:
                self._api_fresh_streak = self._switch_to_api_after_consecutive
                self._last_counted_api_timestamp = packet.timestamp
                return SourceDecision(source="api", switched=False, reason=None)

            self._api_fresh_streak = 0
            if local_available:
                return self._switch_to("local", current_time, "api_stale")

            return SourceDecision(source="api", switched=False, reason="api_stale")

        if api_fresh:
            if self._last_counted_api_timestamp != packet.timestamp:
                self._api_fresh_streak += 1
                self._last_counted_api_timestamp = packet.timestamp
        else:
            self._api_fresh_streak = 0

        if (
            api_fresh
            and self._api_fresh_streak >= self._switch_to_api_after_consecutive
            and self._cooldown_elapsed(current_time)
        ):
            return self._switch_to("api", current_time, "api_recovered")

        return SourceDecision(source="local", switched=False, reason=None)

    def _cooldown_elapsed(self, current_time: float) -> bool:
        if self._last_switch_timestamp <= 0.0:
            return True
        return (
            current_time - self._last_switch_timestamp
        ) >= self._switch_cooldown_seconds

    def _switch_to(
        self, source: FrameSource, current_time: float, reason: str
    ) -> SourceDecision:
        if self._active_source == source:
            return SourceDecision(source=source, switched=False, reason=None)

        self._active_source = source
        self._last_switch_timestamp = current_time
        return SourceDecision(source=source, switched=True, reason=reason)


frame_provider = FrameProvider()
