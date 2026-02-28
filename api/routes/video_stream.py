from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from perception.frame_provider import frame_provider

router = APIRouter()

FrameQueueItem = Union[Tuple[bytes, float], object]
_SENTINEL = object()


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    encoded = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _push_latest(queue: asyncio.Queue[FrameQueueItem], item: FrameQueueItem) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


async def _consume_frames(queue: asyncio.Queue[FrameQueueItem]) -> None:
    while True:
        item = await queue.get()
        if item is _SENTINEL:
            return

        data, received_timestamp = item
        frame = await asyncio.to_thread(_decode_jpeg, data)
        if frame is None:
            continue

        frame_provider.update(frame, received_timestamp)


@router.websocket("/ws/video")
async def video_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    queue: asyncio.Queue[FrameQueueItem] = asyncio.Queue(maxsize=1)
    consumer_task = asyncio.create_task(_consume_frames(queue))

    try:
        while True:
            payload = await websocket.receive_bytes()
            received_timestamp = time.time()
            _push_latest(queue, (payload, received_timestamp))
    except WebSocketDisconnect:
        pass
    finally:
        _push_latest(queue, _SENTINEL)
        await consumer_task
