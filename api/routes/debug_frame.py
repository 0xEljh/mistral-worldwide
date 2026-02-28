from __future__ import annotations

import asyncio
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, HTTPException

from perception.frame_provider import frame_provider

router = APIRouter()

_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "e2e"
_LATEST_FRAME_PATH = _ARTIFACTS_DIR / "latest.jpg"


@router.get("/debug/frame-status")
async def frame_status() -> dict:
    packet = frame_provider.get_frame()
    has_frame = packet.frame is not None

    if not has_frame:
        return {
            "has_frame": False,
            "timestamp": 0.0,
            "age_seconds": None,
            "shape": None,
        }

    return {
        "has_frame": True,
        "timestamp": packet.timestamp,
        "age_seconds": max(0.0, time.time() - packet.timestamp),
        "shape": list(packet.frame.shape),
    }


@router.post("/debug/save-frame")
async def save_frame() -> dict:
    packet = frame_provider.get_frame_copy()

    if packet.frame is None:
        raise HTTPException(status_code=404, detail="No frame available yet")

    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    wrote_file = await asyncio.to_thread(
        cv2.imwrite, str(_LATEST_FRAME_PATH), packet.frame
    )

    if not wrote_file:
        raise HTTPException(status_code=500, detail="Failed to save frame")

    return {
        "saved": True,
        "path": str(_LATEST_FRAME_PATH),
        "timestamp": packet.timestamp,
        "shape": list(packet.frame.shape),
    }
