#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "opencv-python",
#   "websockets",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import time

import cv2
import websockets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream webcam frames as JPEG bytes to a WebSocket endpoint."
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/video",
        help="WebSocket URL for video ingest.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="OpenCV camera device index.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Target send FPS.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="JPEG quality (1-100).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional capture width (0 keeps camera default).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional capture height (0 keeps camera default).",
    )
    return parser.parse_args()


async def stream_frames(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device {args.device}")

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.quality))]
    min_frame_interval = 1.0 / max(1.0, args.fps)

    sent = 0
    started_at = time.time()

    print(f"Connecting to {args.url}")
    print("Press Ctrl+C to stop streaming")

    try:
        async with websockets.connect(args.url, max_size=None) as websocket:
            while True:
                loop_started = time.time()
                ok, frame = cap.read()
                if not ok:
                    print("Failed to capture frame from camera")
                    await asyncio.sleep(0.05)
                    continue

                encoded_ok, encoded = cv2.imencode(".jpg", frame, jpeg_params)
                if not encoded_ok:
                    print("Failed to JPEG-encode frame")
                    continue

                await websocket.send(encoded.tobytes())
                sent += 1

                if sent % 30 == 0:
                    elapsed = max(1e-6, time.time() - started_at)
                    avg_fps = sent / elapsed
                    print(f"Sent {sent} frames ({avg_fps:.1f} fps average)")

                elapsed_loop = time.time() - loop_started
                if elapsed_loop < min_frame_interval:
                    await asyncio.sleep(min_frame_interval - elapsed_loop)
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(stream_frames(args))
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
