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
import json
import time
import urllib.error
import urllib.parse
import urllib.request

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
    parser.add_argument(
        "--debug-check",
        action="store_true",
        help="Run /debug/frame-status and /debug/save-frame checks after a few frames.",
    )
    parser.add_argument(
        "--debug-base-url",
        default="",
        help="HTTP base URL for debug endpoints (defaults to ws URL host).",
    )
    parser.add_argument(
        "--debug-check-after",
        type=int,
        default=10,
        help="Run debug checks after this many frames are sent.",
    )
    return parser.parse_args()


def _default_debug_base_url(ws_url: str) -> str:
    parsed = urllib.parse.urlparse(ws_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("--url must use ws:// or wss://")

    http_scheme = "https" if parsed.scheme == "wss" else "http"
    return urllib.parse.urlunparse((http_scheme, parsed.netloc, "", "", "", ""))


def _http_json_request(url: str, method: str) -> dict:
    request = urllib.request.Request(url=url, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


async def _run_debug_checks(base_url: str) -> None:
    status_url = urllib.parse.urljoin(base_url + "/", "debug/frame-status")
    save_url = urllib.parse.urljoin(base_url + "/", "debug/save-frame")

    print(f"[debug] Checking ingestion: GET {status_url}")
    try:
        status_payload = await asyncio.to_thread(_http_json_request, status_url, "GET")
    except urllib.error.URLError as exc:
        print(f"[debug] Ingestion check failed: {exc}")
        return

    has_frame = bool(status_payload.get("has_frame"))
    age_seconds = status_payload.get("age_seconds")
    shape = status_payload.get("shape")
    if has_frame:
        print(
            f"[debug] Ingestion check OK: has_frame=true age_seconds={age_seconds} shape={shape}"
        )
    else:
        print("[debug] Ingestion check NOT READY: has_frame=false")
        return

    print(f"[debug] Checking save-frame: POST {save_url}")
    try:
        save_payload = await asyncio.to_thread(_http_json_request, save_url, "POST")
    except urllib.error.URLError as exc:
        print(f"[debug] Save-frame check failed: {exc}")
        return

    saved = bool(save_payload.get("saved"))
    saved_path = save_payload.get("path")
    saved_shape = save_payload.get("shape")
    if saved:
        print(
            f"[debug] Save-frame check OK: saved=true path={saved_path} shape={saved_shape}"
        )
    else:
        print("[debug] Save-frame check FAILED: saved flag not true")


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
    debug_checked = False
    debug_base_url = args.debug_base_url.strip() or _default_debug_base_url(args.url)

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

                if (
                    args.debug_check
                    and not debug_checked
                    and sent >= args.debug_check_after
                ):
                    await _run_debug_checks(debug_base_url)
                    debug_checked = True

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
