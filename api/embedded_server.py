from __future__ import annotations

import threading
import time

import uvicorn

from api.main import app


class EmbeddedApiServer:
    """Runs the FastAPI websocket ingest server in-process.

    This keeps websocket frame ingestion and perception in the same Python process,
    so both share the same in-memory frame provider singleton.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "warning",
    ) -> None:
        if not host:
            raise ValueError("host must be non-empty")
        if port < 1 or port > 65535:
            raise ValueError("port must be in [1, 65535]")

        self._host = host
        self._port = port
        self._server = uvicorn.Server(
            uvicorn.Config(
                app=app,
                host=host,
                port=port,
                log_level=log_level,
                access_log=False,
            )
        )
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    @property
    def websocket_url(self) -> str:
        public_host = self._host
        if public_host in {"0.0.0.0", "::"}:
            public_host = "127.0.0.1"
        return f"ws://{public_host}:{self._port}/ws/video"

    def start(self, startup_timeout_seconds: float = 5.0) -> str:
        if startup_timeout_seconds <= 0.0:
            raise ValueError("startup_timeout_seconds must be > 0")
        if self._thread is not None and self._thread.is_alive():
            return self.websocket_url

        self._failure = None
        self._thread = threading.Thread(
            target=self._run,
            name="embedded-api-server",
            daemon=True,
        )
        self._thread.start()

        startup_deadline = time.monotonic() + startup_timeout_seconds
        while True:
            if self._failure is not None:
                raise RuntimeError(
                    "Embedded API server crashed during startup"
                ) from self._failure

            if self._server.started:
                return self.websocket_url

            if not self._thread.is_alive():
                raise RuntimeError(
                    "Embedded API server exited before startup. "
                    "The configured host/port may already be in use."
                )

            if time.monotonic() >= startup_deadline:
                raise RuntimeError("Timed out waiting for embedded API server startup")

            time.sleep(0.05)

    def stop(self, shutdown_timeout_seconds: float = 5.0) -> None:
        if shutdown_timeout_seconds < 0.0:
            raise ValueError("shutdown_timeout_seconds must be >= 0")

        if self._thread is None:
            return

        self._server.should_exit = True
        self._thread.join(timeout=shutdown_timeout_seconds)

    def _run(self) -> None:
        try:
            self._server.run()
        except BaseException as exc:  # pragma: no cover - startup/runtime forwarding
            self._failure = exc
