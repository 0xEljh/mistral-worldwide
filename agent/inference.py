from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from agent.prompt_builder import PromptBundle

_DEFAULT_LLAMA_CLI_PATH = (
    Path(__file__).resolve().parents[1] / "llama.cpp" / "llama-cli"
)
_DEFAULT_LLAMA_SERVER_PATH = (
    Path(__file__).resolve().parents[1] / "llama.cpp" / "llama-server"
)

_QUANTIZATION_ALIASES = {
    "2bit": "Q2_K",
    "4bit": "Q4_K_M",
    "q2_k": "Q2_K",
    "q2_k_l": "Q2_K_L",
    "q4_k_m": "Q4_K_M",
    "q4_k_s": "Q4_K_S",
}

StreamChunkHandler = Callable[[str], None]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: str | None = None


def _extract_assistant_response(raw_stdout: str, prompt_text: str) -> str:
    response_region = raw_stdout

    prompt_anchor = f"> {prompt_text}"
    prompt_anchor_index = raw_stdout.rfind(prompt_anchor)
    if prompt_anchor_index != -1:
        response_region = raw_stdout[prompt_anchor_index + len(prompt_anchor) :]

    response_region = response_region.lstrip("\r\n")

    footer_markers = [
        "\n[ Prompt:",
        "\nExiting...",
        "\nllama_memory_breakdown_print:",
    ]

    cutoff = len(response_region)
    for marker in footer_markers:
        marker_index = response_region.find(marker)
        if marker_index != -1:
            cutoff = min(cutoff, marker_index)

    return response_region[:cutoff].strip()


def _normalize_quantization(quantization: str) -> str:
    normalized = quantization.strip()
    if not normalized:
        raise ValueError("quantization cannot be empty")

    lowered = normalized.lower()
    if lowered in _QUANTIZATION_ALIASES:
        return _QUANTIZATION_ALIASES[lowered]

    return normalized.upper()


def _drain_text_stream(
    stream: TextIO,
    sink: list[str],
    on_chunk: StreamChunkHandler | None,
) -> None:
    while True:
        chunk = stream.read(1)
        if chunk == "":
            return

        sink.append(chunk)

        if on_chunk is None:
            continue

        try:
            on_chunk(chunk)
        except Exception:
            on_chunk = None


@dataclass(frozen=True)
class LlamaCppConfig:
    hf_repo: str = "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
    quantization: str = "Q4_K_M"
    max_tokens: int = 256
    context_size: int = 8192
    temperature: float = 0.2
    threads: int | None = None
    timeout_seconds: float = 120.0
    offline: bool = False
    binary_path: Path = _DEFAULT_LLAMA_CLI_PATH


@dataclass(frozen=True)
class LlamaCppServerConfig:
    hf_repo: str = "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
    quantization: str = "Q4_K_M"
    max_tokens: int = 256
    context_size: int = 8192
    temperature: float = 0.2
    threads: int | None = None
    timeout_seconds: float = 120.0
    startup_timeout_seconds: float = 120.0
    host: str = "127.0.0.1"
    port: int = 8081
    gpu_layers: str = "auto"
    cpu_only: bool = False
    cpu_fallback: bool = True
    offline: bool = False
    binary_path: Path = _DEFAULT_LLAMA_SERVER_PATH


class LlamaCppInference:
    """One-shot local inference through llama.cpp's llama-cli.

    This backend keeps implementation dependencies minimal and works directly with
    GGUF models from Hugging Face repos.
    """

    def __init__(self, config: LlamaCppConfig | None = None) -> None:
        self._config = config or LlamaCppConfig()
        self._quantization = _normalize_quantization(self._config.quantization)

        if not self._config.binary_path.exists():
            raise FileNotFoundError(
                f"llama-cli not found at {self._config.binary_path}. "
                "Build or symlink llama.cpp first."
            )

    @property
    def model_ref(self) -> str:
        return f"{self._config.hf_repo}:{self._quantization}"

    def generate(
        self,
        prompt: PromptBundle,
        on_stdout: StreamChunkHandler | None = None,
        on_stderr: StreamChunkHandler | None = None,
    ) -> str:
        command = [
            str(self._config.binary_path),
            "--hf-repo",
            self.model_ref,
            "--single-turn",
            "--conversation",
            "--simple-io",
            "--no-display-prompt",
            "--ctx-size",
            str(self._config.context_size),
            "--n-predict",
            str(self._config.max_tokens),
            "--temp",
            str(self._config.temperature),
            "--system-prompt",
            prompt.system_prompt,
            "--prompt",
            prompt.user_prompt,
        ]

        if self._config.offline:
            command.append("--offline")

        if self._config.threads is not None:
            command.extend(["--threads", str(self._config.threads)])

        env = os.environ.copy()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("llama-cli did not expose stdout/stderr pipes")

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        stdout_thread = threading.Thread(
            target=_drain_text_stream,
            args=(process.stdout, stdout_chunks, on_stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_drain_text_stream,
            args=(process.stderr, stderr_chunks, on_stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            return_code = process.wait(timeout=self._config.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            return_code = process.wait()
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            stdout = "".join(stdout_chunks).strip()
            stderr = "".join(stderr_chunks).strip()
            stderr_tail = stderr[-4000:] if len(stderr) > 4000 else stderr
            stdout_tail = stdout[-1000:] if len(stdout) > 1000 else stdout
            raise RuntimeError(
                "llama-cli inference timed out "
                f"(timeout={self._config.timeout_seconds}s, model={self.model_ref}, "
                f"exit={return_code}). "
                f"stderr={stderr_tail or '<empty>'} stdout={stdout_tail or '<empty>'}"
            ) from exc
        finally:
            stdout_thread.join(timeout=1.0)
            stderr_thread.join(timeout=1.0)
            process.stdout.close()
            process.stderr.close()

        stdout = "".join(stdout_chunks).strip()
        stderr = "".join(stderr_chunks).strip()

        if return_code != 0:
            if "unknown model architecture: 'mistral3'" in stderr:
                raise RuntimeError(
                    "llama-cli cannot load this model because your llama.cpp build "
                    "does not support the 'mistral3' architecture yet. "
                    "Update llama.cpp to a newer version and try again. "
                    f"model={self.model_ref}"
                )

            stderr_tail = stderr[-4000:] if len(stderr) > 4000 else stderr
            stdout_tail = stdout[-1000:] if len(stdout) > 1000 else stdout
            raise RuntimeError(
                "llama-cli inference failed "
                f"(exit={return_code}, model={self.model_ref}). "
                f"stderr={stderr_tail or '<empty>'} stdout={stdout_tail or '<empty>'}"
            )

        response = _extract_assistant_response(stdout, prompt.user_prompt)
        if not response:
            stdout_tail = stdout[-1000:] if len(stdout) > 1000 else stdout
            raise RuntimeError(
                "llama-cli returned an empty response after parsing. "
                f"stdout={stdout_tail or '<empty>'}"
            )

        return response


class LlamaCppServerInference:
    """Persistent local inference through llama.cpp's llama-server."""

    def __init__(self, config: LlamaCppServerConfig | None = None) -> None:
        self._config = config or LlamaCppServerConfig()
        self._quantization = _normalize_quantization(self._config.quantization)
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_chunks: list[str] = []
        self._stderr_chunks: list[str] = []
        self._started_in_cpu_mode = False

        if not self._config.binary_path.exists():
            raise FileNotFoundError(
                f"llama-server not found at {self._config.binary_path}. "
                "Build or symlink llama.cpp first."
            )

    @property
    def model_ref(self) -> str:
        return f"{self._config.hf_repo}:{self._quantization}"

    @property
    def base_url(self) -> str:
        return f"http://{self._config.host}:{self._config.port}"

    def start(self, on_stderr: StreamChunkHandler | None = None) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            return

        if process is not None:
            self.stop()

        if self._config.cpu_only:
            self._emit_log(
                on_stderr,
                "[llm] Starting llama-server in CPU-only mode (--device none).\n",
            )
            self._start_once(cpu_only=True, on_stderr=on_stderr)
            return

        self._emit_log(
            on_stderr,
            "[llm] Starting llama-server with GPU offload enabled.\n",
        )

        try:
            self._start_once(cpu_only=False, on_stderr=on_stderr)
        except Exception as gpu_exc:
            if not self._config.cpu_fallback:
                raise

            self._emit_log(
                on_stderr,
                "[llm] GPU startup failed; retrying in CPU-only mode.\n",
            )
            try:
                self._start_once(cpu_only=True, on_stderr=on_stderr)
            except Exception as cpu_exc:
                raise RuntimeError(
                    "Failed to start llama-server with GPU and CPU fallback. "
                    f"gpu_error={gpu_exc} cpu_error={cpu_exc}"
                ) from cpu_exc

    def _start_once(
        self,
        *,
        cpu_only: bool,
        on_stderr: StreamChunkHandler | None,
    ) -> None:
        command = self._build_server_command(cpu_only=cpu_only)

        env = os.environ.copy()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait(timeout=1.0)
            raise RuntimeError("llama-server did not expose stdout/stderr pipes")

        self._stdout_chunks = []
        self._stderr_chunks = []
        self._process = process
        self._started_in_cpu_mode = cpu_only
        self._stdout_thread = threading.Thread(
            target=_drain_text_stream,
            args=(process.stdout, self._stdout_chunks, on_stderr),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_drain_text_stream,
            args=(process.stderr, self._stderr_chunks, on_stderr),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise

    def _build_server_command(self, *, cpu_only: bool) -> list[str]:
        command = [
            str(self._config.binary_path),
            "--hf-repo",
            self.model_ref,
            "--ctx-size",
            str(self._config.context_size),
            "--host",
            self._config.host,
            "--port",
            str(self._config.port),
        ]

        if self._config.offline:
            command.append("--offline")

        if cpu_only:
            command.extend(["--device", "none", "--n-gpu-layers", "0"])
        else:
            gpu_layers = self._config.gpu_layers.strip()
            if not gpu_layers:
                gpu_layers = "auto"
            command.extend(["--n-gpu-layers", gpu_layers])

        if self._config.threads is not None:
            command.extend(["--threads", str(self._config.threads)])

        return command

    def stop(self) -> None:
        process = self._process
        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        finally:
            self._join_stream_threads()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            self._process = None
            self._started_in_cpu_mode = False

    def generate(
        self,
        prompt: PromptBundle,
        on_stdout: StreamChunkHandler | None = None,
        on_stderr: StreamChunkHandler | None = None,
    ) -> str:
        return self._request_chat_completion(
            messages=prompt.messages,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )

    def generate_with_tools(
        self,
        prompt: PromptBundle,
        *,
        tools: list[dict[str, Any]],
        on_stderr: StreamChunkHandler | None = None,
    ) -> ChatCompletionResult:
        return self.complete_with_tools(
            prompt.messages,
            tools=tools,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            on_stderr=on_stderr,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_stdout: StreamChunkHandler | None = None,
        on_stderr: StreamChunkHandler | None = None,
    ) -> str:
        resolved_max_tokens = (
            self._config.max_tokens if max_tokens is None else max_tokens
        )
        resolved_temperature = (
            self._config.temperature if temperature is None else temperature
        )

        if resolved_max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if resolved_temperature < 0.0:
            raise ValueError("temperature must be >= 0")

        return self._request_chat_completion(
            messages=messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_stderr: StreamChunkHandler | None = None,
    ) -> ChatCompletionResult:
        if not tools:
            raise ValueError("tools cannot be empty")

        resolved_max_tokens = (
            self._config.max_tokens if max_tokens is None else max_tokens
        )
        resolved_temperature = (
            self._config.temperature if temperature is None else temperature
        )

        if resolved_max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if resolved_temperature < 0.0:
            raise ValueError("temperature must be >= 0")

        payload = self._request_chat_completion_payload(
            messages=messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
            tools=tools,
            on_stderr=on_stderr,
        )
        return self._parse_chat_completion_result(payload)

    def _request_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        on_stdout: StreamChunkHandler | None,
        on_stderr: StreamChunkHandler | None,
    ) -> str:
        self._ensure_running()

        payload = {
            "model": self.model_ref,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": on_stdout is not None,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                if on_stdout is None:
                    return self._read_non_streaming_response(response)
                return self._read_streaming_response(response, on_stdout)
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            if on_stderr is not None and error_body:
                try:
                    on_stderr(error_body)
                except Exception:
                    on_stderr = None
            raise RuntimeError(
                "llama-server chat completion request failed "
                f"(status={exc.code}, model={self.model_ref}). "
                f"body={error_body or '<empty>'}"
            ) from exc
        except urllib.error.URLError as exc:
            if on_stderr is not None:
                try:
                    on_stderr(str(exc))
                except Exception:
                    on_stderr = None
            raise RuntimeError(
                f"Failed to reach llama-server for chat completion at {self.base_url}."
            ) from exc

    def _request_chat_completion_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        tools: list[dict[str, Any]],
        on_stderr: StreamChunkHandler | None,
    ) -> dict[str, Any]:
        self._ensure_running()

        payload = {
            "model": self.model_ref,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "tools": tools,
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                raw_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace").strip()
            if on_stderr is not None and error_body:
                try:
                    on_stderr(error_body)
                except Exception:
                    on_stderr = None
            raise RuntimeError(
                "llama-server chat completion request failed "
                f"(status={exc.code}, model={self.model_ref}). "
                f"body={error_body or '<empty>'}"
            ) from exc
        except urllib.error.URLError as exc:
            if on_stderr is not None:
                try:
                    on_stderr(str(exc))
                except Exception:
                    on_stderr = None
            raise RuntimeError(
                f"Failed to reach llama-server for chat completion at {self.base_url}."
            ) from exc

        try:
            parsed_payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            body_tail = raw_body[-1000:] if len(raw_body) > 1000 else raw_body
            raise RuntimeError(
                "llama-server returned invalid JSON for chat completion. "
                f"body={body_tail or '<empty>'}"
            ) from exc

        if not isinstance(parsed_payload, dict):
            raise RuntimeError("llama-server chat completion payload is not an object")
        return parsed_payload

    @staticmethod
    def _parse_chat_completion_result(payload: dict[str, Any]) -> ChatCompletionResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama-server chat completion payload has no choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("llama-server chat completion choice is not an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("llama-server chat completion choice has no message")

        finish_reason = first_choice.get("finish_reason")
        finish_reason_text = finish_reason if isinstance(finish_reason, str) else None

        parsed_tool_calls: list[ToolCall] = []
        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            for index, raw_tool_call in enumerate(raw_tool_calls):
                if not isinstance(raw_tool_call, dict):
                    continue

                function = raw_tool_call.get("function")
                if not isinstance(function, dict):
                    continue

                name = function.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue

                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    arguments_text = json.dumps(arguments, ensure_ascii=True)
                elif isinstance(arguments, str):
                    arguments_text = arguments
                else:
                    arguments_text = "{}"

                call_id = raw_tool_call.get("id")
                if isinstance(call_id, str) and call_id.strip():
                    call_id_text = call_id.strip()
                else:
                    call_id_text = f"tool_call_{index}"

                parsed_tool_calls.append(
                    ToolCall(
                        id=call_id_text,
                        name=name.strip(),
                        arguments=arguments_text,
                    )
                )

        content = message.get("content")
        content_text = (
            content.strip() if isinstance(content, str) and content.strip() else None
        )

        if parsed_tool_calls:
            return ChatCompletionResult(
                content=content_text,
                tool_calls=parsed_tool_calls,
                finish_reason=finish_reason_text,
            )

        if content_text is None:
            raise RuntimeError("llama-server chat completion returned empty content")

        return ChatCompletionResult(
            content=content_text,
            tool_calls=[],
            finish_reason=finish_reason_text,
        )

    def __enter__(self) -> LlamaCppServerInference:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        del exc_type, exc, tb
        self.stop()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + max(self._config.startup_timeout_seconds, 0.0)
        last_error: Exception | None = None
        health_request = urllib.request.Request(
            url=f"{self.base_url}/health",
            method="GET",
        )

        while True:
            self._ensure_running()
            try:
                with urllib.request.urlopen(health_request, timeout=2.0) as response:
                    if 200 <= response.status < 300:
                        return
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                last_error = exc

            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)

        stderr = "".join(self._stderr_chunks).strip()
        stdout = "".join(self._stdout_chunks).strip()
        stderr_tail = stderr[-4000:] if len(stderr) > 4000 else stderr
        stdout_tail = stdout[-1000:] if len(stdout) > 1000 else stdout
        backend_mode = "cpu" if self._started_in_cpu_mode else "gpu"
        raise RuntimeError(
            "llama-server did not become ready before startup timeout "
            f"(timeout={self._config.startup_timeout_seconds}s, model={self.model_ref}, "
            f"backend={backend_mode}). "
            f"last_error={last_error!r} "
            f"stderr={stderr_tail or '<empty>'} stdout={stdout_tail or '<empty>'}"
        )

    def _ensure_running(self) -> None:
        process = self._process
        if process is None:
            raise RuntimeError("llama-server is not running. Call start() first.")

        return_code = process.poll()
        if return_code is None:
            return

        stderr = "".join(self._stderr_chunks).strip()
        stdout = "".join(self._stdout_chunks).strip()
        stderr_tail = stderr[-4000:] if len(stderr) > 4000 else stderr
        stdout_tail = stdout[-1000:] if len(stdout) > 1000 else stdout
        backend_mode = "cpu" if self._started_in_cpu_mode else "gpu"
        raise RuntimeError(
            "llama-server process is not running "
            f"(exit={return_code}, model={self.model_ref}, backend={backend_mode}). "
            f"stderr={stderr_tail or '<empty>'} stdout={stdout_tail or '<empty>'}"
        )

    def _join_stream_threads(self) -> None:
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        self._stdout_thread = None
        self._stderr_thread = None

    def _emit_log(
        self,
        handler: StreamChunkHandler | None,
        message: str,
    ) -> None:
        if handler is None:
            return
        try:
            handler(message)
        except Exception:
            return

    def _read_non_streaming_response(self, response: Any) -> str:
        raw_body = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            body_tail = raw_body[-1000:] if len(raw_body) > 1000 else raw_body
            raise RuntimeError(
                "llama-server returned invalid JSON for chat completion. "
                f"body={body_tail or '<empty>'}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("llama-server chat completion payload is not an object")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama-server chat completion payload has no choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("llama-server chat completion choice is not an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("llama-server chat completion choice has no message")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("llama-server chat completion returned empty content")

        return content.strip()

    def _read_streaming_response(
        self,
        response: Any,
        on_stdout: StreamChunkHandler,
    ) -> str:
        chunks: list[str] = []
        chunk_handler: StreamChunkHandler | None = on_stdout

        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue

            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break

            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                continue

            piece = ""
            delta = first_choice.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    piece = delta_content

            if not piece:
                message = first_choice.get("message")
                if isinstance(message, dict):
                    message_content = message.get("content")
                    if isinstance(message_content, str):
                        piece = message_content

            if not piece:
                continue

            chunks.append(piece)
            if chunk_handler is not None:
                try:
                    chunk_handler(piece)
                except Exception:
                    chunk_handler = None

        response_text = "".join(chunks).strip()
        if not response_text:
            raise RuntimeError("llama-server returned an empty streaming response")
        return response_text
