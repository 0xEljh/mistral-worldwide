from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent.prompt_builder import PromptBundle

_DEFAULT_LLAMA_CLI_PATH = (
    Path(__file__).resolve().parents[1] / "llama.cpp" / "llama-cli"
)

_QUANTIZATION_ALIASES = {
    "2bit": "Q2_K",
    "4bit": "Q4_K_M",
    "q2_k": "Q2_K",
    "q2_k_l": "Q2_K_L",
    "q4_k_m": "Q4_K_M",
    "q4_k_s": "Q4_K_S",
}


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


@dataclass(frozen=True)
class LlamaCppConfig:
    hf_repo: str = "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
    quantization: str = "Q4_K_M"
    max_tokens: int = 256
    context_size: int = 8192
    temperature: float = 0.2
    threads: int | None = None
    timeout_seconds: float = 120.0
    binary_path: Path = _DEFAULT_LLAMA_CLI_PATH


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

    def generate(self, prompt: PromptBundle) -> str:
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

        if self._config.threads is not None:
            command.extend(["--threads", str(self._config.threads)])

        env = os.environ.copy()
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=env,
            timeout=self._config.timeout_seconds,
        )

        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()

        if completed.returncode != 0:
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
                f"(exit={completed.returncode}, model={self.model_ref}). "
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
