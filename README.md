# mistral-worldwide

## Quick setup with uv

If you don't have `uv` installed yet, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install project dependencies with:

```bash
uv sync
```

## Run the orchestration pipeline

The MVP pipeline connects:

- perception (`WorldState` scene memory),
- prompt building (`agent/prompt_builder.py`),
- local GGUF inference (`llama.cpp` + Ministral 3 3B),
- and loop orchestration (`orchestration/pipeline.py`).

### 1) Install and verify `llama.cpp` binaries

The pipeline runs local inference through `llama.cpp` and expects these repo-local
paths to exist:

- `llama.cpp/llama-cli`
- `llama.cpp/llama-server`

If you use Nix, this is handled for you (the shell creates the symlinks):

```bash
nix develop
```

The Nix shell pins a CUDA-enabled `llama.cpp` build, so inference runs with GPU
offload by default when NVIDIA drivers are available.

If you are not using Nix, install `llama.cpp` with one of the official methods:

- Install docs: <https://github.com/ggml-org/llama.cpp/blob/master/docs/install.md>
- Build docs: <https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md>

Common install commands:

```bash
# macOS / Linux (Homebrew)
brew install llama.cpp

# Windows (Winget)
winget install llama.cpp
```

After install, wire binaries into this repo (Linux/macOS):

```bash
mkdir -p llama.cpp
ln -sf "$(command -v llama-cli)" llama.cpp/llama-cli
ln -sf "$(command -v llama-server)" llama.cpp/llama-server
```

Verify both binaries are available:

```bash
./llama.cpp/llama-cli --help >/dev/null
./llama.cpp/llama-server --help >/dev/null
```

### 2) Install dependencies

```bash
uv sync
```

### 3) Run a short smoke test (3 turns)

```bash
uv run python -m orchestration.pipeline --quantization Q4_K_M --max-steps 3
```

### 4) Run continuously

```bash
uv run python -m orchestration.pipeline --quantization Q4_K_M
```

Press `q` in the OpenCV window to stop.

### Useful flags

- `--quantization Q4_K_M`: default 4-bit quantized model.
- `--quantization Q2_K`: smaller 2-bit quantized model.
- `--llm-gpu-layers auto`: use llama.cpp GPU offload defaults (recommended).
- `--llm-cpu-only`: force CPU inference (no GPU offload).
- `--no-llm-cpu-fallback`: fail fast if GPU startup fails instead of retrying on CPU.
- `--user-prompt "..."`: optional user instruction appended to scene-memory prompt.
- `--poll-interval-seconds 1.0`: how often the agent loop checks for updates.
- `--perception-startup-timeout-seconds 5.0`: fail fast if no camera frames arrive.
- `--camera-index 0`: select webcam index.
- `--no-display-perception`: run without the OpenCV display window.
- `--frame-source-mode auto|api|local`: source selection (`auto` prefers API, falls back local).
- `--api-ingest-host` / `--api-ingest-port`: embedded websocket ingest bind address for API frames.

### API frame ingest (same-process)

When pipeline frame mode is `auto` or `api`, `orchestration.pipeline` now starts an
embedded FastAPI websocket ingest server in the same process as perception. This is
required so websocket frames and YOLO share the same in-memory `frame_provider`.

Stream frames into the running pipeline with:

```bash
uv run python send_webcam_ws.py --url ws://127.0.0.1:8000/ws/video
```

In `local` mode, no embedded ingest server is started and the pipeline reads directly
from OpenCV camera input.

### Notes

- Model repo: `unsloth/Ministral-3-3B-Instruct-2512-GGUF`.
- First run downloads GGUF weights to your Hugging Face cache.

### Troubleshooting

- **Camera cannot open (`can't open camera by index`)**: try a different camera index, for example `--camera-index 1`.
- **No frames before timeout**: increase `--perception-startup-timeout-seconds` and confirm camera permissions.
- **Embedded ingest server fails to start**: another process is likely using the port. Stop the conflicting server or pass `--api-ingest-port <free-port>` and stream to that websocket URL.
- **`llama-cli` / `llama-server` not found**: ensure `llama.cpp/llama-cli` and `llama.cpp/llama-server` exist in the repo root. Re-run the symlink step from section 1.
- **GPU not used for LLM**: run `./llama.cpp/llama-server --list-devices`; if only `BLAS` appears, rebuild your shell with `nix develop` so the CUDA-enabled `llama.cpp` package is used.
