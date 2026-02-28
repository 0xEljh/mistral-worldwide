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

### 1) Verify `llama.cpp` binary

The pipeline expects `llama.cpp/llama-cli` to exist in the repo root.

If you use Nix, enter the dev shell first (this wires in the pinned `llama-cpp` build):

```bash
nix develop
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
- `--user-prompt "..."`: optional user instruction appended to scene-memory prompt.
- `--poll-interval-seconds 1.0`: how often the agent loop checks for updates.
- `--perception-startup-timeout-seconds 5.0`: fail fast if no camera frames arrive.
- `--camera-index 0`: select webcam index.
- `--no-display-perception`: run without the OpenCV display window.

### Notes

- Model repo: `unsloth/Ministral-3-3B-Instruct-2512-GGUF`.
- First run downloads GGUF weights to your Hugging Face cache.

### Troubleshooting

- **Camera cannot open (`can't open camera by index`)**: try a different camera index, for example `--camera-index 1`.
- **No frames before timeout**: increase `--perception-startup-timeout-seconds` and confirm camera permissions.
