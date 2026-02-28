#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required but not installed."
  echo "Install it from: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating project environment with uv..."
  uv sync --project "${ROOT_DIR}"
fi

if [ ! -x "${VENV_DIR}/bin/uvicorn" ]; then
  echo "Installing dependencies with uv..."
  uv sync --project "${ROOT_DIR}"
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

if [ "${RELOAD}" = "1" ]; then
  exec "${VENV_DIR}/bin/uvicorn" api.main:app --host "${HOST}" --port "${PORT}" --reload
fi

exec "${VENV_DIR}/bin/uvicorn" api.main:app --host "${HOST}" --port "${PORT}"
