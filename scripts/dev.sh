#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-5173}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-japanese_teacher}"
LAN_HOST="${LAN_HOST:-}"

if [ -z "${LAN_HOST}" ]; then
  LAN_HOST="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi
if [ -z "${LAN_HOST}" ]; then
  LAN_HOST="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi
if [ -z "${LAN_HOST}" ]; then
  LAN_HOST="127.0.0.1"
fi

ALLOWED_HOSTS_VALUE="${ALLOWED_HOSTS:-localhost,127.0.0.1,${LAN_HOST}}"

if ! command -v npm >/dev/null 2>&1; then
  echo "[dev] npm not found. Install Node/npm first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[dev] conda not found. Install conda or run API manually."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "[dev] conda env '${CONDA_ENV_NAME}' not found."
  echo "[dev] Create it first or set CONDA_ENV_NAME to an existing env."
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[dev] warning: ollama command not found. LLM chat will fail until installed."
else
  if ! ollama ps >/dev/null 2>&1; then
    echo "[dev] warning: ollama server does not look active."
    echo "[dev] start it in another terminal: ollama serve"
  fi
fi

if [ ! -d "${ROOT_DIR}/apps/web/node_modules" ]; then
  echo "[dev] installing web dependencies..."
  (cd "${ROOT_DIR}/apps/web" && npm install)
fi

API_PID=""
WEB_PID=""

cleanup() {
  if [ -n "${API_PID}" ] && kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
  fi
  if [ -n "${WEB_PID}" ] && kill -0 "${WEB_PID}" >/dev/null 2>&1; then
    kill "${WEB_PID}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "[dev] starting API on http://${API_HOST}:${API_PORT}"
ALLOWED_HOSTS="${ALLOWED_HOSTS_VALUE}" conda run -n "${CONDA_ENV_NAME}" \
  uvicorn main:app \
  --app-dir "${ROOT_DIR}/apps/api" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --reload \
  --reload-dir "${ROOT_DIR}/apps/api" &
API_PID="$!"

echo "[dev] starting web on http://${WEB_HOST}:${WEB_PORT}"
(
  cd "${ROOT_DIR}/apps/web"
  npm run dev -- --host "${WEB_HOST}" --port "${WEB_PORT}"
) &
WEB_PID="$!"

echo "[dev] running. Press Ctrl+C to stop both servers."
echo "[dev] API: http://${API_HOST}:${API_PORT}"
echo "[dev] Web: http://${WEB_HOST}:${WEB_PORT}"
echo "[dev] LAN host: ${LAN_HOST}"

while true; do
  if ! kill -0 "${API_PID}" >/dev/null 2>&1; then
    echo "[dev] API process exited; stopping both."
    break
  fi
  if ! kill -0 "${WEB_PID}" >/dev/null 2>&1; then
    echo "[dev] Web process exited; stopping both."
    break
  fi
  sleep 1
done
