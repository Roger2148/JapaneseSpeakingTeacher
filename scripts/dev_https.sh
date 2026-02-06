#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8443}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-5173}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-japanese_teacher}"
CERT_DIR="${CERT_DIR:-${ROOT_DIR}/deploy/certs}"
CERT_FILE="${CERT_FILE:-${CERT_DIR}/dev-cert.pem}"
KEY_FILE="${KEY_FILE:-${CERT_DIR}/dev-key.pem}"
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
  echo "[dev-https] npm not found. Install Node/npm first."
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[dev-https] conda not found. Install conda or run API manually."
  exit 1
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "[dev-https] openssl not found."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "[dev-https] conda env '${CONDA_ENV_NAME}' not found."
  echo "[dev-https] Create it first or set CONDA_ENV_NAME to an existing env."
  exit 1
fi

if [ ! -d "${ROOT_DIR}/apps/web/node_modules" ]; then
  echo "[dev-https] installing web dependencies..."
  (cd "${ROOT_DIR}/apps/web" && npm install)
fi

mkdir -p "${CERT_DIR}"
if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
  echo "[dev-https] generating self-signed cert for localhost + ${LAN_HOST}"
  if openssl req \
    -x509 \
    -newkey rsa:2048 \
    -sha256 \
    -days 365 \
    -nodes \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LAN_HOST}"; then
    true
  else
    SAN_FILE="${CERT_DIR}/openssl-san.cnf"
    cat > "${SAN_FILE}" <<EOF
[req]
distinguished_name=req_dn
x509_extensions=v3_req
prompt=no

[req_dn]
CN=localhost

[v3_req]
subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LAN_HOST}
EOF
    openssl req \
      -x509 \
      -newkey rsa:2048 \
      -sha256 \
      -days 365 \
      -nodes \
      -keyout "${KEY_FILE}" \
      -out "${CERT_FILE}" \
      -config "${SAN_FILE}" \
      -extensions v3_req
  fi
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

echo "[dev-https] starting API on https://${LAN_HOST}:${API_PORT}"
ALLOWED_HOSTS="${ALLOWED_HOSTS_VALUE}" AUTH_COOKIE_SECURE=true conda run -n "${CONDA_ENV_NAME}" \
  uvicorn main:app \
  --app-dir "${ROOT_DIR}/apps/api" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --reload \
  --reload-dir "${ROOT_DIR}/apps/api" \
  --ssl-keyfile "${KEY_FILE}" \
  --ssl-certfile "${CERT_FILE}" &
API_PID="$!"

echo "[dev-https] starting web on https://${LAN_HOST}:${WEB_PORT}"
(
  cd "${ROOT_DIR}/apps/web"
  VITE_API_BASE_URL="https://${LAN_HOST}:${API_PORT}" \
  VITE_DEV_HTTPS_KEY="${KEY_FILE}" \
  VITE_DEV_HTTPS_CERT="${CERT_FILE}" \
    npm run dev -- --host "${WEB_HOST}" --port "${WEB_PORT}" --strictPort
) &
WEB_PID="$!"

echo "[dev-https] running. Press Ctrl+C to stop both."
echo "[dev-https] API: https://${LAN_HOST}:${API_PORT}"
echo "[dev-https] Web: https://${LAN_HOST}:${WEB_PORT}"
echo "[dev-https] note: self-signed cert; trust cert on phone to unlock mic."

while true; do
  if ! kill -0 "${API_PID}" >/dev/null 2>&1; then
    echo "[dev-https] API process exited; stopping both."
    break
  fi
  if ! kill -0 "${WEB_PID}" >/dev/null 2>&1; then
    echo "[dev-https] Web process exited; stopping both."
    break
  fi
  sleep 1
done
