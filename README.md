# Japanese Speaking Teacher

LAN/Tailscale accessible Japanese speaking practice app.

You speak (Japanese or English), the app transcribes your voice, generates a tutor reply with a local LLM, then speaks the reply back.

## What It Does

- Browser chat UI (desktop + mobile) with mic recording.
- STT: local `faster-whisper`.
- LLM: local Ollama model (default `qwen3:8b`).
- TTS: local speech synthesis (system voice / optional engines).
- Username login (no password), saved conversation history, export options.
- Optional live transcript overlay during recording.
- Topic-based conversation starters with random topic suggestions.

## Current Scope

- Milestones M1-M10 are implemented.
- This is an alpha project focused on local/self-hosted usage.
- Auth is intentionally lightweight (username only), suitable for private/home usage.

## Architecture

- Frontend: React + TypeScript + Vite (`apps/web`)
- Backend: FastAPI + WebSocket (`apps/api`)
- Audio flow:
  1. Browser captures mic audio chunks.
  2. Chunks stream to backend over WebSocket.
  3. Backend transcribes -> generates reply -> synthesizes audio.
  4. Frontend displays messages and supports playback.

## Repository Layout

- `apps/web`: web frontend
- `apps/api`: FastAPI backend, auth/history, STT/LLM/TTS pipeline
- `apps/api/prompt_profile.json`: tutor prompt behavior preset
- `apps/api/data`: runtime state (ignored in git)
- `apps/api/generated_audio`: generated TTS files (ignored in git)
- `scripts/dev.sh`: one-command HTTP dev launcher
- `scripts/dev_https.sh`: one-command HTTPS launcher (LAN + Tailscale aware)
- `docs/milestones`: milestone checklists

## Prerequisites

- macOS or Linux
- Conda
- Node.js + npm
- Ollama
- Optional: Tailscale (for remote private network access)

## 1) Environment Setup

From repo root:

```bash
conda env create -f environment.yml || conda env update -f environment.yml --prune
conda activate japanese_teacher
cd apps/web && npm install && cd ../..
ollama pull qwen3:8b
```

Start Ollama (if not already running):

```bash
ollama serve
```

## 2) Run (Recommended)

### Fast local HTTP

```bash
./scripts/dev.sh
```

- Web: `http://<your-host>:5173`
- API: `http://<your-host>:8000`

### HTTPS (required for reliable mobile mic)

```bash
./scripts/dev_https.sh
```

- Web: `https://<your-host>:5173`
- API: `https://<your-host>:8443`
- Generates self-signed certs in `deploy/certs/` if needed.
- Script includes LAN and Tailscale IPs in cert SAN when available.

## 3) Manual Run

### API

```bash
conda activate japanese_teacher
uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload --reload-dir apps/api
```

### Web

```bash
cd apps/web
npm run dev -- --host 0.0.0.0 --port 5173
```

## Configuration

Use `.env.example` as reference for backend/frontend environment variables.

Important keys:

- `LLM_MODEL` (default `qwen3:8b`)
- `LLM_OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`)
- `STT_MODEL_NAME` (default `small`)
- `TTS_PROVIDER`
- `AUTH_COOKIE_SECURE`
- `CORS_ALLOW_ORIGINS`, `CORS_ALLOW_ORIGIN_REGEX`
- `ALLOWED_HOSTS`
- `VITE_API_BASE_URL` (optional)
- `VITE_API_PORT` (optional; useful for dynamic host with fixed API port)

## Save/History Semantics

- Session is unsaved by default.
- User can save explicitly via Save modal:
  1. Audio package export (`.zip`)
  2. Text export (`.txt`)
  3. Save on server history
- History is for conversation continuity, not implicit long-term tutor memory.

## Prompt and Tutor Behavior Tuning

- Edit `apps/api/prompt_profile.json` to change:
  - conversational tone
  - correction strictness
  - follow-up question style
- Override profile path with `LLM_PROMPT_PROFILE_PATH`.

## Topic Suggestions

- Topic DB path: `apps/api/data/topics.db`
- On login/new chat, backend returns a starter with 3 random topics.
- If user requests a specific topic, tutor steers conversation to it.

## Troubleshooting

### Login works on LAN IP but fails on Tailscale IP

- Run with `./scripts/dev_https.sh` (it now handles Tailscale host + cert SAN).
- Reopen the Tailscale URL and trust the certificate for that host.
- Ensure browser has accepted the cert; otherwise requests fail with certificate errors.

### iPhone mic says insecure context

- Use HTTPS URL, not HTTP.
- Ensure certificate is trusted on the phone.

### LLM unavailable

- Verify Ollama is running: `ollama ps`
- Pull model if missing: `ollama pull qwen3:8b`

## Validation Commands

```bash
cd apps/web && npm run check && npm run build
python -m py_compile apps/api/main.py
```

Milestone test checklists are in `docs/milestones/`.

## Security Notes

- Current login is username-only and intentionally simple.
- Do not expose this alpha build directly to the public internet as-is.
- For internet deployment, add real auth, TLS cert management, and stronger session controls.
