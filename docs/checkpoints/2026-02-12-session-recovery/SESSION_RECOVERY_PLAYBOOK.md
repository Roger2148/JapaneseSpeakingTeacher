# Japanese Speaking Teacher - Session Recovery Playbook

## 1. Purpose

This document is a **high-detail checkpoint** so the project can be resumed by another agent/session without relying on chat history.

Use this playbook when:
- the current agent session is lost/corrupted,
- a new contributor needs full context quickly,
- you need a deterministic restart procedure for dev/testing.

---

## 2. Snapshot Metadata

- Repository: `JapaneseSpeakingTeacher`
- Branch at checkpoint: `main`
- Commit at checkpoint: `51bf695dd7f0f570bf2db729cb5c48655de33b37`
- Latest commit summary: `Add IDE configs, demo assets, update README`
- Working tree status when this checkpoint was generated: clean (`git status --short` = 0 lines)

---

## 3. Product Goal (Current Alpha)

A LAN/Tailscale-accessible Japanese speaking tutor web app:
1. User speaks in Japanese or English.
2. Browser captures audio and sends it to backend.
3. Backend transcribes speech (STT), generates tutor reply (LLM), and synthesizes audio (TTS).
4. Frontend displays chat bubbles and supports playback for user/assistant voice bubbles.
5. User can save/export conversations and re-open saved sessions later.

Current auth is intentionally lightweight:
- username-only login,
- cookie session,
- private/home network usage focus.

---

## 4. Milestone Status (M1-M10)

Milestones are implemented and documented in:
- `docs/milestones/M1-checklist.md`
- `docs/milestones/M2-checklist.md`
- `docs/milestones/M3-checklist.md`
- `docs/milestones/M4-checklist.md`
- `docs/milestones/M4.5-checklist.md`
- `docs/milestones/M5-checklist.md`
- `docs/milestones/M6-checklist.md`
- `docs/milestones/M7-checklist.md`
- `docs/milestones/M8-checklist.md`
- `docs/milestones/M9-checklist.md`
- `docs/milestones/M10-checklist.md`

Functional progression delivered:
- M1: base UI/settings/mock chat
- M2: real microphone capture
- M3: WS audio transport + backend debug
- M4: final STT transcript
- M4.5: user bubble playback
- M5: live partial transcript overlay
- M6: Ollama + Qwen tutor replies
- M7: assistant TTS playback
- M8: post-record decision flow (re-record/edit/send)
- M9: save/export/history semantics
- M10: username auth + HTTPS + LAN hardening

---

## 5. Repository Map (Important Files)

## Root
- `README.md` - primary user/contributor doc
- `.env.example` - runtime config reference
- `environment.yml` - Python env dependencies
- `scripts/dev.sh` - HTTP dev launcher
- `scripts/dev_https.sh` - HTTPS dev launcher (LAN/Tailscale aware)

## Backend
- `apps/api/main.py` - all backend logic (FastAPI, auth, history, STT/LLM/TTS, WS)
- `apps/api/prompt_profile.json` - tutor behavior profile
- `apps/api/data/` - runtime state (ignored except `.gitkeep`)
- `apps/api/generated_audio/` - generated TTS wav files

## Frontend
- `apps/web/src/App.tsx` - app state, auth, recording flow, chat UX
- `apps/web/src/components/SettingsPanel.tsx` - settings controls
- `apps/web/src/types.ts` - shared frontend types
- `apps/web/src/styles/app.css` - UI styles/layout
- `apps/web/vite.config.ts` - HTTPS cert integration for dev server

## Showcase assets
- `exmaples/fig0.png`
- `exmaples/fig1.png`
- `exmaples/mov1.mov`

Note: folder is intentionally named `exmaples` (typo in folder name). README currently references that exact path.

---

## 6. Runtime Architecture

## 6.1 Frontend (React + Vite)
- Initializes settings from localStorage.
- Resolves API/WS endpoints based on env + current hostname.
- Auth bootstrap calls `/auth/me` with cookie credentials.
- After auth, opens WS `/ws/audio` and listens for transport/STT events.
- Recording flow:
  - start -> send `recording_started` + `audio_chunk` events
  - stop -> send `recording_stopped`
  - backend returns `transcription_result`
- Sending message calls `/chat`, then `/tts`, then renders assistant bubble with playable audio.

## 6.2 Backend (FastAPI)
- CORS + trusted host middleware.
- Cookie auth for HTTP endpoints and WS handshake.
- Stateful JSON store for users/sessions/history.
- Topic SQLite store for random topic suggestions.
- STT engine: `faster-whisper`.
- LLM engine: Ollama `/api/chat` using `qwen3:8b` default.
- TTS engine: auto provider selection (`piper` -> `say` -> `espeak`).

---

## 7. Key Frontend Implementation Details

Source: `apps/web/src/App.tsx`

## 7.1 Defaults and settings
- Default settings object includes:
  - `tutorStyle: balanced`
  - `replyLanguage: jp_en`
  - `correctionIntensity: medium`
  - `responseLength: short`
  - `showLiveTranscript: true`
  - `autoPlayAssistantVoice: true`
  - `showStatusPanel: false`

## 7.2 API/WS URL resolution
- If `VITE_API_BASE_URL` is set, it is used.
- Otherwise frontend builds URL from current page host.
- Port fallback behavior:
  - `https` page -> default API port `8443`
  - `http` page -> default API port `8000`
- Optional override: `VITE_API_PORT`

This matters for previous login failures (`Failed to fetch`) when frontend was HTTPS but API defaulted incorrectly.

## 7.3 Settings panel toggles
Source: `apps/web/src/components/SettingsPanel.tsx`
- Show live transcript while recording
- Auto-play assistant voice
- Show status/debug panel

By default, `showStatusPanel = false`, so idle/debug blocks are hidden unless enabled.

---

## 8. Key Backend Implementation Details

Source: `apps/api/main.py`

## 8.1 Auth/session
- Cookie name: env `AUTH_COOKIE_NAME` (default `jst_session`)
- Session lifetime: env `AUTH_SESSION_DAYS` (default 30)
- Session cookie is HttpOnly, SameSite=Lax, secure under HTTPS
- Username validation regex: `[A-Za-z0-9._\- ]{2,40}`

## 8.2 Persistent stores
- JSON state store (`apps/api/data/state.json`):
  - `users`
  - `sessions`
  - `history`
- Topic store (`apps/api/data/topics.db`): SQLite `topics` table with `name`, `aliases`

## 8.3 STT
- Class: `LocalWhisperTranscriber`
- Model config from env:
  - `STT_MODEL_NAME` (default `small`)
  - `STT_DEVICE` (default `auto`)
  - `STT_COMPUTE_TYPE` (default `int8`)
- First run can download model artifacts.

## 8.4 LLM
- Class: `LocalOllamaTutor`
- Env:
  - `LLM_OLLAMA_BASE_URL` default `http://127.0.0.1:11434`
  - `LLM_MODEL` default `qwen3:8b`
  - `LLM_TIMEOUT_SEC`, `LLM_TEMPERATURE`, `LLM_MAX_HISTORY_TURNS`
  - `LLM_ENABLE_THINKING` default false
  - `LLM_PROMPT_PROFILE_PATH` profile JSON path
- Behavior:
  - loads prompt profile with hot-reload by mtime,
  - strips `<think>...</think>` from replies,
  - retries once if Ollama returns empty content,
  - adapts follow-up frequency (not always asking questions),
  - adds conversational connectors probabilistically,
  - supports topic steering when user selects topic.

## 8.5 TTS
- Class: `LocalSpeechSynthesizer`
- Provider chain in auto mode: `piper` -> `say` (macOS) -> `espeak` (Linux)
- Mixed-language handling options in env:
  - `TTS_SAY_MIXED_LANGUAGE=true`
  - `TTS_FORCE_JAPANESE_WHEN_JP_TEXT=true`

---

## 9. API Contract Summary

## 9.1 HTTP endpoints
- `GET /health`
- `POST /auth/login`
- `POST /auth/logout`
- `GET /auth/me`
- `GET /topics/welcome`
- `GET /history/list`
- `GET /history/{item_id}`
- `POST /history/save`
- `DELETE /history/{item_id}`
- `POST /export/audio-package`
- `POST /chat`
- `POST /tts`
- static: `GET /generated_audio/*`

## 9.2 WebSocket endpoint
- `WS /ws/audio`

Client -> server events:
- `recording_started`
- `audio_chunk` (base64 payload)
- `recording_stopped`
- `transcription_partial_request`
- `ping`

Server -> client events:
- `server_ready`
- `recording_started_ack`
- `chunk_received`
- `recording_summary`
- `transcription_started`
- `transcription_result`
- `partial_transcription_result`
- `partial_transcription_error`
- `transcription_error`
- `pong`
- `error`

---

## 10. Runbooks

## 10.1 Full local HTTP run
```bash
cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher
./scripts/dev.sh
```

Expected:
- Web: `http://<host>:5173`
- API: `http://<host>:8000`

## 10.2 Full HTTPS run (phone + secure mic + Tailscale)
```bash
cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher
./scripts/dev_https.sh
```

Expected:
- API on `8443`
- Web on `5173`
- Self-signed cert generated/updated in `deploy/certs/`
- SAN includes localhost + LAN IP + Tailscale IP (if detected)

---

## 11. Smoke Test (Minimal Resume Verification)

Run after any restart/new session:

1. Open app URL.
2. Login with username.
3. Record 3 seconds and stop.
4. Confirm STT status transitions to `done` and input gets text.
5. Send message and confirm assistant text appears.
6. Confirm assistant TTS playable from bubble.
7. Save to history, open History modal, re-open item.
8. Reload page and confirm auth/session resumes.

---

## 12. Troubleshooting Matrix

## 12.1 `Login failed: Failed to fetch`
Likely causes:
- API not running,
- HTTPS cert not trusted,
- frontend talking to wrong API port.

Checks:
```bash
# API health (HTTPS path)
curl -vk https://localhost:8443/health

# Auth endpoint reachability with origin
curl -vk -H "Origin: https://localhost:5173" https://localhost:8443/auth/me
```

Important implementation note:
- frontend now defaults API port by protocol (`8443` for HTTPS, `8000` for HTTP).

## 12.2 iPhone mic blocked/insecure
- Must use HTTPS URL.
- Must trust certificate on device.

## 12.3 STT errors
- Ensure `faster-whisper` installed in conda env.
- Ensure `ffmpeg` exists on path.
- First STT run may be slower due model load/download.

## 12.4 LLM unavailable
```bash
ollama serve
ollama pull qwen3:8b
ollama ps
```

## 12.5 History/save issues
- Check backend logs for `/history/*` endpoints.
- Ensure `apps/api/data/` is writable.

---

## 13. User-Facing Behavior Decisions Captured

These behavior preferences were intentionally implemented:
- conversation-first tutoring (not over-correcting every turn),
- follow-up questions often but not always,
- casual Japanese connectors (`なるほど`, `いいね`, etc.) in non-strict modes,
- default no status/debug clutter in UI (toggle in settings),
- voice bubble replay for both user and assistant,
- save/history is explicit and opt-in; unsaved sessions are not auto-persisted.

---

## 14. Known Non-Goals / Deferred Items

Not implemented yet (by design/defer):
- robust internet-grade auth (password/OAuth/ACL),
- production TLS/cert automation (e.g., Let’s Encrypt + reverse proxy setup),
- VAD auto-stop when silence detected,
- high-fidelity multilingual neural TTS upgrade path.

---

## 15. Recovery Procedure for a New Agent

When starting from a fresh session:

1. Read this file first.
2. Read `README.md`.
3. Run `git status --short` and confirm clean/dirty state.
4. Run local smoke test (Section 11).
5. If failing, use troubleshooting matrix (Section 12).
6. Before edits, re-validate milestone scope in `docs/milestones/*.md`.
7. Implement next task and rerun:
   - `cd apps/web && npm run check && npm run build`
   - `python -m py_compile apps/api/main.py`

---

## 16. Useful Commands (Copy/Paste)

```bash
# repo status
cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD

# web checks
cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/web
npm run check
npm run build

# backend syntax check
python -m py_compile /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/api/main.py

# launchers
cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher
./scripts/dev.sh
./scripts/dev_https.sh
```

---

## 17. Checkpoint Assets

This checkpoint doc lives at:
- `docs/checkpoints/2026-02-12-session-recovery/SESSION_RECOVERY_PLAYBOOK.md`

Visual demo assets:
- `exmaples/fig0.png`
- `exmaples/fig1.png`
- `exmaples/mov1.mov`

