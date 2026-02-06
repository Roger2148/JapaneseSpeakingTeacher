# Japanese Speaking Teacher

LAN-accessible Japanese speaking practice app (web frontend + Python API).

## Current Milestone Status

- M1-M8: implemented.
- M9: implemented (save/export/history semantics).
- M10: implemented baseline (username login, cookie session, host/CORS hardening, HTTPS dev launch path for phone mic testing).

## Quick Start

### HTTP dev (fast path)

From repo root:

`./scripts/dev.sh`

### HTTPS dev (phone mic path)

From repo root:

`./scripts/dev_https.sh`

Notes:
- This generates a self-signed cert at `deploy/certs/`.
- On iPhone/Safari, trust the cert if needed; mic capture requires secure context.
- Override LAN IP detection by setting `LAN_HOST=...` before running.

### Manual launch

1. API:
   - `conda activate japanese_teacher`
   - `conda env update -f environment.yml --prune`
   - `uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload --reload-dir apps/api`
2. Ollama:
   - `ollama serve`
   - `ollama pull qwen3:8b`
3. Web:
   - `cd apps/web`
   - `npm install`
   - `npm run dev -- --host 0.0.0.0 --port 5173`

## M9 Save/History Behavior

- Default session is unsaved.
- Save modal options:
  1. Save as audio package (`.zip`, with transcript + clips)
  2. Save as text (`.txt`)
  3. Save on server history
- Saved history is user-scoped.
- API state file: `apps/api/data/state.json`

## Login/Auth (M10)

- Username-only login (no password).
- Returning user logs in with same username.
- Session is stored in HttpOnly cookie.

## Prompt Tuning

- Edit `apps/api/prompt_profile.json` to tune conversation style/correction/follow-up behavior.
- Optional: set `LLM_PROMPT_PROFILE_PATH` to point to a different prompt profile JSON.

## Validation Checklists

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
