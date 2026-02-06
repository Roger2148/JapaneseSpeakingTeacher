# API App

FastAPI backend for audio transport and model orchestration.

## Milestone 7 Scope

- `GET /health`: health check endpoint.
- `WS /ws/audio`: accepts audio chunks, emits chunk/session summaries, returns live partial STT, and final STT result.
- `POST /chat`: runs local Ollama (`qwen3:8b` by default) to produce tutor replies.
- `POST /tts`: generates assistant speech audio and returns a playback URL.
- `GET /generated_audio/*`: serves generated TTS files for frontend playback.

## STT Notes

- Uses `faster-whisper` locally.
- First run may download the selected Whisper model (`STT_MODEL_NAME`) from Hugging Face.
- If STT fails, check:
  - `faster-whisper` installed in conda env
  - `ffmpeg` available on system path

## LLM Notes

- Uses local Ollama by default (`LLM_MODEL=qwen3:8b`).
- Start Ollama and pull model once:
  - `ollama serve`
  - `ollama pull qwen3:8b`
- Default behavior uses non-thinking mode (`LLM_ENABLE_THINKING=false`) for faster, stable tutor replies.
- Prompt behavior is in `/Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/api/prompt_profile.json`.
- You can edit `prompt_profile.json` to tune tone/correction/follow-up rules without editing Python code.
- `LLM_PROMPT_PROFILE_PATH` can point to another JSON profile file.
- If chat replies fail, verify Ollama endpoint (`LLM_OLLAMA_BASE_URL`) and model availability.

## TTS Notes

- `TTS_PROVIDER=auto` attempts providers in this order: `piper` -> `say` (macOS) -> `espeak` (Linux).
- Set `TTS_PIPER_MODEL` to enable Piper with a local model path.
- On macOS `say`, mixed Japanese+English text can be split into language chunks and voiced separately when `TTS_SAY_MIXED_LANGUAGE=true`.
- `TTS_FORCE_JAPANESE_WHEN_JP_TEXT=true` forces single Japanese voice whenever text contains Japanese (recommended for natural JP flow).
- Generated files are written to `apps/api/generated_audio` and served via `/generated_audio`.

## Run

From repo root:

1. Activate your conda env: `conda activate japanese_teacher`
2. Start API server:
   `uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload`
