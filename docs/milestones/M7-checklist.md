# Milestone 7 Checklist (Assistant TTS Playback)

## Goal

Verify assistant replies are converted to speech and can be replayed from assistant chat bubbles.

## Setup

1. Start backend and web as in M6.
2. Ensure Ollama is running and model is available:
   - `ollama serve`
   - `ollama pull qwen3:8b` (first time only)
3. Ensure one TTS provider is available:
   - macOS: built-in `say` is enough when `TTS_PROVIDER=auto`
   - Linux fallback: install `espeak`
   - Optional: install Piper and set `TTS_PIPER_MODEL`

## Pass/Fail Tests

1. Assistant TTS generation
- Send a normal text message.
- Assistant reply appears.
- Debug should show `TTS: done`.

2. Assistant bubble replay button
- Assistant bubble should show `[▶]`.
- Click it, audio should play.
- Click again while playing, audio should stop.

3. Auto-play toggle ON
- In Settings, keep `Auto-play assistant voice` ON.
- Send message and confirm assistant voice starts automatically.

4. Auto-play toggle OFF
- Turn OFF `Auto-play assistant voice`.
- Send message again.
- Assistant audio should not auto-start.
- Manual replay button still works.

5. Existing user voice replay unaffected
- Record a user voice message and send.
- User bubble `[▶]` still works as before.

## Exit Criteria

Milestone 7 is done only if all five tests pass.
