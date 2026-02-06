# Milestone 3 Checklist (Backend Audio Transport)

## Goal

Verify microphone audio chunks are sent from browser to backend WebSocket and backend counters update in real time.

## Setup

1. Activate backend env: `conda activate japanese_teacher`
2. Go to repo root:
   `cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher`
3. Start API:
   `uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload --reload-dir apps/api`
4. In another terminal, start web:
   - `cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/web`
   - `npm install`
   - `npm run dev`
5. Open web URL.

## Pass/Fail Tests

1. WebSocket connection
- Open app and check debug panel.
- If panel is collapsed, click `Show Debug`.
- `WS status` should become `connected`.
- `WS session` should show a non-`-` session id.

2. Chunk transport during recording
- Start recording and speak for 3-5 seconds.
- `Chunks` should increase locally.
- `Server chunks` should also increase.
- `Server bytes` should become non-zero.

3. Stop summary event
- Stop recording.
- `Last backend event` should change to `recording_summary`.

4. Local preview remains intact
- `Last capture preview` player should appear and play your audio.

5. Reconnect action
- Click `Reconnect WS`.
- Status should cycle through connecting and return to connected.

6. Existing chat UX not broken
- Send one text message.
- User and assistant bubbles should still appear.
- Thread should auto-scroll to newest.

## Exit Criteria

Milestone 3 is done only if all six tests pass.
