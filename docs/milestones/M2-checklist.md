# Milestone 2 Checklist (Real Mic Capture)

## Goal

Verify real microphone capture works in browser while keeping the M1 chat flow stable.

## Setup

1. Open terminal in `/Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/web`.
2. Install deps if needed: `npm install`.
3. Start dev server: `npm run dev`.
4. Open shown URL in desktop browser.

## Pass/Fail Tests

1. Mic permission prompt
- Click `Start Recording`.
- Browser should request microphone permission.
- If allowed, `Permission` in debug panel should become `granted`.

2. Start/stop state
- Start recording.
- Button text should become `Stop Recording (Xs)`.
- Click again to stop.
- Status should return to `Idle`.

3. Recording timer
- While recording, duration should increase continuously in debug panel.
- After stop, duration should remain at final value.

4. Audio level meter
- Speak loudly and softly during recording.
- Input level bar should move in real time.

5. Audio preview generation
- Stop recording.
- `Last capture preview` audio player should appear.
- Playback should contain your recorded voice.

6. Chunk and size telemetry
- After stop, `Chunks` should be greater than 0.
- `Last audio` should show a non-zero KB value.

7. Existing chat flow intact
- Send a text message.
- User/assistant bubbles should still work, and thread should auto-scroll to newest.

## Exit Criteria

Milestone 2 is done only if all seven tests pass.
