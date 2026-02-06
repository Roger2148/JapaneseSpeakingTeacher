# Milestone 5 Checklist (Live Partial STT Overlay)

## Goal

Verify live partial speech-to-text appears during recording and is controlled by the settings toggle.

## Setup

1. Start API and web as in M4.
2. Open app and confirm WS is connected.
3. In settings, ensure `Show live transcript while recording` is ON for tests 1-4.

## Pass/Fail Tests

1. Overlay appears while recording
- Start recording.
- A full-screen-like overlay should appear with `Live Transcript`.
- Overlay should disappear when recording stops.

2. Partial transcript updates while speaking
- Speak continuously for 5+ seconds.
- Overlay text should update progressively (not only at the end).
- Debug panel `Event` should show `partial_transcription_result` at least once.

3. Final transcript still works
- Stop recording.
- STT should finish with `transcription_result`.
- Input box should be filled with final transcript text.

4. Toggle OFF behavior
- Turn OFF `Show live transcript while recording`.
- Start recording again.
- Overlay should not appear.
- Final transcript after stop should still work.

5. Voice bubble flow preserved
- Send the transcript.
- User bubble should still support `[▶]` playback and render correctly.

## Exit Criteria

Milestone 5 is done only if all five tests pass.
