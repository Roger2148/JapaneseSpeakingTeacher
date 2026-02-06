# Milestone 8 Checklist (Post-Record Decision Flow)

## Goal

Verify that after recording stops, user can explicitly choose `Re-record`, `Edit text`, or `Send now`.

## Setup

1. Start backend and web as in M7.
2. Open app and confirm WS is connected.
3. Keep microphone permission granted.

## Pass/Fail Tests

1. Decision panel appears after stop
- Start recording and speak for 2-5 seconds.
- Stop recording.
- A post-record panel should appear with:
  - `Re-record`
  - `Edit text`
  - `Send now`

2. Edit path
- After stop, click `Edit text`.
- Input should focus for quick editing.
- Modify transcript manually.
- Click `Send` and verify user bubble contains edited text.

3. Re-record path
- After stop, click `Re-record`.
- A new recording should start.
- Previous pending capture should be replaced by the new one.

4. Send-now path
- After stop, do not edit.
- Click `Send now`.
- Pending voice capture should be sent as a user voice bubble.
- Assistant should respond normally.

5. Empty transcript guard
- If transcript is empty, `Send now` should remain blocked and show guidance to edit/re-record.

## Exit Criteria

Milestone 8 is done only if all five tests pass.
