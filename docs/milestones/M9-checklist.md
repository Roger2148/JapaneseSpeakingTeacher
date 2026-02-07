# Milestone 9 Checklist (Save / Export / History)

## Goal

Verify unsaved-by-default behavior and all three save options.

## Setup

1. Start app (`./scripts/dev.sh` or manual).
2. Log in with a username.
3. Create at least one short conversation (user + assistant turns).

## Pass/Fail Tests

1. Save as text
- Open `Save`.
- Click `Save as text file (.txt)`.
- A `.txt` file should download with user/assistant lines.

2. Save as audio package
- Ensure at least one voice message exists.
- Open `Save`.
- Click `Save as audio package`.
- A `.zip` should download with:
  - `transcript.txt`
  - `metadata.json`
  - optional files under `audio/`

3. Save on server history
- Open `Save`.
- Click `Save on server history`.
- Open `History`.
- New saved item should appear.

4. Load + continue + delete
- In `History`, click one saved item.
- Conversation should load into chat and be ready to continue.
- Delete one saved item and confirm it disappears from list.

5. Reload visibility for saved sessions
- Reload page and log in again with same username.
- Saved conversation should remain accessible in `History`.

6. Unsaved-by-default behavior
- Start new conversation and do not click save.
- Reload page.
- Unsaved conversation should not be auto-persisted.

## Exit Criteria

Milestone 9 is done only if all six tests pass.
