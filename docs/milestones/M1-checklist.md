# Milestone 1 Checklist (UI + Settings + Mock Chat)

## Goal

Verify that:
- Settings panel exists and persists values.
- Chat bubbles render correctly for user and assistant.
- Record button state toggles in UI.

## Setup

1. Open terminal in `/Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/web`.
2. Install deps: `npm install`.
3. Start dev server: `npm run dev`.
4. Open shown URL on desktop browser.

## Pass/Fail Tests

1. Settings panel open/close
- Click `Settings`.
- Panel should slide in.
- Click `Close`.
- Panel should slide out.

2. Settings persistence
- Change all 5 settings once.
- Reload page.
- All selected settings should remain unchanged.

3. User message bubble
- Type a message and click `Send`.
- User bubble should appear on right with dark background.

4. Assistant mock reply
- After ~0.65s, assistant bubble should appear on left.
- Text should include `Mock ... reply`.

5. Recording button visual state
- Click `Start Recording`.
- Button text should switch to `Stop Recording`.
- Status row should show recording placeholder.
- Click again and state should revert.

6. Mobile layout sanity
- Open on mobile width (DevTools or phone).
- Composer should stack vertically and remain usable.

7. Composer anchoring
- Send 20+ short messages.
- The message thread should scroll.
- The composer row (`Start/Stop`, input, `Send`) should remain visible at the bottom.

8. Auto-scroll to newest message
- Scroll near top of thread.
- Send one new message.
- Thread should jump to the latest message automatically.

9. Save panel options
- Click `Save` in top bar.
- Modal should show 3 options:
  - Save as audio package
  - Save as text file
  - Save on server history pool
- Text export should download a `.txt` file.
- Save to history pool should show a success status message.

## Exit Criteria

Milestone 1 is done only if all nine tests pass.
