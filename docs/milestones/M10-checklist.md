# Milestone 10 Checklist (Username Login + HTTPS + LAN Hardening)

## Goal

Verify username login flow, cookie session behavior, and HTTPS/LAN readiness.

## Setup

1. Start HTTPS dev stack:
   - `./scripts/dev_https.sh`
2. Open web URL shown in terminal.
3. If testing on phone, ensure same LAN and certificate is trusted.

## Pass/Fail Tests

1. Username login (no password)
- Enter a new username and submit.
- App should enter chat view.
- No password prompt should appear.

2. Returning user login
- Logout.
- Login again with same username.
- History for that username should still be available.

3. Unauthorized guard
- Clear cookies (or use private window).
- Try calling protected actions (chat/save/tts) without login.
- UI should prompt for login instead of proceeding.

4. Cookie-backed session
- After login, refresh page.
- Session should persist and skip login screen.
- After logout, refresh should return to login screen.

5. HTTPS microphone path
- Open app over `https://`.
- Start/stop recording should work.
- `WS` status should become connected and STT should complete.

6. Host/CORS hardening sanity
- Normal app origin should work.
- Invalid cross-origin or host usage should be rejected by backend config.

## Exit Criteria

Milestone 10 is done only if all six tests pass.
