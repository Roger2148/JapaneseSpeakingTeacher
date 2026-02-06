# Milestone 4 Checklist (Final STT Transcript)

## Goal

Verify that after recording stops, backend STT returns a final transcript and the text appears in the input box.

## Setup

1. Activate backend env: `conda activate japanese_teacher`
2. Ensure STT dependencies are installed:
   - `faster-whisper` in `japanese_teacher`
   - `ffmpeg` available on your system
3. Go to repo root:
   `cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher`
4. Start API:
   `uvicorn main:app --app-dir apps/api --host 0.0.0.0 --port 8000 --reload --reload-dir apps/api`
5. In another terminal, start web:
   - `cd /Users/heng/Documents/GitHub/JapaneseSpeakingTeacher/apps/web`
   - `npm install`
   - `npm run dev`

## Pass/Fail Tests

1. Transport still works
- Start recording and speak for 2-4 seconds.
- `Server chunks` should increase above 0.
- `Server bytes` should be non-zero.

2. STT state flow
- During post-stop processing, `STT` should become `transcribing`.
- Then it should become `done`.

3. Final transcript in input
- After stop, input box should be filled with recognized text.
- The placeholder `[M3 audio captured ...]` should not be the final behavior anymore.

4. Transcript quality sanity
- Speak one short English sentence and one short Japanese sentence in separate tests.
- Returned text should be broadly correct for both.

5. Send continues to work
- Click `Send` after transcript appears.
- Message bubbles should still work with mock assistant reply.

6. First recording after reload
- Reload the webpage.
- Start recording immediately for 2-4 seconds.
- This first recording should also complete with `STT: done` (no initial fail).

## Exit Criteria

Milestone 4 is done only if all six tests pass.
