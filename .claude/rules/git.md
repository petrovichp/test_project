# Git rules

## Commits

- One experiment = one commit. Don't fold unrelated changes together.
- Commit message format:
  ```
  Short imperative summary (≤72 chars)

  Body: what was tested, key numbers (WF, val, test, fold count),
  and the decision/conclusion. Multiple paragraphs allowed.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  ```
- Commits must include both the code and the docs for that experiment. Code without doc = incomplete.

## Files to never commit

- Anything in `cache/` (gitignored — binaries, npz, parquet, .pt, .png, .json results).
- Anything in `.tmp/` or `/tmp/`.
- Personal scratch notebooks.

## Branch policy

- All work happens on `main` for now (single-author research).
- Tag stable points if needed (e.g. before a refactor). Don't force-push history.

## Pull request bodies

- End with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- For research PRs: include the WF/val/test deltas vs baseline in the description.
