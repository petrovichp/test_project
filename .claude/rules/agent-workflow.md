# Agent workflow rules

How Claude should operate on this project.

## Before starting work

- Read `CLAUDE.md` and any `RESULTS.md` / latest `docs/*.md` referenced there. Don't restart from scratch on every session.
- Check the current git status; uncommitted work in progress should be honored, not reverted.
- Never recompute something that's already cached in `cache/*.npz` or `cache/*.parquet`.

## Long-running work

- Training takes ~2.4 min per seed (h=64, 200k steps). 5-seed ensembles are ~12 min.
- Run multi-seed training as a single shell loop in the background (`nohup bash -c '...' &`). Don't launch separate Bash tool calls per seed.
- Use `ScheduleWakeup` only if there's a specific signal to check (e.g. a process completing). Otherwise just check progress at natural break points.

## Discussion vs execution

- When the user asks a strategic question ("what should we do next?", "what are the ideas?"), respond with structured options and let them choose. Don't auto-launch experiments without confirmation.
- When the user gives a directive ("retrain with X", "run the eval"), execute immediately and report results.
- When experiment results are surprising or ambiguous, present them clearly and ask before deciding the next step.

## Reporting results

- Use the table format that matches existing docs (`docs/fee_aware_retrain.md`, `docs/voting_ensemble.md`).
- Per-fold breakdown alongside aggregate.
- Always quote the baseline number in the same line where you quote the new result.

## What not to do

- Do not invent new file naming conventions mid-session. Match `BASELINE_*`, `VOTE5_*`, `FEE4_*` patterns.
- Do not "fix" code that the user hasn't asked to be fixed, even if it looks suspect. Note it instead.
- Do not commit experimental results without first showing them to the user and getting confirmation, **unless** the experiment was explicitly requested.
