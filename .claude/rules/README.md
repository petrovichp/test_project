# Project rules

Rules in this folder are **auto-loaded** into every Claude Code session via `@import` directives in the project's [`CLAUDE.md`](../../CLAUDE.md).

## Files

| File | What it covers |
|---|---|
| [data-integrity.md](data-integrity.md) | No leakage, no random splits, embargo, normalization, test-split lock, cache rule, raw data |
| [code-style.md](code-style.md) | Comments, naming, imports, print formatting, file organization, primitives to reuse |
| [experiments.md](experiments.md) | Reproducibility, documentation requirements, comparison discipline, walk-forward methodology |
| [model-registry.md](model-registry.md) | Every trained model must update `model_registry.json` + a doc before session ends |
| [git.md](git.md) | Commit format, never-commit list, branching, PR bodies |
| [agent-workflow.md](agent-workflow.md) | How Claude operates on this project — long-running work, discussion vs execution, reporting |

## Adding a new rule

1. Add the markdown file here (lowercase, dash-separated, `.md` extension).
2. Add `@.claude/rules/{filename}.md` to the "Project rules (auto-loaded)" section near the top of `CLAUDE.md`.
3. Add a row to the table above.

## Editing rules

Rules are intentionally short. If a rule needs > 200 words to express, it's probably actually two rules — split it.

When a rule changes, the change is a normal commit. Reference the rule file in the commit body so future-you understands why behavior shifted.
