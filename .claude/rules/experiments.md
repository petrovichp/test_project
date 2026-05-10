# Experiment rules

## Reproducibility

- Every trained model has an explicit `--tag` so the policy file is `cache/btc_dqn_policy_{tag}.pt`.
- Every training run sets `--seed` explicitly. Default seed is 42 for the canonical baseline; multi-seed runs use 42, 7, 123, 0, 99.
- Fixed config: `total_steps=200_000`, `lr=1e-3`, `batch=128`, `buffer=80_000`, `warmup=5_000`, `gamma=0.99`. Document any deviation in the experiment doc.

## Documentation

- Every meaningful experiment gets a `docs/{experiment_name}.md` doc with: what was tested, the command(s), the per-fold + aggregate results table, and a verdict.
- Cross-link new docs from `CLAUDE.md` and from the `docs/experiments_log.md` if it exists.
- After committing, the doc must be enough for someone (including future-me) to reproduce without rereading the conversation.
- If the experiment trained any new model artifact, also follow [model-registry.md](model-registry.md) — register every `.pt` / `.txt` file in `model_registry.json`.

## Update RESULTS.md after every experiment

`RESULTS.md` is the project's top-level findings log. After **any** experiment completes (positive, negative, or inconclusive), update it before considering the experiment finished.

What to add:

1. **A status-block bullet** (the `> ...` quoted block at the top): one line summarizing the finding, headline metrics, and a link to the experiment's `docs/{experiment_name}.md`. Include a verdict word: POSITIVE / NEGATIVE / MIXED / DIAGNOSTIC.

2. **Update the date** in the status header (`**Status (YYYY-MM-DD, post {experiment} ...):**`) so future readers know how fresh the summary is.

3. **If the experiment changes the deployable baseline**: also update the headline baselines table and the "Forward path" line at the end of the status block.

4. **If a previous status-block bullet is now outdated** (e.g. a finding was revised, a new baseline supersedes the old one): edit the old bullet rather than appending a contradictory one. Mark superseded items with a strikethrough or remove them — the status block stays current, not historical.

The bullet format follows the existing pattern in `RESULTS.md`:

```
> - **{Experiment name} — {VERDICT}** ([docs/{experiment_name}.md](docs/...)). One-sentence summary. Headline metrics in **bold** for the new finding (e.g. WF +X.XX, val +X.XX). Optional second sentence on implication.
```

This update lives in the same commit as the docs/code/registry-entry — one commit per experiment, no exceptions. A finding without a `RESULTS.md` line is a finding that future-me will not see.

## Comparison discipline

- Always compare against the relevant frozen baseline:
  - **`BASELINE_VOTE5`** at fee=0 → WF +10.40, val +3.53, test +4.19, 6/6 folds. The bar to beat for any zero-fee experiment.
  - **`vanilla VOTE5 + top-5 + vote≥3`** at fee=4.5bp → WF +3.72, test +0.97. The bar at real taker.
- Report the same metrics: WF mean Sharpe, val Sharpe, test Sharpe, folds-positive count, trade counts.
- Decision criterion stated upfront — "✓ keep if WF ≥ baseline + 0.0 with no fold worse than baseline by >0.5".

## Walk-forward methodology

- 6 folds across the full RL period (`RL_START_REL=100_000` to `RL_END_REL=383_174`).
- Sharpe annualized at √525,960 (1-minute bars).
- Always report folds-positive count alongside aggregate Sharpe — aggregate alone hides regime fragility.

## When in doubt about validity

- Re-run the baseline command with the new code path and confirm it reproduces the published Sharpe within ±0.1.
- This catches accidental changes to the eval pipeline that would silently shift all numbers.
