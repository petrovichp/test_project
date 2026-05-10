# Experiment rules

## Reproducibility

- Every trained model has an explicit `--tag` so the policy file is `cache/btc_dqn_policy_{tag}.pt`.
- Every training run sets `--seed` explicitly. Default seed is 42 for the canonical baseline; multi-seed runs use 42, 7, 123, 0, 99.
- Fixed config: `total_steps=200_000`, `lr=1e-3`, `batch=128`, `buffer=80_000`, `warmup=5_000`, `gamma=0.99`. Document any deviation in the experiment doc.

## Documentation

- Every meaningful experiment gets a `docs/{experiment_name}.md` doc with: what was tested, the command(s), the per-fold + aggregate results table, and a verdict.
- Cross-link new docs from `CLAUDE.md` and from the `docs/experiments_log.md` if it exists.
- After committing, the doc must be enough for someone (including future-me) to reproduce without rereading the conversation.

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
