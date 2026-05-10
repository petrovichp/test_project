# Code style rules

## Comments

- No docstring walls. Brief one-line module/function summary at most.
- Comments only for non-obvious decisions — *why*, not *what*.
- No inline summaries of what was just done. The code shows what was done.
- No "step 1 / step 2" narration in code.

## Naming

- Match the surrounding code's naming and idiom. If a module uses `sp` for state-pack arrays, new code in that module also uses `sp`.
- Tags follow existing convention: `BASELINE_FULL`, `VOTE5_DD_seed{N}`, `FEE4_p001_seed{N}`. Don't invent new naming schemes mid-experiment.

## Imports

- Group: stdlib, third-party, local. Each group separated by a blank line.
- Don't reorder existing imports unless adding a new one in the right group.

## Print formatting

- Tables: fixed-width columns, right-align numbers, sign-aware `+.3f` for Sharpe. Match the pattern in `models/audit_vote5_dd.py` and similar audit scripts.
- Section headers use `'='*120` or `'='*60` separators. Match what's used nearby.
- Always include "→ {output_file_name}    [{elapsed:.1f}s]" at the end of long-running scripts.

## File length

- One focused module per concern. Don't append unrelated functions to existing modules.
- Audit / experiment scripts go in `models/` with a `audit_*`, `eval_*`, `analyze_*`, or `fee_*` prefix matching their purpose.

## Reuse

- Before writing new evaluation logic, check `models/audit_vote5_dd.py`, `models/voting_ensemble.py`, and `models/dqn_rollout.py` for `_simulate_one_trade_fee`, `_build_exit_arrays`, `run_fold`, `run_walkforward`. These are the canonical primitives.
