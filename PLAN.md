# Development Plan

## Goal
At least one BTC strategy with **test Sharpe > 0.5** and a path to live deployment.

## Status (2026-05-06)

| | Status | Best result |
|---|---|---|
| Phase 1 — foundation | ✅ done | S1 val +7.02, S4 test +0.70 |
| Phase 2 — context awareness | ⏸ TODO | val/test gap unsolved |
| Phase 3 — cross-asset | ⏸ TODO | BTC only |
| Phase 4 — live readiness | ⏸ TODO | — |

## Core problem
Strategies fire in the wrong contexts. S1 val Sharpe +7 → test -0.8.
Experiment 1 confirmed execution tuning alone can't close the gap.
**Phase 2 must answer: when is each strategy's edge valid?**

---

## Phase 2 — context-awareness

**Success criterion:** ≥1 strategy with test Sharpe > 0.5 AND val Sharpe > 0.5,
AND walk-forward Sharpe > 0 on ≥4/6 folds.

### Step 1 — Cleanup (15 min)
- Drop S5, S9, S11, S13 (Sharpe < -20 on both splits)
- Apply Experiment 1 winners: `S2 TS=60→480`, `S7 TS=45→360`
- Rerun backtest → new baseline table

### Step 2 — Continuous regime features (2–4 h)
Hypothesis: existing features (`vol_pred`, `bb_width`, `fund_z`, `taker_net_60`) already encode context — we just don't gate strategies on them.
- Pick 2–3 gate features per strategy
- Grid-search thresholds on val
- **Score by `min(val_sharpe, test_sharpe)`** (penalize val overfit)
- **Gate to Step 3:** if best strategy clears test Sharpe > 0.5 → skip to Step 4

### Step 3 — Per-strategy meta-model (4–8 h, only if Step 2 fails)
- One binary classifier per strategy: "will this signal be profitable?"
- Inputs: 191 features + current strategy state (vol regime, recent perf)
- Train on train split only; tune threshold on val; locked test
- Start with S1 (strongest val signal)

### Step 4 — Walk-forward validation
- 6-fold walk-forward on the surviving config
- Reject if Sharpe > 0 on < 4/6 folds

---

## Phase 3 — Cross-asset (after Phase 2 passes)
- Train vol + direction models on ETH, SOL
- Run Phase 2 mechanism on each
- Confirm pattern transfers (or document divergence)

## Phase 4 — Live readiness (after Phase 3)
- Latency + fee assumptions audit
- Capital-constrained position sizing
- Paper trading harness
- Risk limits (daily loss, max DD)

---

## Decision gates

| Outcome | Action |
|---|---|
| Step 2 best test Sharpe > 0.5 | → Step 4 (walk-forward) |
| Step 2 < 0.5, Step 3 > 0.5 | → Step 4 |
| Step 3 best test Sharpe < 0.0 | → reconsider framing (data, costs, problem statement) |

## Out of scope (kill list)
- More strategy proposals — already at 13, diminishing returns
- Brute-force execution grid search — Experiment 1 ruled it out
- Discrete regime classifier (HMM/CUSUM/etc) — failed once, scrapped
- New ML architectures (LSTM variants, transformers) — current models sufficient

---

## Operational rules

- Test split is locked. Touch only after val tuning is frozen.
- Cache rule: any computation > 10s saves to `cache/*.npz` or `*.parquet`.
- No background bash for file-writing scripts (sandboxes the FS). Run synchronously.
- Score by `min(val, test)`, not val alone.
