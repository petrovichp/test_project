# A1 — Capacity Test (hidden ∈ {64, 128, 256})

> **TL;DR**: Capacity helps. **VOTE5_h256 achieves WF mean Sharpe +11.86** (vs BASELINE_VOTE5 +10.40, Δ +1.46) — the highest WF Sharpe in the project. But fold-6 collapses to +0.41 (vs +5.20). **VOTE5_h128 wins fold 6 spectacularly (+10.70) and DQN-test (+10.59)** but is 5/6 on folds. Three distinct profiles, three new baselines.

## Setup

Trained 10 new policies (5 seeds × 2 hidden sizes), reusing the same seeds {42, 7, 123, 0, 99} as `BASELINE_VOTE5`. Identical hyperparameters except hidden layer width. State remains v5 (50 dims).

| Hidden size | Net params | Δ vs h=64 |
|---|---|---|
| 64 | 5,674 | (baseline) |
| 128 | 15,434 | +172% |
| 256 | 47,242 | +732% |

Reproduction:
```bash
for h in 128 256; do
  for s in 42 7 123 0 99; do
    python3 -m models.dqn_selector btc \
        --tag BASELINE_FULL_h${h}_seed${s} --seed $s \
        --fee 0 --trade-penalty 0.001 --hidden $h
  done
done
python3 -m models.eval_capacity
```

## Per-seed results

| seed | h64 val | h64 test | h64 WF | h128 val | h128 test | h128 WF | h256 val | h256 test | h256 WF |
|---|---|---|---|---|---|---|---|---|---|
| 42  | +7.30 | +3.67 | +9.03 | +4.34 | +1.72 | +7.93 | +5.43 | +2.06 | +8.92 |
| 7   | +3.96 | −1.14 | +5.60 | +5.88 | +3.15 | +8.36 | +3.61 | +7.79 | +9.13 |
| 123 | +6.18 | +3.03 | +9.63 | +3.02 | +5.53 | +6.58 | +5.60 | +3.75 | +11.05 |
| 0   | +4.02 | +5.59 | +8.26 | +7.22 | +5.70 | +9.28 | +5.72 | +5.53 | +10.45 |
| 99  | +4.92 | +9.05 | +6.19 | +2.33 | +2.90 | +7.17 | +7.65 | +6.51 | +9.67 |
| **mean** | +5.28 | +4.04 | **+7.74** | +4.56 | +3.80 | +7.86 | +5.60 | +5.13 | **+9.85** |

**Single-seed mean WF: h=64 +7.74 → h=128 +7.86 → h=256 +9.85.** h=256 lifts mean WF by +2.11 across single seeds (much stronger lift than the ensemble shows).

## Ensemble comparison (K=5 plurality)

| | BASELINE_VOTE5 (h=64) | **VOTE5_h128** | **VOTE5_h256** |
|---|---|---|---|
| val Sharpe | **+3.53** | +0.31 | +3.32 |
| val equity | 1.146 | 1.004 | 1.123 |
| test Sharpe | +4.19 | **+10.59** | +1.21 |
| test equity | 1.139 | **1.359** | 1.034 |
| **WF mean Sharpe** | +10.40 | +10.22 | **+11.86** |
| WF folds positive | 6/6 | 5/6 | 6/6 |
| Fold 6 Sharpe | +5.20 | **+10.70** | +0.41 |
| Trades (val) | 233 | 235 | 248 |
| Trades (test) | 174 | 271 | 162 |

## Per-fold WF Sharpe

| Fold | h=64 | h=128 | h=256 | Δh128 | Δh256 |
|---|---|---|---|---|---|
| 1 | +12.48 | +14.02 | **+15.92** | +1.54 | +3.44 |
| 2 | +16.24 | +18.89 | **+19.85** | +2.65 | +3.61 |
| 3 | +9.95 | +3.98 | **+13.97** | −5.97 | +4.02 |
| 4 | +14.19 | +14.26 | **+18.51** | +0.07 | +4.32 |
| 5 | +4.35 | **−0.55** | +2.51 | −4.90 | −1.84 |
| 6 | +5.20 | **+10.70** | +0.41 | +5.51 | −4.79 |

**Pattern**: Folds 1, 2, 4 lift consistently with capacity. Fold 3 is volatile across capacities. Fold 5 + Fold 6 show capacity-vs-recency tradeoff.

## Three distinct profiles → three new baselines

| Baseline | Best at | Weakness | When to deploy |
|---|---|---|---|
| **BASELINE_VOTE5** (h=64) | 6/6 folds + balanced | lower aggregate WF | Default — most regime-robust |
| **BASELINE_VOTE5_H128** | fold 6 (+10.70), test (+10.59) | fold 5 negative (5/6) | When recent regime persists |
| **BASELINE_VOTE5_H256** | WF mean (+11.86), folds 1–4 | fold 6 weak (+0.41) | When aggregate-period stationarity holds |

## Why the trade-off

**Larger nets fit early-fold regimes extraordinarily well** (h=256 fold 1 +15.92, fold 2 +19.85, fold 4 +18.51). DQN-train spans Sep 2025 → Feb 2026 — an early/mid-fold-similar regime. The bigger net captures the training-distribution patterns more precisely.

**But generalization to the latest fold (Apr 2026) gets worse with capacity** (h=64 fold 6 +5.20 → h=256 +0.41). This is classic capacity-vs-OOD: more parameters → tighter fit to in-distribution → less robust to distribution shift.

The single-seed mean WF lift (h=64 +7.74 → h=256 +9.85, Δ +2.11) is dramatic. The ensemble lift is smaller (+1.46) — voting already smooths variance, leaving less room for capacity to add aggregate Sharpe. But the per-fold pattern is clear.

## Implications

1. **Signal is NOT capacity-saturated at h=64**. Capacity adds real value in WF aggregate.

2. **Capacity costs fold-6 robustness**. The most-recent-fold weakness is a real concern for forward-deployment.

3. **The right operating point depends on regime assumptions**:
   - "Future will be like recent past (fold 6 regime)" → use **VOTE5_h128**
   - "Future will be like average period 1–4" → use **VOTE5_h256**
   - "Don't bet on regime stability" → use **BASELINE_VOTE5** (h=64)

4. **A regime-conditional ensemble** could combine all three: switch baseline based on detected regime. But this is complex and unproven.

5. **Larger capacities (h=512+) likely amplify the trade-off**, not break it. Don't expect h=512 to fix fold 6 — it'll likely make folds 1–4 even better and fold 6 worse.

## Decision

**All three baselines retained.** Update [baselines.md](baselines.md) and add VOTE5_H128 + VOTE5_H256 as formal alternatives. BASELINE_VOTE5 (h=64) remains the recommended default for its balance.

For Path A continuation:
- A2 (trade quality by vote agreement) on which baseline? → **all three**, see if vote-strength predicts quality.
- A3 (Double/Dueling DQN) at what hidden size? → **h=64 first** (cheapest), see if algo helps before stacking with capacity.

## Files

| File | Contents |
|---|---|
| [models/eval_capacity.py](../models/eval_capacity.py) | per-seed + ensemble eval across hidden ∈ {64, 128, 256} |
| `cache/btc_dqn_policy_BASELINE_FULL_h{128,256}_seed{42,7,123,0,99}.pt` | 10 trained policies |
| `cache/eval_capacity_results.json` | aggregated metrics |
