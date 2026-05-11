# Z3 Step 1 — standalone validation of "killed" strategies S5, S9, S11, S13

Per [development_plan.md](../reference/development_plan.md) §Z3 Step 1.

## Background

[strategy/agent.py](../strategy/agent.py) defines 13 strategies but only 9 are registered in the `STRATEGIES` dict. The comment reads:
```python
# Killed (Sharpe < -20 on both splits, no recoverable signal):
#   S5_OFIScalp, S9_LargeOrd, S11_Basis, S13_OBDiv
```

Question: does this kill verdict still hold with current data, and does it preclude adding them to the DQN action space (where weak-standalone strategies can still be useful as gated actions)?

## Method

Monkey-patched the 4 killed strategies back into `STRATEGIES` and `DEFAULT_PARAMS`, granted full regime-gate access, ran `backtest/run.run('btc')` — standard rule-based-exit backtest pipeline. Same data, same execution machinery as current 9 strategies.

[models/eval_killed_strategies.py](../models/eval_killed_strategies.py).

## Results

### Killed strategies — actual current Sharpes

| strategy | val Sharpe | val Win% | val Tr | test Sharpe | test Win% | test Tr |
|---|---:|---:|---:|---:|---:|---:|
| S5_OFISpike | **−5.15** | 37.6% | 117 | **−17.37** | 25.9% | 143 |
| S9_LargeOrd | **−5.39** | 30.0% | 120 | **−11.72** | 25.9% | 185 |
| S11_Basis | **−5.66** | 29.7% | 118 | **−8.28** | 28.7% | 171 |
| S13_OBDiv | **−2.81** | 30.3% | 33 | **−2.81** | 30.8% | 39 |

The comment's claim of "Sharpe < −20 on both splits" is **NOT borne out by current data**. Actual numbers are in the −2.81 to −17.37 range across both splits.

### Context — currently-used 9 strategies

| strategy | val Sharpe | test Sharpe |
|---|---:|---:|
| S1_VolDir | **+7.85** | −0.58 |
| S2_Funding | **−7.34** | **−10.64** |
| S3_BBRevert | **−28.00** | **−22.39** |
| S4_MACDTrend | +2.09 | +3.13 |
| S6_TwoSignal | −0.69 | −2.62 |
| S7_OIDiverg | **−19.40** | **−13.04** |
| S8_TakerFlow | +2.47 | −4.27 |
| S10_Squeeze | **−7.98** | −7.08 |
| S12_VWAPVol | 0.00 (no trades) | +3.85 (1 trade) |

**Observation**: 5 of the currently-registered 9 strategies have *worse* val Sharpe than 3 of the 4 "killed" strategies. The DQN uses these weak-standalone strategies successfully by gating them with regime + state context. The kill verdict cannot be justified on "weak standalone Sharpe" alone if S3 (-28) and S7 (-19.4) remain in the action space.

## Verdict per strategy

| | Verdict | Reason |
|---|---|---|
| **S5_OFISpike** | ❌ drop | Test −17.37 is the worst of the 4. Signal (OFI spike + taker confirm) overlaps heavily with S8_TakerFlow. |
| **S9_LargeOrd** | ❌ drop | Test −11.72. Large-order count is a coarse proxy already captured by S8 + spot_imbalance / spot_bid_concentration features. |
| **S11_Basis** | ✅ try in DQN action space | Unique signal type (basis momentum) — **not covered by any current strategy**. Val −5.66 / test −8.28 is similar to current S10_Squeeze (val −7.98 / test −7.08) which IS in the action space. |
| **S13_OBDiv** | ✅ try in DQN action space | Cross-instrument OB disagreement — **unique signal**. Smallest absolute Sharpe loss (−2.81 both splits) suggests minimal downside risk in action space. Note: only 33-39 trades, so DQN might invoke it rarely. |

## Decision

Skip the originally-planned full Z3.1 retrain with all 4 killed strategies.

**Revised Step 4**: retrain `VOTE5_H256_DD` with action space expanded by **S11_Basis + S13_OBDiv only** (10 → 12 actions). Tag: `VOTE5_v8_H256_DD`. Expected: small lift if these signal types are genuinely orthogonal; null if they're noise.

Cost: 5 seeds × ~5 min training = ~25 min + signal regeneration ~5 min = ~30 min total.

## Pass-criterion in plan was unrealistic

The plan's gate "win-rate > 50% AND mean PnL > 0.15%" cannot be applied — **none of the currently-used 9 strategies clear 50% win-rate on val** (best is S8 at 49.4%, then S4 at 47.1%, then S1 at 43.8%). All other deployed strategies are below 35%. The right criterion is **comparative**: val Sharpe ≥ worst-currently-used (S3 at −28). All 4 killed strategies pass this. Refined: S11 + S13 pass the *orthogonality* test in addition.

## Why this matters for the broader Z3 plan

This validates the original Z3 thesis: the action space is not optimally chosen. Currently we have:
- 4 weak-but-used strategies (S2, S3, S7, S10) — kept because of in-context value
- 4 strong strategies (S1, S4, S8, S12) — but S1, S8 deteriorate on test
- 2 unique-signal strategies (S11, S13) sitting unused

The plan's Step 4 should still run, just with 2 strategies (S11, S13) rather than 4. Step 6 (S15_VolBreakout) remains independent.

## Files

- [models/eval_killed_strategies.py](../models/eval_killed_strategies.py) — monkey-patch + run wrapper
- `cache/btc_backtest_results.parquet` (overwritten by this run)
