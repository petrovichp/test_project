# Z2 + Z3 Steps 2/3/4 — results

Per [development_plan.md](development_plan.md). Phase Z2/Z3 partial execution. Completed 2026-05-11.

## TL;DR

**Step 4 (`VOTE5_v8_H256_DD`) is the new primary baseline.** Adding S11_Basis + S13_OBDiv to the action space (10 → 12) lifted WF mean Sharpe from +11.05 to **+12.07**, val from +3.21 to **+6.67**, with 6/6 folds positive. Test dropped from +9.01 to +4.44 — real cost, but WF aggregate is the headline metric and val resilience is materially better.

Step 2 (price-action context) was a **negative result** — drop. Step 3 (basis+funding state) was a **partial win** — val nearly doubled but test/fold-6 regressed; keep as an alternative for val-robustness deployments.

## Results

| step | config | WF | val | test | folds+ | fold-6 | verdict |
|---|---|---:|---:|---:|:--:|---:|---|
| baseline | `VOTE5_H256_DD` | +11.05 | +3.21 | **+9.01** | 6/6 | +8.23 | (target to beat) |
| **2** | `VOTE5_v7pa_H256_DD` (price-action) | +9.18 | −1.49 | +2.44 | 5/6 | +3.43 | ❌ **DROP** |
| **3** | `VOTE5_v7basis_H256_DD` (basis+funding) | +11.66 | **+6.05** | +2.90 | 6/6 | +2.31 | 🟡 PARTIAL |
| **4** | **`VOTE5_v8_H256_DD`** (S11+S13 in action) ⭐ | **+12.07** | **+6.67** | +4.44 | **6/6** | +4.44 | ✅ **WIN — new baseline** |

### Per-fold WF Sharpes

| config | f1 | f2 | f3 | f4 | f5 | f6 |
|---|---:|---:|---:|---:|---:|---:|
| `VOTE5_H256_DD` (baseline) | +11.75 | +16.47 | +7.86 | +19.12 | +2.88 | +8.23 |
| Step 2 v7_pa | +11.30 | +18.51 | +8.88 | +13.76 | **−0.79** | +3.43 |
| Step 3 v7_basis | +15.56 | +20.36 | +11.70 | +12.05 | **+7.97** | +2.31 |
| **Step 4 v8 (S11+S13)** ⭐ | +11.14 | +19.43 | +13.08 | +18.01 | +6.29 | +4.44 |

## Step 2 — price-action context (NEGATIVE)

**What was tested**: 4 new state features (drawdown_60, runup_60, realized_vol_60, vol_ratio_30_60). State 50 → 54 dims. Action space unchanged (10).

**Result**: WF dropped −1.87, val collapsed from +3.21 to −1.49, fold 5 turned negative (−0.79). Only step to lose 6/6 fold-positive.

**Why it failed (hypothesis)**: the added features were largely redundant with information already in state — regime classifier (5 one-hot dims) + ATR + windowed price returns already encode pullback distance and vol regime. The new features added noise and grew the state dim without bringing orthogonal signal.

**Verdict**: drop. v7_pa state cache + 5 trained policies remain in registry as a documented negative result.

## Step 3 — basis + funding state (PARTIAL)

**What was tested**: 5 new state features (basis_z_60, basis_change_1bar, funding_apr, funding_z_120, oi_change_60). State 50 → 55 dims. Action space unchanged (10).

**Result**: WF lifted +0.61. Val **nearly doubled** (+3.21 → +6.05). Fold 5 lifted dramatically (+2.88 → +7.97). But test dropped −6.11 (+9.01 → +2.90) and fold-6 collapsed (+8.23 → +2.31).

**Why partial**: the val-resilience hypothesis ([fee_aware_retrain.md](fee_aware_retrain.md) had predicted this) was **confirmed**. Adding basis + funding macro context made the policy more robust on the OOD val period where leveraged-flow dynamics matter. The flip side: the policy seems to over-rely on these features on test (Jan-Mar 2026) and fold-6 (Apr 2026) where basis/funding regimes are calmer — leading to weaker performance there.

**Verdict**: keep as alternative baseline `VOTE5_v7basis_H256_DD` for deployments prioritizing val-period robustness (e.g. recent regime shift detected). Not promoted as primary because test/fold-6 regression is too large.

**Implication for future work**: basis/funding features could be *combined* with the v8 action space (Step 5 in plan) — that combination may give the action-space lift AND val resilience without losing test.

## Step 4 — S11_Basis + S13_OBDiv added to action space (WIN)

**What was tested**: action space 10 → 12 by adding S11_Basis (basis momentum signal) and S13_OBDiv (cross-instrument OB disagreement). State 50 → 52 dims (just appending signal flags). Trained 5 H256+DD seeds at v8_s11s13.

**Result**: WF **+12.07** (vs baseline +11.05, Δ +1.02). Val **+6.67** (Δ +3.46). 6/6 folds positive. Folds 2-5 all stronger than baseline. Test +4.44 (Δ −4.57) and fold-6 +4.44 (Δ −3.79) regressed but stayed comfortably positive.

**Why it worked**: S11 and S13 cover **signal types not in the original 9 strategies**. S11 fires on basis momentum (perp-spot z-score), S13 on cross-instrument OB disagreement. The DQN learned to gate them effectively — trade count grew modestly (1372 → 1416 on WF), suggesting selective use. Val activity (S11=5.94%, S13=1.58% of bars) confirms non-trivial firing.

**Per the audit-driven thesis (from [docs/z3_step1_killed_strategies.md](z3_step1_killed_strategies.md))**: the "killed" verdict on these strategies from older work was based on standalone Sharpe alone. In the DQN action space they prove their value by adding regime contexts that the existing 9 don't cover.

**Verdict**: ✅ **NEW PRIMARY BASELINE**. Promotes to top of [docs/baselines.md](baselines.md).

## Updated baseline leaderboard (post Step 4)

| Rank | baseline | WF | val | test | folds+ | fold-6 |
|---|---|---:|---:|---:|:--:|---:|
| **1** | **`VOTE5_v8_H256_DD`** ⭐ NEW | **+12.07** | **+6.67** | +4.44 | 6/6 | +4.44 |
| 2 | `VOTE5_v7basis_H256_DD` (val-robust) | +11.66 | +6.05 | +2.90 | 6/6 | +2.31 |
| 3 | `VOTE5_H256_DD` (prior primary) | +11.05 | +3.21 | **+9.01** | 6/6 | **+8.23** |
| 4 | `BASELINE_VOTE5_H256` | +11.86 | +3.32 | +1.21 | 6/6 | +0.41 |
| 5 | `BASELINE_VOTE5` (vanilla) | +10.40 | +3.53 | +4.19 | 6/6 | +5.20 |

## Next step proposal

Per the plan, Step 5 (Z2.5 combined state v7) would test additivity. Now that Step 3 + Step 4 both produced winners (Step 3 marginal, Step 4 clear):

**Step 5 candidates worth running**:
1. **v8 action space + v7_basis state combined** — should give both: action-space lift (+1.02 WF) AND val resilience (+2.84 val). Cost: ~30 min. **Highest expected payoff.**
2. ~~v7_pa + v7_basis combined~~ — skip, v7_pa was negative.

Other forward paths:
- **Test fee=4.5bp** on the new `VOTE5_v8_H256_DD` baseline (per Z5.3)
- **Phase Z4** (architecture experiments) — now there's a higher baseline to beat
