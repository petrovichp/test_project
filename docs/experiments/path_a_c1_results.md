# Path A (validation) + Path C1 (curriculum) — results

Executed 2026-05-11 per [development_plan.md](../reference/development_plan.md) recommendation.

## Path A — validation of `VOTE5_v8_H256_DD` baseline

### A1 — Plot vs prior baselines

[cache/plots/plot_v8_vs_baselines.png](../cache/plots/plot_v8_vs_baselines.png).

| split | BASELINE_VOTE5 | VOTE5_H256_DD | **VOTE5_v8** ⭐ | BTC B&H |
|---|---|---|---|---|
| val | +3.53 ×1.15 (233 tr) | +3.21 ×1.13 (320) | **+6.67 ×1.34 (300)** | ×1.07 |
| test | +4.19 ×1.14 (174) | **+9.01 ×1.32 (228)** | +4.44 ×1.17 (199) | ×1.09 |

v8 wins val by a wide margin. H256_DD retains the best test.

### A2 — Fee-curve scan

| fee bp | WF | val | test | folds+ |
|---:|---:|---:|---:|---|
| 0.0 | **+12.07** | +6.67 | +4.44 | 6/6 |
| 1.0 | +10.55 | −0.28 | −1.18 | 6/6 |
| 4.0 | +5.42 | −4.22 | −4.62 | 4/6 |
| **4.5 (OKX taker)** | **+4.58** | −5.77 | −5.08 | 4/6 |
| 6.0 | +1.70 | −6.39 | −5.62 | 4/6 |
| 8.0 | −3.39 | −10.11 | −7.92 | 1/6 |

At realistic taker fee (4.5 bp), v8 holds WF +4.58 — **better than the prior best at 4.5 bp** (vanilla VOTE5 + top-5+vote≥3 was +3.72). Val and test go negative under fees as expected. Breakeven ~5 bp/side. Maker-only execution remains the only path to retain zero-fee Sharpe.

### A3 — Disjoint-seed validation (CRITICAL FINDING)

Same architecture trained on disjoint seed pool {1, 13, 25, 50, 77}:

| pool | WF | val | test | fold-6 | folds+ |
|---|---:|---:|---:|---:|:--:|
| orig {42,7,123,0,99} | **+12.07** | **+6.67** | +4.44 | +4.44 | 6/6 |
| disjoint {1,13,25,50,77} | +8.91 | **−2.12** | +5.62 | +0.04 | 6/6 |
| **mean of pools** | **+10.49** | **+2.28** | +5.03 | +2.24 | 6/6 |

**v8's val Sharpe is partly seed-luck.** The headline +6.67 collapses to −2.12 in the disjoint pool. Realistic expected val is +2-3, not +6.67.

But the structural improvement is real:
- **6/6 fold positivity preserved across both pools**
- **Mean WF +10.49 still beats vanilla VOTE5 pools' mean (~+10.23)** — small but real lift
- Test is more reproducible than val (mean +5.03 across pools)

This is the same lesson as Z1.4 disjoint validation of H128 (which we dropped) and H256 (which we kept). v8's disjoint variance is similar to H128's in magnitude (Δ WF −3.16 vs H128's −3.35), but unlike H128, v8 preserves 6/6 fold positivity → keep as primary, but understand val number is noisy.

### Path A verdict

`VOTE5_v8_H256_DD` retained as primary baseline with **realistic expectations**:
- WF mean Sharpe: ~+10.5 expected (was advertised +12.07 — seed-luck inflated)
- val Sharpe: ~+2-3 expected (was advertised +6.67)
- test Sharpe: ~+5 expected
- 6/6 fold positivity: structural, holds across pools
- At 4.5 bp fees: ~+4.5 WF still expected — best fee-robustness of any baseline

---

## Path C1 — Curriculum learning (Z4.3) — NEGATIVE

### Setup
- Same architecture as v8 primary (H256 + Double_Dueling, v8_s11s13 state, 5 seeds)
- Plus `--curriculum` flag: 3-phase training by regime difficulty
  - Phase 1 (steps 0-50k): only regime_id == 0 (calm bars)
  - Phase 2 (50-120k): + regime_id ∈ {1, 2} (trend_up, trend_down)
  - Phase 3 (120k+): all 5 regimes (+ ranging, chop)

### Results

| | WF | val | test | folds+ | fold-6 |
|---|---:|---:|---:|---|---:|
| VOTE5_v8 (no curriculum) | **+12.07** | **+6.67** | +4.44 | 6/6 | +4.44 |
| **VOTE5_v8_CURR** | **+3.36** | +2.77 | +5.10 | **5/6** | +4.09 |
| Δ | **−8.72** | −3.90 | +0.66 | −1 fold | −0.35 |

Per-fold WF: +4.35, +4.99, +3.19, **−0.50**, +4.02, +4.09 — fold 4 went negative.

### Why curriculum failed

The targeted benefit (improved fold-6 = recent regime robustness) **did not materialize** — fold-6 changed by only −0.35.

Likely mechanism of failure:
1. **Phase 1 (calm-only) biases the buffer**: 50k transitions all from calm regime means by step 50k, the replay buffer is 25% saturated with one regime's transitions
2. **Phase 2 + 3 don't recover the diversity**: by step 200k, the buffer has limited samples from harder regimes (ranging, chop) — they were only added in phase 3 and only get ~80k steps of trajectory accumulation vs the original 200k
3. **Trajectory skipping slows rollout**: when the rollout cursor encounters a forbidden-regime bar, it advances without pushing to buffer. Net effect: phase 1 produces fewer trajectories per wall-time than baseline because most bars get skipped

This is the inverse of what we hoped. Curriculum learning works in domains where "simple" examples genuinely build robust priors. Here, the regime gating just deprives the policy of crucial signal diversity.

### Verdict

❌ Drop `VOTE5_v8_CURR`. Document as a negative Z4 result.

### What we learned

- Z4.3 curriculum doesn't work on this signal — the regime classes aren't really "difficulty levels" in a sense that supports staged learning
- For Z4 to find gains, the architectural change must let the policy extract more from the existing buffer, not change what's IN the buffer

---

## Summary

| experiment | verdict | key finding |
|---|---|---|
| A1 plot | ✅ done | v8 wins val, H256_DD wins test |
| A2 fee curve | ✅ done | Breakeven ~5 bp; v8 has best fee robustness of any baseline |
| A3 disjoint validation | 🟡 **CRITICAL DIAGNOSTIC** | v8's val (+6.67) is partly seed-luck; realistic +2-3 |
| C1 curriculum | ❌ NEGATIVE | WF dropped −8.72; fold-6 target didn't materialize |

## Path forward

Given:
- v8 baseline retained but understood with realistic expectations
- Curriculum learning failed
- C2 (self-distillation) and C3 (QR-DQN) are larger research bets (~1-2 days each)
- The action+state surface looks saturated (Step 5 didn't compose, Step 6 failed, C1 failed)

Three honest options:
1. **Accept `VOTE5_v8_H256_DD` as production-ready** (with realistic expectations ~+10.5 WF mean) and move to Path X (maker-only execution scoping)
2. **Continue Path C** — C2 (self-distillation, ~1 day) and C3 (QR-DQN with CVaR, ~2 days)
3. **Pivot to data collection** — pull cross-asset (ETH, SOL) and re-run Z2.1, which is the genuinely-new-information bet
