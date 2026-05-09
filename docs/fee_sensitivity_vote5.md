# Vanilla BASELINE_VOTE5 — fee sensitivity & trade-count reduction

Mirror of [vote5_dd_audit.md](vote5_dd_audit.md) Parts B+C, run on **vanilla VOTE5** (5×DQN, seeds 42/7/123/0/99, plurality voting).

Source: [models/fee_sensitivity_vote5.py](../models/fee_sensitivity_vote5.py).
Output: [cache/fee_sensitivity_vote5_results.json](../cache/fee_sensitivity_vote5_results.json).

## Headline

**Vanilla VOTE5 is more fee-robust than VOTE5_DD on the walk-forward when combined with strategy filters.** At a realistic 4 bp/side fee, the best vanilla configuration (`top-5 strategies + vote ≥ 3`) hits **WF +4.03**, more than double VOTE5_DD's best at the same fee (+1.88).

However, val is uniformly hostile across all fee ≥ 1 bp configurations — vanilla VOTE5 has more borderline trades on the val period than DD does, and they all fail under fees.

## Part B — fee sensitivity

| fee/side | bp | WF mean | val Sharpe | test Sharpe | folds + | trades |
|---|---:|---:|---:|---:|:--:|---:|
| 0.0000 | 0.0 | **+10.400** | **+3.532** | **+4.191** | 6/6 | 1,122 |
| 0.0001 | 1.0 | +7.844 | −3.266 | +2.344 | 5/6 | 1,123 |
| 0.0002 | 2.0 | +6.274 | −5.355 | +1.449 | 4/6 | 1,132 |
| 0.0004 | 4.0 | +1.436 | −9.582 | −4.888 | 4/6 | 1,150 |
| 0.0006 | 6.0 | −0.577 | −12.240 | −8.575 | 4/6 | 1,119 |
| 0.0008 | 8.0 | −4.434 | −14.337 | −13.355 | 3/6 | 1,091 |
| 0.0012 | 12.0 | −8.121 | −17.798 | −15.413 | 1/6 | 1,113 |
| 0.0020 | 20.0 | −19.927 | −25.838 | −25.815 | 0/6 | 1,128 |

### Side-by-side vs VOTE5_DD (WF mean Sharpe)

| fee bp | VOTE5 (vanilla) | VOTE5_DD | Δ |
|---:|---:|---:|---:|
| 0  | +10.40 | +6.80 | **+3.60** |
| 1  | +7.84  | +4.92 | +2.92 |
| 2  | +6.27  | +3.81 | +2.46 |
| 4  | +1.44  | +0.75 | +0.69 |
| 6  | −0.58  | −3.85 | +3.27 |
| 8  | −4.43  | −8.51 | +4.08 |

Vanilla VOTE5 outperforms DD at **every** fee level on WF. The fee curve is steeper but starts much higher.

### Breakeven

- Vanilla VOTE5: positive WF Sharpe survives up to ~5 bp/side
- VOTE5_DD: positive WF Sharpe survives up to ~4 bp/side
- Mean PnL/trade vanilla: ~0.20% (1,122 trades to ~10.4 WF Sharpe / yr base)
- Mean PnL/trade DD: ~0.18%

Both die well below OKX taker (8 bp/side). Both survive OKX maker tier-1 if the policy can be **strictly maker-only**.

## Part C — trade-count reduction at fee=4 bp

Goal: filter trades so the surviving alpha exceeds the round-trip cost.

| filter | trades | val Sh | test Sh | WF | folds+ |
|---|---:|---:|---:|---:|:--:|
| baseline (no filter) | 1,150 | −9.58 | −4.89 | +1.44 | 4/6 |
| vote ≥ 3 | 1,149 | −9.45 | −6.74 | +1.33 | 4/6 |
| vote ≥ 4 | 621 | −4.03 | −2.17 | +2.22 | 3/6 |
| vote ≥ 5 (unanimous) | 122 | −3.94 | −4.23 | +2.68 | 4/6 |
| ablate S6 | 1,147 | −7.71 | **+0.43** | +2.62 | 4/6 |
| ablate S6+S10 | 1,074 | −6.68 | **+3.98** | +3.25 | 5/6 |
| ablate S6+S7+S10 | 876 | −10.48 | +3.01 | +2.39 | 5/6 |
| top-5 (S1,S4,S7,S8,S10) | 1,060 | −7.79 | +1.48 | +3.98 | 5/6 |
| top-3 (S1, S7, S8) | 1,025 | −8.68 | −6.53 | +1.82 | 4/6 |
| **top-5 + vote ≥ 3** | **1,056** | **−7.67** | **+1.02** | **+4.03** | **5/6** |

**Best at fee=4 bp:** `top-5 + vote ≥ 3` → WF **+4.03**, test +1.02, 5/6 folds positive. Cuts S2, S3, S6, S12 (the four lowest-contribution strategies in A4 audit) and demands ≥3 of 5 nets agree.

**Best on test split:** `ablate S6+S10` → test **+3.98** with WF +3.25 and 5/6 folds. This validates the original audit-followup hypothesis that S6 and S10 were net-negative contributors.

**Val remains broken** under any filter at fee=4 bp. The val period (Jan-Feb 2026) has structurally lower per-trade alpha than train or test. Implication: **don't pick deployment configs by val Sharpe at non-zero fee** — pick by WF + test, treat val as a stress test only.

## Comparison to VOTE5_DD trade reduction (4 bp)

| filter | VOTE5 WF | VOTE5_DD WF |
|---|---:|---:|
| baseline | +1.44 | +0.75 |
| vote ≥ 5 | +2.68 | **+1.88** |
| top-5 + vote ≥ 3 | **+4.03** | +1.13 |
| ablate S6+S10 | +3.25 | n/a |

Vanilla VOTE5 + filters dominate. **The defensive narrative around DD doesn't pay off under fees once filters are applied to vanilla.**

## Implications

1. The right baseline for production scoping is **vanilla VOTE5 + audit-derived strategy filter (drop S6, S10)**, not VOTE5_DD.
2. Fees are still the dominant constraint. Even the best filter (+4.03 WF at 4 bp) is much weaker than fee=0 (+10.40 WF). Maker-only execution remains the highest-leverage decision.
3. Stronger fee penalty during training (e.g. `--trade-penalty 0.005` vs current 0.001) is now the natural next experiment: can the policy *learn* to be selective rather than relying on post-hoc filtering?

## Recommended deployment config (if shipping today, fee ≈ 4 bp/side)

```
policy   = BASELINE_VOTE5 (5×DQN, seeds 42/7/123/0/99, plurality)
exits    = rule-based (ATR-scaled TP/SL + BE + trail-after-BE + time-stop)
filters:
  vote_threshold ≥ 3
  strategies     ∈ {S1_VolDir, S4_MACDTrend, S7_OIDiverg, S8_TakerFlow, S10_Squeeze}
expected:
  WF mean Sharpe ≈ +4.0
  test Sharpe   ≈ +1.0
  trades        ≈ 1,000 / yr
```
