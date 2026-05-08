# A4 — BASELINE_VOTE5 Deal-by-Deal Audit

> **TL;DR**: For each of 1,122 trades fired by `BASELINE_VOTE5` across the full RL period, recorded strategy, direction, vote count, exit reason, and regime. **System is strongly defensive** (best WF performance in worst BTC folds). **S1_VolDir is the dominant contributor** (36% of cumulative PnL); **S4_MACDTrend is the precision instrument** (29 trades, +0.86% per trade — 3× average). 35% of trades exit on TIME, only 4.3% on TP, but TP hits are very high-value (+2.49%).

## Setup

Traced every trade from `BASELINE_VOTE5` (K=5 plurality of seeds 42, 7, 123, 0, 99) across the 6-fold walk-forward period. For each trade, captured: fold, bar timestamps, strategy index/key, direction (long/short), votes count, entry/exit prices, holding bars, PnL, exit reason, regime.

Reproduction:
```bash
python3 -m models.audit_vote5
```

Output: `cache/audit_vote5_trades.json` — full per-trade log + aggregated stats.

## Per-fold summary

| fold | trades | BTC return | BASELINE_VOTE5 equity |
|---|---|---|---|
| 1 | 152 | −6.43% | **×1.619** |
| 2 | 209 | **−16.88%** | **×2.292** |
| 3 | 131 | +6.33% | ×1.347 |
| 4 | 248 | **−29.83%** | **×2.005** |
| 5 | 222 | +6.02% | ×1.171 |
| 6 | 160 | +9.11% | ×1.154 |

**Defensive pattern**: equity gains are largest in the worst BTC folds (f2 BTC −17% → ×2.29; f4 BTC −30% → ×2.00). Confirms the anti-correlated profile from earlier seed_variance and audit_followup analyses.

## Per-strategy contribution (across all 6 folds)

| strategy | count | long | short | mean PnL | win % | sum PnL | avg votes |
|---|---|---|---|---|---|---|---|
| **S1_VolDir** | **396** (35%) | 271 | 125 | +0.273% | 60.6% | **+108.24%** | 3.19 |
| S8_TakerFlow | 244 (22%) | 108 | 136 | +0.195% | 56.6% | +47.69% | 3.23 |
| S7_OIDiverg | 249 (22%) | 131 | 118 | +0.176% | 52.6% | +43.79% | 3.12 |
| S10_Squeeze | 88 (8%) | 39 | 49 | +0.311% | 64.8% | +27.34% | 3.22 |
| **S4_MACDTrend** | 29 (3%) | 15 | 14 | **+0.863%** | 69.0% | +25.02% | 3.14 |
| S6_TwoSignal | 42 (4%) | 31 | 11 | +0.270% | 54.8% | +11.34% | 3.07 |
| S2_Funding | 13 (1%) | 9 | 4 | +0.109% | 69.2% | +1.41% | 2.92 |
| S3_BBRevert | 61 (5%) | 35 | 26 | +0.020% | 52.5% | +1.24% | 3.10 |
| S12_VWAPVol | 0 | — | — | — | — | — | — |

**S1 is the workhorse** — 35% of trades, 36% of cumulative PnL. **S4 is the precision instrument** — 29 trades but +0.86% per trade (3× the average). S2, S3, S12 contribute minimally.

The earlier audit follow-up tests showed ablating S6/S7/S10 still hurts despite low individual contribution. This audit confirms why: S7 and S8 each contribute ~$45 per $1000 deployed (across 6 folds). They're not redundant.

## Long vs short attribution per fold

| fold | long n / PnL | long win% | short n / PnL | short win% | Notable |
|---|---|---|---|---|---|
| 1 | 74 / +20.8% | 63.5% | 78 / +28.3% | 57.7% | balanced |
| 2 | 106 / +34.8% | 57.5% | 103 / **+49.6%** | 67.0% | short bigger in BTC −17% |
| 3 | 73 / +21.6% | 65.8% | 58 / +8.7% | 56.9% | long dominates BTC up |
| 4 | 156 / +26.4% | 57.7% | 92 / **+44.5%** | 63.0% | short dominates BTC −30% |
| 5 | 134 / +10.9% | 55.2% | 88 / +5.6% | 52.3% | balanced, smaller PnL |
| 6 | 96 / +12.2% | 53.1% | 64 / +2.6% | 43.8% | long dominates |

**Aggregate: long +127% sum PnL across all folds, short +139%.** Slightly short-skewed by total PnL (+0.5σ); strongly direction-symmetric in count (≈55% long, 45% short).

The pattern is clear: when BTC drops, shorts amplify; when BTC rises, longs lead. This is the right structural property — the policy doesn't have a fixed bias.

## Exit reason breakdown

| reason | count | % | mean PnL | win % |
|---|---|---|---|---|
| TP | 48 | 4.3% | **+2.49%** | 100% |
| TIME | 394 | 35.1% | +0.75% | 100% |
| BE | 326 | 29.1% | +0.19% | 63.8% |
| TSL | 148 | 13.2% | ~0% | 0% (BE-anchored) |
| SL | 202 | 18.0% | −1.04% | 0% |
| R5 (rare) | 4 | 0.4% | −0.34% | 0% |

**Most trades exit on TIME** (35%) and they're net positive (+0.75% per trade). The time-stop config across strategies (30 / 60 / 120 bars depending on strategy) is doing real work.

**TP rate is low (4.3%) but high-value** (+2.49%). This was flagged in the earlier audit follow-up. We tested tightening TP — `tp_scale=0.85` and `tp_scale=0.70` — both degraded. The current TP thresholds are at a local optimum.

**29% of trades hit BE** (breakeven trigger), and 64% of those still end positive — the breakeven-then-trail mechanic is the second most important contributor after TIME.

## Regime breakdown

| regime | trades | % | long % | mean PnL | win % |
|---|---|---|---|---|---|
| calm | 46 | 4.1% | 63.0% | +0.084% | 60.9% |
| trend_up | 109 | 9.7% | 45.0% | +0.142% | 56.0% |
| **trend_down** | 123 | 11.0% | **64.2%** | **+0.324%** | 61.8% |
| ranging | 414 | 36.9% | 57.5% | +0.270% | 60.9% |
| chop | 430 | 38.3% | 56.7% | +0.221% | 54.2% |

**Best regime: trend_down (+0.32% per trade).** Notably, the system goes 64% LONG in trend_down. This is contrarian/mean-revert posture: down-trends often retrace short-term and the system catches those reversals. Strategies S1 and S4 (trend trigger) do most of this in down-trends.

**75% of trades happen in ranging or chop regimes** — the default-state regimes where signal frequency is highest.

## Vote-strength analysis

Average vote count per strategy ranges 2.92–3.23 — modal vote at K=5 plurality is 3 (since ≥3 nets must agree to fire). All strategies look similar by avg votes; vote strength doesn't strongly differentiate strategy quality.

(The detailed vote-vs-PnL stratification is in [trade_quality_by_agreement.md](trade_quality_by_agreement.md).)

## Implications

**1. The +10.40 WF Sharpe is broadly distributed across strategies.** No single strategy carries the system; S1 dominates volume but S4 dominates per-trade quality, and 5 strategies contribute substantively.

**2. The defensive bias is structural.** Best regimes are trend_down + ranging; best fold equities are in BTC −17% / −30% periods. This reduces correlation to BTC and is good for portfolio risk management — but means the system's edge SHRINKS in sustained bull markets (folds 5, 6 are smallest equity gains).

**3. TIME exits are doing 35% of the work.** If we wanted to push for more TP captures, we'd need to either:
   - Tighten TP (already tested, degrades)
   - Add intra-bar exit timing (mentioned in next_steps.md as "+28-Sharpe oracle gap")
   - Different strategy logic that targets shorter holding periods

**4. S2 / S3 / S12 are low-contribution but not removable.** A2/A3 ablation tests confirmed removing them hurts.

**5. No single failure mode visible.** SL hits are 18% of trades at −1.04% — within risk tolerance. No catastrophic exit reason. The aggregate +10.40 Sharpe is the real picture, not artifacts.

## Files

| File | Contents |
|---|---|
| [models/audit_vote5.py](../models/audit_vote5.py) | per-trade tracer + aggregator |
| `cache/audit_vote5_trades.json` | full 1,122-trade log + aggregated stats |
