# P1 (revised) — exit-reason fee model: maker-on-TP

Executed 2026-05-13 as a redesign of the original P1 maker-fill feasibility
study after the user proposed an exit-side simplification.

**Verdict: NEGATIVE for current configuration.** Treating TP exits as maker
fills (2bp instead of 4.5bp) lifts WF Sharpe by only **+0.04** for both
deployables. The binding constraint is TP-hit rate (2.8-3.4%), not the
maker-taker fee differential.

## Original plan vs revised plan

The original P1 plan tried to estimate maker fill rates at trade ENTRY
from 1-min OB snapshots. That plan had four major flaws:
1. No intra-bar OHLC data — couldn't tell if bid was actually touched
2. Adverse selection unhandled (maker fills bias toward unfavorable trades)
3. Per-strategy fill timescales ignored
4. Arbitrary +5 Sharpe decision threshold

User-proposed reframe: model the EXIT side instead. If price reaches our TP
during the trade, we can assume the exit was a maker fill (limit order placed
at TP fires passively when price comes up). Other exit types (SL, TSL, BE,
time) are urgent → taker.

This is a **clean simplification** because the simulator already reports
exit reasons. No fill-rate estimation needed.

## Model

| Side | Fee assumption |
|---|---|
| **Entry** | 4.5bp (taker) — current assumption, unchanged |
| **Exit via TP** | **2bp (maker)** — limit fills passively |
| Exit via SL / TSL / BE / TIME / EOD | 4.5bp (taker) — urgent market exit |

OKX BTC perp fees: maker 0.020%, taker 0.045%. Maker savings per TP-exit
trade = 4.5 − 2 = **2.5bp** added to PnL.

## Method

1. Run walk-forward simulator at uniform fee=4.5bp with `with_reason=True`
   to get realistic trade sequences with per-trade exit reasons.
2. Baseline reproduces Z5.3 numbers (sanity check):
   - DISTILL_v8 −0.549 (Z5.3: −0.5486 ✓)
   - VOTE5_v8 K=5 +4.582 (Z5.3: +4.58 ✓)
3. For each TP-exit trade, refund 2.5bp to PnL.
4. Reconstruct equity curves per fold; compute Sharpe.

Code: [models/mixed_fee_eval.py](../../models/mixed_fee_eval.py).
Output: `cache/results/mixed_fee_eval.json`.

## Results

### Aggregate

| Policy | n_trades | TP-rate | Uniform 9bp WF | **Mixed WF** | Δ |
|---|---:|---:|---:|---:|---:|
| DISTILL_v8_seed42 | 1,561 | 2.8% | −0.549 | **−0.505** | **+0.044** |
| VOTE5_v8_H256_DD K=5 | 1,394 | 3.4% | +4.582 | **+4.618** | **+0.036** |

### Per-fold delta is uniformly tiny

| | f1 | f2 | f3 | f4 | f5 | f6 |
|---|---:|---:|---:|---:|---:|---:|
| DISTILL_v8 Δ | +0.039 | +0.021 | +0.075 | +0.046 | +0.064 | +0.018 |
| VOTE5_v8 Δ | +0.026 | +0.023 | +0.022 | +0.031 | +0.064 | +0.050 |

No fold lifts by more than +0.075 Sharpe. The improvement is essentially
within noise.

### Exit reason distribution explains why

| Reason | DISTILL_v8 | VOTE5_v8 |
|---|---:|---:|
| TIME stop | 40.2% | 40.8% |
| TSL (trailing) | 25.8% | 26.6% |
| SL | 20.3% | 17.6% |
| BE (break-even SL) | 10.4% | 11.3% |
| **TP** | **2.8%** | **3.4%** |
| EOD | 0.4% | 0.3% |

The trade-management profile in [execution/config.py](../../execution/config.py)
intentionally avoids fixed TP exits for trend strategies (`time_stop_bars=0`
+ `trail_after_breakeven=True` for S1, S4, S6, S8, S10). Most positions
either ride to a trailing stop or hit a time limit before reaching TP.

### Per-strategy TP-hit rate

DISTILL_v8 per-strategy (sorted by trade volume):

| strategy | n | TP rate | sum_uniform% | sum_mixed% | Δ% |
|---|---:|---:|---:|---:|---:|
| S1_VolDir | 390 | 2.6% | +41.31 | +41.56 | +0.25 |
| S10_Squeeze | 350 | **0.0%** | −37.35 | −37.35 | +0.00 |
| S7_OIDiverg | 277 | 1.4% | −6.25 | −6.15 | +0.10 |
| S8_TakerSus | 270 | 6.7% | +1.01 | +1.46 | +0.45 |
| S11_Basis | 120 | 6.7% | +1.52 | +1.72 | +0.20 |
| S4_MACD | 50 | 2.0% | +18.77 | +18.80 | +0.03 |
| S6_TwoSignal | 40 | 2.5% | −9.62 | −9.60 | +0.02 |
| S3_BBExt | 25 | 0.0% | −3.00 | −3.00 | +0.00 |
| S13_OBDiv | 25 | 0.0% | +1.16 | +1.16 | +0.00 |
| **S2_Funding** | 14 | **14.3%** | +1.62 | +1.67 | +0.05 |

Strategies with tighter TP + no trail (mean-reversion: S2, S8, S11) hit TP
most often. Strategies with trailing stops (S1, S4, S6, S10) almost never
hit TP — they get caught by the trail first.

## Interpretation

The savings ceiling is bounded by:

```
Average maker savings = (TAKER - MAKER) × TP_hit_rate = 2.5bp × 0.03 ≈ 0.08 bp per trade
```

That's **two orders of magnitude smaller** than the per-trade alpha needed
to materially shift WF Sharpe. Even at 100% TP-hit rate (impossible), the
ceiling would be 2.5bp × 1.0 = 2.5bp savings, which would lift Sharpe by
~5-6 (using Z5.3 fee-sensitivity slope) — that's the absolute theoretical
maximum. With realistic 3% TP-hit, we capture <2% of that ceiling.

## What this changes about deployment

| Question | Answer |
|---|---|
| Should we deploy DISTILL_v8 at taker pricing with maker-on-TP? | No — the −0.5 Sharpe doesn't move to viability. |
| Should we deploy VOTE5_v8 K=5 with maker-on-TP refinement? | Marginal — +0.04 Sharpe doesn't change deployment economics. |
| Is the Z5.4 freeze decision affected? | No. DISTILL stays maker-only deployable, VOTE5 stays taker deployable. |
| Is maker-on-TP worth implementing in production? | **Yes, trivially** — it costs nothing to send TP as a limit instead of market. The +0.04 Sharpe is real money long-term even if not transformative. |

## What this unlocks for future research

The bottleneck is TP-hit rate, not the fee differential. If we tightened TPs
(audit_followup task #4 in [PLAN.md](../../PLAN.md)), TP-hit rate might rise
to 30-50%, and maker-on-TP savings could become material (~1bp avg = 2-3
Sharpe). But tighter TPs also mean smaller per-winner profit — net effect
unclear.

**The natural next experiment is joint: tighten TPs by 30-50% AND apply
maker-on-TP simultaneously, evaluate Sharpe at OKX taker.** Likely to either
lose money (smaller winners overwhelm fee savings) or gain a real edge
(many small maker wins compound).

## Code touchpoints

- [models/mixed_fee_eval.py](../../models/mixed_fee_eval.py) — the runner
- [models/diagnostics_ab.py:40](../../models/diagnostics_ab.py#L40) `_simulate_one_trade_fee` — reused
- [models/analyze_a2_rule.py:42](../../models/analyze_a2_rule.py#L42) `_simulate_one_trade_fee_with_reason` — exit-reason variant used here

## Outputs

- `cache/results/mixed_fee_eval.json` — per-policy detail
- This document
