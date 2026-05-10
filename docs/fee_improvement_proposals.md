# Proposals — improve trading under non-zero fees

Saved 2026-05-09 after the fee-aware retrain experiment ([fee_aware_retrain.md](fee_aware_retrain.md)) confirmed that vanilla VOTE5 + post-hoc filter (WF +3.72 at 4.5 bp) still beats fee-baked-in retraining. Per-trade alpha is ~0.20%; round-trip taker fee is 9 bp = 0.09%; ~50% of edge is consumed by fees.

This is the parking lot of improvements. **Status: deferred — proceeding with zero-fee algo improvements first.**

## 1. Reduce the fee itself (highest impact)

| sub-idea | cost | leverage | notes |
|---|---|---|---|
| Maker-only execution (Path X) | engineering, ~1 wk | 5 bp/round-trip saved (taker 4.5 → maker 2.0) | place entries as post-only at best bid/ask; needs OB-fill simulation on existing depth data |
| Maker-on-entry / taker-on-exit hybrid | engineering, ~3 d | 2.5 bp/round-trip | TP/SL/trail in stressed conditions often won't fill post-only; accept taker for exits |
| Funding-rate offset on perp | ~2 h | flips marginal trades net-positive | `fund_rate` is in features but not in reward; add `fund_rate × bars_held / 525960` to per-trade pnl |
| OKX VIP tier scaling | volume-driven | gradual fee reduction | irrelevant until live capacity sustained |

**Scoping experiment**: simulate post-only fill-rate at best bid/ask across existing OB data. If >70% fill within 1 bar, maker path is viable.

## 2. Concentrate capital where confidence is highest (no retrain)

A2 audit ([trade_quality_by_agreement.md](trade_quality_by_agreement.md)) showed 5-vote trades have ~3× mean PnL of 3-vote trades. Currently all trades sized equally.

| sub-idea | cost | expected lift @ 4.5 bp | notes |
|---|---|---:|---|
| Vote-strength sizing | ~1 h | +0.5 to +1.5 Sharpe | size = {3v: 0.4, 4v: 0.7, 5v: 1.0} |
| Q-margin threshold | ~1 h | +0.3 to +0.8 | post-hoc calibrate `Q[a*] − Q[no_trade] ≥ τ` per net before voting |

## 3. Tighter TP for trend strategies (audit follow-up #4)

Audit found TP-hit rate is only 3-5% — most trades resolve via TSL/TIME/SL/BE. At fees, slow-bleed trades drag harder than at fee=0. Test 4 from [audit_followup_tests.md](audit_followup_tests.md) — still not run.

```
python3 -m models.group_c2_walkforward --tp-scale 0.85 --out-tag test4a_tp0.85
python3 -m models.group_c2_walkforward --tp-scale 0.70 --out-tag test4b_tp0.70
```

Expected: TP-hit rate ↑, TSL-captures ↓, ~+1 to +2 Sharpe @ 4.5 bp.

## 4. Architectural training fixes

| sub-idea | cost | risk | notes |
|---|---|---|---|
| Fee as state feature, train at random fee ∈ {0, 2, 4.5, 8} bp | ~1 day | medium | `state[20] = fee_bp / 10`; one model generalizes across execution paths |
| Predict `E[pnl − 2×fee]` directly via regression | ~2 days | medium | skip Q-learning; trade only if predicted net edge > 0 |
| Distributional RL + CVaR action selection | ~2 days | high (research bet) | replace `argmax E[Q]` with `argmax CVaR_{0.7}[Q]`; naturally penalizes high-variance trades |

## 5. Asymmetric exits by regime

Regime in state vector already. Trending regimes warrant wider TP (let winners run); chop wants tighter (faster scratches). `tp_scale[regime] × base_tp`. ~2 h.

## Recommended order (when we return to fee work)

| # | idea | cost | expected Sharpe lift @ 4.5 bp | risk |
|---|---|---|---:|---|
| 1 | Vote-strength sizing | ~1 h | +0.5 to +1.5 | low |
| 2 | Tighter TP for trend strategies | ~30 min + analysis | +1 to +2 | low |
| 3 | Funding-rate in reward | ~2 h | +0.5 (asymmetric, helps shorts in bear funding) | low |
| 4 | Maker-fill-rate scoping | ~2 h | none directly, unlocks +3 to +5 if viable | engineering |
| 5 | Q-margin threshold calibration | ~1 h | +0.3 to +0.8 | low |
| 6 | Fee-as-state retrain | ~1 day | unclear — could be +1 or 0 | medium |
| 7 | CVaR action selection | ~2 days | +0.5 to +2 | high research bet |

Combined #1+#2+#3 (~4 h) is the cheapest batch with bounded downside. **#4 is the only path that breaks the fee ceiling**; everything else is bounded above by it.
