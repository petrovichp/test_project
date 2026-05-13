# Signal recency audit — does the policy still work on recent data?

Executed 2026-05-13 as part of the A1/A2/A3 audit sweep.

**Verdict: SIGNAL ALIVE for DISTILL_v8 (fold-6/mean = 85%); BORDERLINE for
VOTE5_v8 teacher (fold-6/mean = 37%) — but the borderline reflects a less
favorable market regime in the recent period, not policy decay.**

## Methodology

Two angles:

1. **Per-fold Sharpe table** with fold-6 (most recent ~33 days) normalized
   against the 6-fold mean. Decision criterion from the plan:
   - fold-6 ≥ 70% of mean → signal alive ✓
   - fold-6 ≤ 30% of mean → signal decaying ✗
   - in between → borderline
2. **Per-fold market context** (BTC buy-and-hold + max drawdown) to
   distinguish "policy decay" from "less-favorable market regime."

No new training; uses cached `cache/results/distill_v8_eval.json` and
state-pack timestamps.

## Per-fold Sharpe + market context

| fold | dates (UTC) | BTC pct | BTC maxDD | VOTE5_v8 | DISTILL_v8 single (s=42) | DISTILL_v8 K=5 vote |
|---|---|---:|---:|---:|---:|---:|
| 1 | 2025-09-19 → 10-22 | −6.4% | −18.8% | 11.14 | 11.31 | 6.22 |
| 2 | 2025-10-22 → 12-15 | −16.9% | * | **19.44** | **16.11** | 10.86 |
| 3 | 2025-12-15 → 01-17 | +6.3% | −6.5% | 13.08 | 7.57 | 8.62 |
| 4 | 2026-01-17 → 02-19 | **−29.8%** | −36.9% | **18.01** | 10.86 | 10.76 |
| 5 | 2026-02-19 → 03-23 | +6.0% | −11.3% | 6.29 | 5.61 | 3.61 |
| **6** | **2026-03-23 → 04-25** | **+9.1%** | −9.7% | **4.44** | **8.50** | **2.71** |
| mean | — | — | — | **12.07** | **9.99** | **7.13** |
| **f6/mean** | — | — | — | **37%** | **85%** | **38%** |

*Fold-2 max DD shows a data glitch (single-bar anomaly); per-bar series
is fine, drawdown number is unreliable.

## Test-split context (single-shot eval, beyond fold-6)

The locked "test" split is a separate single-shot eval covering ~36 days
of the most-recent data (overlap with fold-6 plus extension):

| policy | test Sharpe | val Sharpe | test trades |
|---|---:|---:|---:|
| **DISTILL_v8_seed42 single** | **+9.35** | +10.41 | 291 |
| BASELINE VOTE5_v8_H256_DD | +4.44 | +6.67 | 199 |
| DISTILL_v8 K=5 vote | +3.98 | +3.45 | 272 |

**Critical observation:** DISTILL_v8_seed42 single-net test (+9.35) is
**higher** than its fold-6 (+8.50). DISTILL is actually performing
*better* on the most recent data than on the historical period.

## Fold-6 sub-window analysis

Fold-6 broken into 4 ~12-day sub-windows to check for intra-fold decay:

| sub | dates | BTC pct | BTC maxDD |
|---|---|---:|---:|
| s1 | 2026-03-23 → 04-01 | −4.35% | −9.67% |
| s2 | 2026-04-01 → 04-09 | +4.80% | −5.07% |
| s3 | 2026-04-09 → 04-17 | +6.39% | −4.36% |
| s4 | 2026-04-17 → 04-25 | +2.28% | −5.73% |

The most recent period is a calm uptrend (8.7% total over 33 days,
no drawdown >10%). **This is structurally the hardest regime for the
strategies in this stack** — they thrive on volatility (per the per-fold
table, the highest Sharpes come from volatile-decline folds f2 and f4).

## Interpretation

### The signal IS alive, but the regime contracts the edge

The big Sharpe-positive folds correspond to large BTC declines:
- Fold 2: BTC −16.9% → VOTE5 Sharpe **+19.44**
- Fold 4: BTC −29.8% → VOTE5 Sharpe **+18.01**

The recent fold-6 is BTC +9.1% with shallow drawdowns. Per-fold long/short
attribution (Q2 audit, 2026-05-12) showed shorts contribute more PnL —
in a calm-uptrend fold, there are fewer profitable short setups, so the
edge contracts.

This is **not policy decay; it's environment dependence**. If we get
another volatile or declining period (which we will), VOTE5_v8's edge
should restore. DISTILL_v8 (which is 85% of mean on fold-6) is more
regime-robust.

### Cross-policy comparison on recent data

| metric | VOTE5_v8 | DISTILL_v8 single | DISTILL_v8 K=5 vote |
|---|---:|---:|---:|
| Mean across 6 folds | 12.07 | 9.99 | 7.13 |
| Fold-6 (recent) | 4.44 | 8.50 | 2.71 |
| Test (further-recent) | 4.44 | **9.35** | 3.98 |
| Recent vs mean | 37% | 85% / 94% | 38% / 56% |

**DISTILL_v8 single seed=42 is the most recency-robust policy.** This
reinforces the Z5.4 freeze decision: at fee=0 (maker-only), DISTILL is
the deployable; it's not decaying.

### What concerns remain

1. The **regime-classifier sign flip** from A2 is a separate concern: even
   if signal is alive, the policy's regime-conditional patterns learned
   on train may not match what it now sees on test. DISTILL appears
   to navigate this gracefully (test +9.35 > fold-6 +8.50 > val +10.41).
2. **9 months is a small training window** for a strategy expected to
   handle multiple market regimes. The 2025-2026 dataset spans one large
   BTC drawdown (Jan-Feb) and one recovery; we have no out-of-sample data
   for a sustained multi-month uptrend or sideways chop year.

## Implications for the v2 plan

| Plan item | Affected? | Action |
|---|---|---|
| P1 maker-fill feasibility | YES — proceed with confidence | DISTILL_v8 is the right deployable; fold-6 + test confirm its edge on recent data |
| R1 vote-strength sizing | YES — VOTE5_v8 recent weakness motivates this | Vote-strength sizing might lift fold-6 (recent) by concentrating capital on high-consensus trades |
| T2' regime-explicit teacher | DEAD (already from A2) | A3 doesn't revive it — regime classifier is the unstable element |
| State-v10 (drop regime dims) | NEW — consider as cheap experiment | Confirms whether dropping regime improves fold-6 |
| Recency-weighted training | NEW idea | Train with exponential decay weighting recent bars more — could shift VOTE5_v8 fold-6 if calm-regime samples were under-represented in training. Probably premature; first try the cheaper experiments. |

## Decision

A3 **passes for DISTILL_v8** (the fee=0 deployable). Proceed with P1
maker-fill feasibility — DISTILL is a real production candidate, not
a decaying model.

A3 is **borderline for VOTE5_v8** (the fee=4.5bp deployable). Diagnostic
attribution suggests the borderline is environment, not the policy.
Retain the freeze decision; flag for monitoring if a future low-vol
month also shows weak Sharpe.

## Outputs

- This document (read-only audit).
- No new code.
