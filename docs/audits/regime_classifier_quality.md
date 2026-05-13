# Regime classifier (CUSUM+Hurst v4) — quality audit

Executed 2026-05-13 as part of the A1/A2/A3 audit sweep.

**Verdict: WEAK / NON-STATIONARY.** The classifier carries statistical
information (Kruskal-Wallis p < 1e-6 at all horizons) but exhibits two
serious defects:

1. **Sign-flipping across splits** for 3 of 5 regimes (trend_up, trend_down,
   ranging). The regime → forward-return relationship learned on train is
   not the relationship present on test.
2. **High flicker rate** — 9.7 transitions per 100 bars; **48.8% of runs
   are shorter than 5 bars**. Many regime labels are sub-5-minute noise.

The "trend_up" / "trend_down" / "chop" / "ranging" *names* are misleading:
on train data, trend_up predicts **mean-reversion** (forward return
−1.19 bp), not trend continuation.

## Methodology

Loaded `cache/preds/btc_regime_cusum_v4.parquet` (383,174 bars, 5 states).
Tested four quality dimensions:

1. Dwell-time distribution per regime
2. Flicker rate + short-run fraction
3. Distribution stability across train / val / test splits
4. Regime label → forward log return statistical relationship at multiple
   horizons (10 / 30 / 60 / 240 min)

## Findings

### 1. Dwell times (one full dataset pass)

| regime | n_runs | mean | median | p25 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| calm | 5,269 | 15.4 | 8 | 3 | 17 | 53 | 393 |
| trend_up | 3,469 | 9.1 | 4 | 1 | 11 | 35 | 125 |
| trend_down | 3,613 | 8.7 | 4 | 1 | 11 | 34 | 85 |
| ranging | 9,512 | 12.0 | 5 | 1 | 14 | 53 | 243 |
| chop | 15,292 | 8.2 | 4 | 1 | 11 | 29 | 129 |

**Median dwell for trend_up / trend_down / chop is 4 bars.** 25th percentile
is 1 bar (instant flip).

### 2. Flicker rate

- Total transitions: 37,154 across 383,174 bars
- **9.70 transitions per 100 bars**
- **48.8% of runs are <5 bars** — half the labels are noise-floor flicker

### 3. Distribution stability across splits

| regime | train% | val% | test% | Δ val−train | Δ test−train |
|---|---:|---:|---:|---:|---:|
| calm | 21.1% | 8.2% | 17.4% | **−12.8%** | −3.7% |
| trend_up | 7.7% | 9.0% | 6.8% | +1.3% | −0.9% |
| trend_down | 7.5% | 11.1% | 8.5% | +3.6% | +1.0% |
| ranging | 31.2% | 32.5% | 33.0% | +1.2% | +1.8% |
| chop | 32.5% | 39.1% | 34.2% | **+6.7%** | +1.7% |

val has 13 pp **less** calm and 7 pp **more** chop than train. The policy
sees very different regime mixes across splits.

### 4. Regime → forward return (the key finding)

Mean forward log return (bps) per regime per split, horizon = 60 min:

| regime | train (bp) | val (bp) | test (bp) | sign flip? |
|---|---:|---:|---:|:---:|
| calm | +0.10 | −0.16 | −0.71 | — |
| **trend_up** | **−1.19** | **+0.57** | **+6.25** | **FLIP** |
| **trend_down** | **−1.59** | **+0.93** | **+2.94** | **FLIP** |
| **ranging** | **−2.58** | **+3.21** | **+1.75** | **FLIP** |
| chop | −2.63 | −1.08 | −0.49 | — |

On train: trend_up means "recent up move just happened → expect
mean-reversion (negative forward return)." On test: trend_up means
the opposite. **The policy was trained on regime-conditional
relationships that don't generalize.**

### 5. Statistical informativeness (aggregate)

Kruskal-Wallis test of "does regime label explain ANY variance in forward
returns" (combining all splits):

| horizon | H | p-value | verdict |
|---|---:|---:|---|
| 10m | 74.98 | < 1e-6 | informative |
| 30m | 116.12 | < 1e-6 | informative |
| 60m | 168.06 | < 1e-6 | informative |
| 240m | 246.28 | < 1e-6 | informative |

The classifier IS informative on aggregate. But effect sizes are tiny
vs noise (t-stats in 1-13 range, ratio of mean to std ~0.03).

### 6. Autocorrelation (persistence)

| lag k | P(regime_{t+k} = regime_t) | random baseline |
|---|---:|---:|
| 1 | 90.3% | 25.4% |
| 5 | 72.9% | 25.4% |
| 30 | 44.1% | 25.4% |
| 60 | 33.7% | 25.4% |
| 240 | 32.0% | 25.4% |

Strong short-term persistence (90% at k=1) decays to ~32% at 4-hour lag
(close to random baseline). So the regime label is sticky on minute
timescales but loses meaning past ~30-60 minutes.

## Interpretation

### What the classifier IS

- A statistically informative but **noisy and non-stationary** label.
- A representation of *recent* price/volatility behavior — retrospective,
  not predictive.
- The Kruskal-Wallis test confirms regime carries information; the per-split
  sign flips show that the information **decays or inverts over time**.

### What the classifier ISN'T

- A reliable predictor of forward returns (effect sizes are <1% of noise).
- A trend-direction predictor (the "trend_up" label predicts mean-reversion
  on train data).
- Stationary across the dataset's 9-month span.

### Implications for prior findings

| Earlier claim | Revised interpretation |
|---|---|
| "Z4.3 curriculum learning failed because regime not in state" (Z5.1 era) | **Wrong rationale** — regime IS in state. Now also: regime classifier is noisy, so even with the correct rationale, curriculum gating by a noisy label would distort the buffer. |
| "Q1: DISTILL_v8 uses regime info" (2026-05-12 audit) | **Revised**: DISTILL responds to regime one-hot inputs, but those inputs carry low-SNR and non-stationary signal. Regime perturbation cost on DISTILL (−2 to −6 Sharpe) likely reflects the policy having learned train-time regime-conditional patterns that don't fully transfer. Some of DISTILL's val/test variance may trace to this. |
| Z2 Step 3 / curriculum / regime-conditional ideas | **All called into question** — proposals that condition on this classifier inherit its non-stationarity. |

## Recommendations

### Immediate (for the v2 plan)

1. **Kill the Tier-3 T2' "regime-explicit teacher" experiment** — there's
   no stable regime signal to condition on. Any regime-explicit embedding
   trained on this label will inherit the sign-flip non-stationarity.
2. **Consider a state-v10 ablation experiment**: retrain
   `VOTE5_v8_H256_DD` and `DISTILL_v8` with regime dims [2:7] **zeroed
   out**. If the WF/test gap closes vs current baseline, regime info is
   a net negative.
3. **Rename labels in any future docs** — call them `state_0` ... `state_4`
   when reporting findings, not "trend_up" / "trend_down" which carry
   misleading semantic content.

### Longer-term

4. **Investigate alternative regime classifiers** — maybe a vol-regime
   classifier (low/med/high realized vol bucket) or a funding-regime
   classifier would be more stable. Park for a future research bet, not
   tomorrow.
5. **Don't extend regime-related features in state v10** (counter to F2/F3
   in earlier plan).

## What this finding does NOT change

- Z5.4 freeze decision is unchanged: it didn't rest on regime claims.
- Q2 (long/short attribution) is unaffected — that finding stood on its
  own statistical legs.
- DISTILL_v8 + VOTE5_v8 baselines remain the deployable artifacts.

## Outputs

- This document (read-only audit).
- No new code.
