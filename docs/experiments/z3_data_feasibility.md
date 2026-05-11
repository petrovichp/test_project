# Z3 strategy feasibility — data check

Verifying whether the four proposed Z3 strategies (S13–S16 in [development_plan.md](../reference/development_plan.md)) can be implemented with **only** the existing cached parquets.

**Data used**:
- `cache/raw/okx_btcusdt_spotpepr_20260425_meta.parquet` — 384,614 rows × 29 cols
- `cache/raw/okx_btcusdt_spotpepr_20260425_ob.parquet` — 384,614 rows × 801 cols (200 bins × 2 sides × 2 instruments + timestamp)

## Per-strategy data availability

| proposed | required field | source | status |
|---|---|---|---|
| **S13_FundingExtreme** | `fund_rate` | meta col | ✓ available, but extremes very rare in this dataset |
| **S14_DepthImbalance** | OB top-of-book sizes + depth at ±2% + microprice | OB parquet (`*_amount_0` to `*_amount_199`) + meta `spot_imbalance` | ✓ fully available |
| **S15_VolBreakout** | ATR_30 / median(ATR_60) ratio + 10-bar direction | computable from `spot_*_price` + `perp_*_price` | ✓ fully computable |
| **S16_BasisDislocation** | spot mid, perp mid, rolling z-score of basis | meta cols `spot_*_price`, `perp_*_price` | ✓ fully available |

**All 4 strategies have data.** Implementation is not blocked by missing inputs.

## Data-driven calibration findings

### S13 — funding rate distribution is much narrower than I described

The dataset (Sep 2025 – Apr 2026) covers a relatively calm funding period:

| quantile | per-bar `fund_rate` | annualized APR |
|---|---:|---:|
| 0.005 | −0.000059 | **−10.30%** |
| 0.05  | −0.000029 | −5.04% |
| 0.50  | +0.000029 | +5.09% |
| 0.95  | +0.000063 | +10.95% |
| 0.995 | +0.000063 | +10.95% |
| max   | +0.000225 | **+24.6%** |
| min   | −0.000561 | **−61.4%** |

**Bars exceeding ±15% APR**: 0.34% of all bars. Effective fire rate: a few hundred trades across ~6 months.

⚠ The "+50% APR for weeks" scenarios that classically motivate this strategy don't appear in this period. Threshold needs to be lower (e.g. > +10% APR / < −5% APR for top/bottom 5%) — but at those thresholds, the signal is barely "extreme" anymore.

**Verdict: technically implementable, but the dataset's funding range is too narrow for the strategy's classical edge thesis. Likely produces few trades with marginal alpha.**

### S14 — pre-computed imbalance already exists in features

The meta parquet **already includes** these pre-computed columns I was going to recompute from raw OB:

| col | range | use |
|---|---|---|
| `spot_imbalance` | [−0.80, +0.81] | direct depth imbalance (spot) |
| `perp_imbalance` | [−0.62, +0.62] | direct depth imbalance (perp) |
| `spot_bid_concentration`, `spot_ask_concentration` | — | further depth nuance |
| `spot_large_bid_count`, `spot_large_ask_count` | — | large-order proxy |

**Bars with `|spot_imbalance| > 0.4`: 4.25%.**

⚠ Top-of-book amount columns (`*_amount_0`) exist but are normalized — microprice computation from them gives near-zero drift (q95 = 0.0001 bps). Either the normalization removes microprice information, or microprice signal at 1-min granularity is genuinely tiny. Should rely on `spot_imbalance` directly, not raw OB-bin reconstruction.

**Verdict: fully implementable using pre-computed features.** Don't recompute from raw OB — the meta parquet already provides the right summary.

### S15 — vol-ratio threshold needs tuning

Computed `vol_ratio = ATR_30 / median(ATR_60)` from `(spot_ask + spot_bid) / 2`:

| threshold | bars firing |
|---:|---:|
| > 1.5 | **25.08%** ← my proposal, way too noisy |
| > 1.8 | 10.48% |
| > 2.0 | 6.38% |
| > 2.5 | 2.52% |
| > 3.0 | 1.32% |

⚠ My initial threshold of 1.5 fires on a quarter of all bars — would crowd out the action space. Better: **`vol_ratio > 2.0`** (~6% of bars) paired with strict 10-bar direction filter.

Also: a few zero-variance ATR_60 windows produce `inf`. Need to handle with `replace(0, np.nan)`.

**Verdict: fully implementable with corrected threshold (1.5 → 2.0).**

### S16 — basis is consistently negative in this period

Basis = (perp_mid − spot_mid) / spot_mid:

| quantile | basis (bps) |
|---:|---:|
| 0.05 | −6.45 |
| 0.50 | **−4.55** (median is negative) |
| 0.95 | −0.34 |
| 0.99 | +4.02 |

Mean basis = **−4.23 bps**, std = 2.09 bps. Bars with `|z| > 2.0`: 5.82%.

⚠ Perp consistently trades *below* spot. This is the opposite of typical bull-market basis (perp at premium). Possible reasons: bear regime, OKX-specific structure, or convention difference between exchanges. The mean-reversion edge still exists — z-score normalization handles the offset — but the strategy's directional asymmetry tilts toward "short perp when relatively expensive vs the negative-basis baseline."

**Verdict: fully implementable. The negative-basis baseline is a feature, not a bug — z-scoring removes it.**

---

## ⚠ Major finding: 3 of 4 proposed strategies overlap with code that already exists but is unused

The strategies module [strategy/agent.py](../strategy/agent.py) defines **13 strategies** (`strategy_1` through `strategy_13`), but the DQN action space only registers **9**. Four strategies are coded but never wired into `STRAT_KEYS`:

| existing function | description (from docstring) | overlaps with my proposed |
|---|---|---|
| `strategy_5` | "Long on abnormal positive OFI spike + taker confirms + RSI not overbought" | partial overlap with S14 (microstructure flow) |
| `strategy_9` | "Net large-order imbalance (bid count − ask count) as institutional proxy" | **strong overlap with S14_DepthImbalance** |
| `strategy_11` | "diff_price z-score (spot ask − perp bid) as basis momentum signal" | **strong overlap with S16_BasisDislocation** |
| `strategy_13` | "Spot and perp order book imbalances disagree → spot leads, perp follows" | partial overlap with S14 + S16 |

**Also**, the existing **`S2_Funding`** (already in DQN action space) does:
> "Short when funding extreme positive + MACD weakening."

This is a **filtered version of my proposed S13_FundingExtreme** (S2 adds MACD-weakening as a secondary filter). My proposal is the unfiltered variant.

## Revised Z3 plan based on findings

The cheapest path is **wire existing-but-unused strategies first**, then add only the genuinely-novel signals:

| ID | Action | Cost | Justification |
|---|---|---|---|
| **Z3.1** | Wire `S5_OFISpike`, `S9_LargeImbalance`, `S11_BasisMomentum`, `S13_OBDisagreement` into `STRAT_KEYS` (action space 10 → 14). Run standalone validation to filter weak ones. Retrain VOTE5_v8. | ~1 day | Code exists, no new logic, tests existing-but-untried alpha |
| **Z3.2** | Add `S15_VolBreakout` (genuinely new signal, ~6% fire rate at `vol_ratio > 2.0`) | ~0.5 day | No equivalent exists |
| Z3.3 (deferred) | Original `S13_FundingExtreme` | — | Subsumed by existing `S2_Funding`; data shows extremes too rare |
| Z3.4 (deferred) | Custom `S14_DepthImbalance` | — | Subsumed by `S9_LargeImbalance` once wired in |

This compresses the original 4-strategy proposal into **2 work items** at ~1.5 days total instead of ~3.5 days. Net result: **same alpha-discovery surface for less code.**

## Recommended order

1. **Z3.1** — wire the 4 existing-unused strategies. Run `backtest/run.py` standalone for each to confirm they produce signal worth feeding to DQN (>50% win rate, mean PnL > 0.15%/trade on val). Drop weak ones, keep the rest.
2. **Z3.2** — implement S15_VolBreakout as a new strategy_14 in agent.py.
3. Retrain `BASELINE_VOTE5_v8` with the expanded action space (5 seeds, h=64) and compare to current `BASELINE_VOTE5` (WF +10.40).
4. If WF improves by ≥+0.5 with no fold worse than current by >0.5, freeze as new baseline.
