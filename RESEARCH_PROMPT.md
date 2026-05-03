# Research Prompt — Crypto Trading ML

## Project

Crypto trading ML research targeting OKX, BTC/ETH/SOL, 1-minute bars.
Phase: research-first — validate signal with honest out-of-sample methodology before any production deployment.

**Repo:** `petrovichp/test_project` at `/Users/petrpogoraev/Documents/Projects/test_project`

---

## Data (preprocessed — never touch raw CSVs)

**BTC meta** — `cache/okx_btcusdt_spotpepr_20260425_meta.parquet` — 384k rows, 29 columns:
`timestamp, oi_usd, fund_rate, spot_ask_price, spot_bid_price, perp_ask_price, perp_bid_price,
span_spot_price, span_perp_price, spot_minute_volume, perp_minute_volume, spot_sell_buy_side_deals,
perp_sell_buy_side_deals, spot_spread_bps, spot_imbalance, spot_bid_concentration, spot_ask_concentration,
spot_large_bid_count, spot_large_ask_count, perp_spread_bps, perp_imbalance, perp_bid_concentration,
perp_ask_concentration, perp_large_bid_count, perp_large_ask_count, taker_sell_buy_ratio,
taker_sell, taker_buy, diff_price`

**BTC orderbook** — `cache/okx_btcusdt_spotpepr_20260425_ob.parquet` — 384k rows, 801 columns:
`timestamp` + `spot/perp bids/asks amount_0..199`

ETH and SOL meta parquets available at same path (`okx_ethusdt_...` / `okx_solusdt_...`).

**OB encoding:** bins 0–199 are equal-width price buckets anchored at best bid/ask
(bin 0 = closest to mid, bin 199 = furthest). Amounts normalized by `spot_ask_price`.
`span_spot_price` / `span_perp_price` = dollar range covered by bins. Currently missing from baseline — must be added.

---

## Infrastructure (already built)

| Module | What it does |
|---|---|
| `data/loader.py` | `load_meta(ticker)` / `load(ticker)` — loads from parquet cache |
| `data/gaps.py` | `clean_mask(timestamps, max_lookback)` — flags gap-contaminated rows |
| `models/splits.py` | `sequential(n, 0.50, 0.25)` and `walk_forward(ts, 90d, 30d, 30d)` → 6 folds |
| `models/volatility.py` | Volatility research — run with `python3 -m models.volatility [ticker]` |

**Caching:** save all expensive intermediate results to `cache/` as `.parquet` or `.npy`. Always check cache before recomputing.

---

## Strict No-Leakage Rules

- All features use only bars `[t - lookback, t]`. Never `shift(-n)` on features.
- `shift(-n)` for label construction only, nothing else.
- `StandardScaler` / any normalization: fit on train split only, transform val/test.
- No `BatchNorm` over full dataset — use `LayerNorm` or `InstanceNorm` in DL models.
- Cross-asset features (BTC → ETH/SOL): lag the predictor by 1+ bars.
- Rolling windows: `min_periods=full_window` — NaN-fill early bars, exclude from samples.
- Embargo: gap of `label_length` bars between train end and val/test start.
- Walk-forward splits only. Never shuffle time-series data.

---

## Volatility Research (done — extend this)

`models/volatility.py` explores 3 target types × 5 horizons using LightGBM.

**Current results on BTC (val set, Spearman correlation):**

| Target | H=15 | H=30 | H=60 | H=100 | H=240 |
|---|---|---|---|---|---|
| ATR | **0.727** | **0.724** | 0.692 | **0.704** | 0.591 |
| Realized vol | 0.696 | 0.674 | 0.628 | 0.617 | 0.448 |
| Price range | 0.614 | 0.554 | 0.493 | 0.445 | 0.265 |

**Key finding:** ATR is the most predictable target across all horizons. Short horizons (15–30 bars)
are more predictable than long ones. Directional accuracy (top-tercile identification) reaches 78–79%
for ATR at 15–30 bars.

**Next steps for volatility research:**
- Run walk-forward validation on the best combinations (ATR H=15, ATR H=30, realized_vol H=15)
  to confirm signal is consistent across market regimes, not just one period
- Add OB features (`span_spot`, `span_perp`, OB bucket imbalance, OB velocity) — currently not used
- Try quantile regression instead of MSE regression: predicting the 75th/90th percentile of vol
  is more useful than predicting the mean (better for TP/SL sizing)
- Test on ETH and SOL — does the same feature set transfer across assets?
- Multi-output model: predict all horizons simultaneously (shared encoder, separate output heads)
- SHAP analysis: which features drive volatility prediction? Expected: current realized vol,
  OI velocity, taker imbalance, spread BPS

---

## Direction Prediction Research (to build)

Build `features/` modules first, then train direction models.

### Feature Engineering

All features backward-looking only. Ablate — measure AUC contribution per group.

**Orderbook** (`features/orderbook.py`):
- Baseline buckets [0–50, 50–100, 100–200] amounts per side (spot + perp)
- `span_spot` and `span_perp` as scalars
- Per-bucket imbalance: `(bid - ask) / (bid + ask)` per bucket
- OB velocity: diff of bucket amounts between consecutive snapshots
- Cumulative depth curve: cumsum of bins from mid outward
- Try finer bucketing [0–20, 20–50, 50–100, 100–200] and 1D conv over all 200 levels

**Price & Momentum** (`features/price.py`):
- Log returns at 1, 5, 15, 30, 60, 120, 240 bars
- SMA, EMA at 5, 10, 20, 50, 100, 200 bars; close/SMA ratio
- MACD (12/26/9), RSI (6, 14), ROC (5, 10, 20), Stochastic (5, 14)
- Bollinger %B and band width (20 bars)
- Rolling VWAP (60, 240, 1440 bars) and deviation from VWAP
- Perp-spot basis bps, basis z-score (60, 240 bar window), basis momentum

**Volatility** (`features/volatility.py`):
- Rolling std at 5, 15, 30, 60, 240, 1440 bars
- ATR (5, 14), Garman-Klass (10, 20, 60)
- Vol of vol, vol ratio short/long, volatility regime bucket (3-class)
- Use predicted volatility from `models/volatility.py` as a feature — the model's own
  vol forecast is informative for direction (high predicted vol = larger expected moves)

**Volume & Taker** (`features/volume.py`):
- Volume z-score (20, 60 bars)
- Taker imbalance at 1, 5, 15, 30 bars
- Cumulative taker net (rolling sum buy - sell)
- OBV (on-balance volume)
- OFI (order flow imbalance): change in best bid/ask quantity between snapshots

**Market & Derivatives** (`features/market.py`):
- OI velocity (1, 5, 15, 60 bars), OI z-score (60, 240, 1440 bars)
- Funding rate, rolling mean/std (last 3, 8, 24 settlements), funding momentum
- OI-price divergence (leading indicator for squeeze/unwind)

**Cross-Asset** (`features/cross_asset.py`):
- BTC lag-1 return as feature for ETH/SOL
- Rolling correlation BTC/ETH, BTC/SOL (60, 240, 1440 bars)
- Relative strength ratio

**Calendar** (inside `features/market.py`):
- Hour and day-of-week as sin/cos pairs — never raw integers
- Session flags: Asia (00–08 UTC), London (07–16 UTC), NY (13–21 UTC)

**Regime** (`features/regime.py`):
- ADX (14 bars): ranging / trending / strong trend bucket
- Autocorrelation lag-1 of returns (20, 60 bars)
- Amihud illiquidity: `abs(return) / dollar_volume`, rolled 20, 60, 240 bars

### Label Construction

```python
# Price moves >threshold in next H bars — two binary targets
Y_up   = (max(price[t+1:t+H+1]) / price[t] - 1) > threshold   # e.g. +0.8%
Y_down = (min(price[t+1:t+H+1]) / price[t] - 1) < -threshold  # e.g. -0.8%
```

Also try risk-adjusted label: `raw_return / realized_vol` — normalises for regime.

### Two-Stage Pipeline (final form)

```
Volatility model → predicted ATR
        ↓
Direction model (uses predicted ATR as a feature)
        ↓
Trade signal: direction × confidence × vol-adjusted size
```

The volatility prediction feeds into the direction model as a feature — high predicted ATR means
larger expected moves, which helps calibrate the confidence threshold for entry.

---

### Model Training Plan

#### Volatility Models — predict *how much* price will move

Used for position sizing and dynamic TP/SL.

| Model | Target | Horizons | Status |
|---|---|---|---|
| LightGBM regressor | ATR | 15, 30, 60, 100 bars | Done — Spearman 0.70–0.73 |
| LightGBM regressor | Realized vol | 15, 30, 60, 100 bars | Done — Spearman 0.62–0.70 |
| LightGBM quantile (75th, 90th) | ATR | 15, 30 bars | Next — better for TP/SL sizing |
| LightGBM + OB features | ATR | 15, 30 bars | Next — add span, imbalance, velocity |
| Multi-output LightGBM | ATR all horizons | 15, 30, 60, 100 | Later — shared features, all horizons at once |

#### Direction Models — predict *which way* price will move

Binary signal: will price go up / down more than X% in the next H bars.
**Do not build DL models until LightGBM confirms real signal on walk-forward.**

**Stage 1 — Tabular baselines**

| Model | Input | Notes |
|---|---|---|
| LightGBM | Flat ~200–300 features per bar | Primary baseline. SHAP for leakage audit. |
| CatBoost | Same + calendar as native categoricals | Compare AUC vs LightGBM |

Params: `num_leaves=64`, `min_data_in_leaf=100`, `feature_fraction=0.7`, early stopping on time-ordered val.
Gate to Stage 2: walk-forward AUC > 0.52 on 5+ of 6 folds.

**Stage 2 — Deep learning**

| Model | Input shape | What it captures |
|---|---|---|
| Multi-input DNN + LSTM branches | `(batch, lookback, n_features)` per group | Temporal structure per feature group |
| CNN-LSTM hybrid | `(batch, 60, ~25 features)` | Local patterns → temporal evolution. Best DL baseline for 1-min crypto. |
| DeepLOB variant | `(batch, 100, 40 OB levels)` | OB spatial + temporal structure. Top 20–50 bins only. |

CNN-LSTM: `CausalConv1D(64, kernel=5, padding='causal')` → `GRU(128)` → Dense → sigmoid.
DeepLOB: Conv2D blocks → Inception module → LSTM(64) → sigmoid.

**Stage 3 — Ensemble**

| Model | Composition | Weights |
|---|---|---|
| Weighted average | LightGBM + CatBoost + CNN-LSTM + DeepLOB | From walk-forward Sharpe |

Evaluate: does ensemble consistently beat best single model across all 6 folds?

### Validation Protocol

1. LightGBM sequential split → compare AUC to original notebook (leaky). Quantify the gap.
2. Walk-forward LightGBM → per-fold AUC. Gate check: 5+ of 6 folds > 0.52.
3. DL models → compare walk-forward AUC vs LightGBM baseline.
4. Ablation: remove one feature group at a time, rank by AUC contribution.
5. Ensemble → does it improve on best single model?

---

## Code Style

- No docstring walls. Comments only for non-obvious decisions.
- No random shuffling of time-series data anywhere.
- Parquet cache for everything expensive.
- Run modules directly: `python3 -m models.volatility btc`
