# Crypto Trading ML — Project Context

## What this project is

Research-first ML system for crypto trading on OKX (BTC/ETH/SOL, 1-minute bars).
Goal: validate signal with honest out-of-sample methodology before any production deployment.
Live exchange target: OKX (spot + perp).

## Data

**Never touch raw CSVs.** Use preprocessed parquets in `cache/`:

| File | Rows | Description |
|---|---|---|
| `cache/okx_btcusdt_spotpepr_20260425_meta.parquet` | 384k | 29 market/microstructure columns |
| `cache/okx_btcusdt_spotpepr_20260425_ob.parquet` | 384k | 800 OB amount columns + timestamp |
| `cache/okx_ethusdt_spotpepr_20260425_meta.parquet` | — | ETH (same schema) |
| `cache/okx_solusdt_spotpepr_20260425_meta.parquet` | — | SOL (same schema) |

Meta columns: `timestamp, oi_usd, fund_rate, spot_ask_price, spot_bid_price, perp_ask_price,
perp_bid_price, span_spot_price, span_perp_price, spot_minute_volume, perp_minute_volume,
spot_sell_buy_side_deals, perp_sell_buy_side_deals, spot_spread_bps, spot_imbalance,
spot_bid_concentration, spot_ask_concentration, spot_large_bid_count, spot_large_ask_count,
perp_spread_bps, perp_imbalance, perp_bid_concentration, perp_ask_concentration,
perp_large_bid_count, perp_large_ask_count, taker_sell_buy_ratio, taker_sell, taker_buy, diff_price`

OB encoding: 200 equal-width price bins per side anchored at best bid/ask (bin 0 = closest to mid).
Amounts normalized by `spot_ask_price`. `span_spot_price` / `span_perp_price` = dollar range covered.

Raw CSV source: `/Users/petrpogoraev/Documents/Projects/options_trading/DATA/last_source_data/`
Loader: `data/loader.py` — `load_meta(ticker)` or `load(ticker, include_ob=True)`

## Infrastructure

| Module | Function |
|---|---|
| `data/loader.py` | CSV→Parquet caching. `load_meta('btc')` skips 800 OB cols |
| `data/gaps.py` | `clean_mask(timestamps, max_lookback)` — flags gap-contaminated rows (9.45% missing) |
| `models/splits.py` | `sequential(n, 0.50, 0.25)` and `walk_forward(ts, 90, 30, 30)` → 6 folds |
| `models/volatility.py` | Volatility research. Run: `python3 -m models.volatility btc` |
| `models/direction.py` | LightGBM direction model. Run: `python3 -m models.direction btc` |
| `models/direction_dl.py` | CNN-LSTM. Run: `python3 -m models.direction_dl btc cnn_lstm` |
| `models/ensemble.py` | LightGBM + CNN-LSTM weighted ensemble. Run: `python3 -m models.ensemble btc` |
| `models/evaluate.py` | Confusion matrix comparison across all models. Run: `python3 -m models.evaluate btc` |

Caching rule: save expensive intermediates to `cache/` as `.parquet` or `.npy`. Always check cache before recomputing.

## Non-negotiable rules

- **No leakage**: all features use only bars `[t - lookback, t]`. `shift(-n)` for labels only.
- **No random splits**: time-series data is never shuffled. Sequential or walk-forward only.
- **Normalization**: fit scaler on train split only, transform val/test.
- **Cross-asset features**: lag predictor asset by 1+ bars.
- **Rolling windows**: `min_periods=full_window` — NaN early bars, exclude from samples.
- **Embargo**: leave `label_length` bar gap between train end and val/test start.

## Current research state

### Features pipeline — done
130 features across 4 modules, zero NaN after gap masking (MAX_LOOKBACK=1440).
Cached to `cache/btc_features_*.parquet`. Load via `features/assembly.py`.

| Module | Features | Key contents |
|---|---|---|
| `features/orderbook.py` | 32 | bucket amounts, imbalance, velocity, span |
| `features/price.py` | 51 | returns, SMA/EMA, RSI, MACD, VWAP, basis |
| `features/volume.py` | 17 | taker imbalance/net, vol z-score, OBV, OFI |
| `features/market.py` | 30 | OI, funding rate, spread, calendar, sessions |

Splits after gap masking: Train 70,902 (Jul→Oct 2025) / Val 35,451 (Oct→Dec 2025) / Test 35,451 (Dec 2025→Apr 2026).

---

### Volatility model — done (`models/volatility.py`)
Targets: `atr_H` (avg true range over next H bars, in $) and `realized_vol_H` (std of log returns, %).
Horizons tested: 15, 30, 60, 100, 240. Results cached at `cache/btc_volatility_eval.parquet`.

**Best results (Spearman correlation):**

| Target | Val | Test | Walk-forward (6 folds) |
|---|---|---|---|
| atr_15 | 0.647 | **0.784** | 0.57–0.80, all positive |
| atr_30 | 0.627 | **0.801** | — |
| realized_vol_15 | 0.605 | **0.790** | — |
| atr_60 | 0.582 | 0.764 | — |
| atr_100 | 0.559 | 0.726 | — |

Val underperforms test — val period (Oct–Dec 2025) includes the Nov outage (harder regime).

**Confusion matrix (top-33% high-vol detection, atr_15, test):**
- Precision=0.606, Recall=0.690, F1=0.646 — usable as a vol filter for the strategy layer
- `realized_vol_15` hits Precision=0.879 on test (best for high-confidence vol filtering)

**Top features (consistent across all targets):**
1. `bb_width` — 27–29% of gain (current volatility → future volatility, valid clustering effect)
2. `dow_sin` — 7–9% (weekly seasonality)
3. `hour_sin` / `hour_cos` — 6–9% (intraday session patterns)
4. `session_ny` — 2–4% (NY session 13–21 UTC is highest-vol window)
5. `vwap_dev_1440` / `vwap_1440` / `obv_1440` — 3–4% each (trend regime)
6. `oi_z_1440` / `fund_mean_1440` — 1.5% each (derivatives positioning)
7. OB features and short-term taker flow: near zero — vol is regime+calendar driven, not microstructure

**`bb_width` removal experiment (horizons 15, 30, 60, 100):**
- Removing `bb_width` + `bb_pct_b` drops test Spearman by 0.028–0.068
- Remaining signal: 0.67–0.76 Spearman — still strong from calendar + trend + derivatives
- Decision: keep `bb_width` (it's legitimate volatility clustering, not leakage)
- Strategy rule: if `bb_width` > 95th pct on its own, use it directly; else use model output

**Quantile regression (atr_15, alpha=0.90, test):** Coverage=0.880 (target=0.90) — well calibrated on test.

---

### Direction model — done (`models/direction.py`, `models/direction_dl.py`, `models/ensemble.py`)
Labels: `Y_up_H = max(price[t+1:t+H]) / price[t] - 1 > 0.8%`, symmetric for `Y_down_H`.
Horizons: 60 and 100 bars. Results cached at `cache/btc_direction_eval.parquet`, `cache/btc_ensemble_eval.parquet`.

**AUC comparison (test set):**

| Label | LightGBM | CNN-LSTM | **Ensemble** |
|---|---|---|---|
| down_60 | 0.698 | 0.732 | **0.736** |
| up_60 | 0.689 | 0.701 | **0.711** |
| down_100 | 0.675 | 0.683 | **0.702** |
| up_100 | 0.655 | 0.647 | **0.656** |

Walk-forward up_60: **6/6 folds > 0.52**, mean AUC=0.66. Gate to DL stage: PASS.

**Top direction features:** `vwap_1440`, `bb_width`, `fund_mom_480/1440`, `obv_1440`, `oi_z_1440`, `hour_sin/cos`, `taker_net_60`. Funding rate momentum and VWAP dominate.

**Confusion matrices (test, optimal F1 threshold):**

| Label | Model | Precision | Recall | F1 | Signal rate |
|---|---|---|---|---|---|
| down_60 | Ensemble | 0.258 | 0.389 | 0.310 | 19% |
| up_60 | Ensemble | 0.258 | 0.313 | 0.283 | 12% |
| down_100 | Ensemble | 0.308 | 0.592 | 0.405 | 37% |
| down_60 | CNN-LSTM | 0.236 | 0.515 | 0.324 | 28% |

**Key findings from confusion matrices:**
- LightGBM alone fires almost nothing at H=60 (0 predicted positives) — not tradeable alone
- CNN-LSTM fires too aggressively (80% signal rate for up_60) — too many false positives
- Ensemble is the only tradeable model — balances precision and recall
- Precision ≥ 0.70 threshold unachievable at any useful recall — model scores are compressed, calibration needed
- Positive rate grows from train→val→test (1.9% → 6.6% → 12.6%) — market became more directional, not leakage

**DeepLOB:** Optimised (20 OB levels, scaled matrix cached to `cache/btc_deeplob_ob20_scaled.npz`). Not yet trained — run: `python3 -m models.direction_dl btc deeplob`

---

### Pending tasks
- Probability calibration (Platt scaling / isotonic) — fix compressed probability scores so precision≥0.70 becomes achievable
- Cross-asset: run vol + direction models on ETH and SOL
- Two-stage pipeline: feed predicted ATR as feature into direction model
- DeepLOB training (infrastructure ready, run when needed)
- Backtest: plug ensemble signals into `backtest/engine.py` with OKX fees (taker 0.08%, maker 0.02%)

---

## Project structure

```
data/          loader.py, gaps.py
features/      orderbook.py, price.py, volume.py, market.py, assembly.py
models/        splits.py
               volatility.py       ← done, confusion matrix included
               direction.py        ← LightGBM done
               direction_dl.py     ← CNN-LSTM done, DeepLOB ready to run
               ensemble.py         ← LightGBM + CNN-LSTM weighted ensemble done
               evaluate.py         ← confusion matrix comparison across all models
strategy/      agent.py, genetic.py          ← empty stubs
backtest/      engine.py, costs.py           ← empty stubs
validation/    walkforward.py                ← empty stub
cache/         parquet and npy files (gitignored)
RESEARCH_PROMPT.md   full research agenda with feature/model details
```

## Code style

No docstring walls. Comments only for non-obvious decisions. No inline summaries of what was just done.
