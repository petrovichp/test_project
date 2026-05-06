# Crypto Trading ML — Project Context

> **For complete experimental results and project status, see [RESULTS.md](RESULTS.md), [docs/experiments_log.md](docs/experiments_log.md), and [docs/next_steps.md](docs/next_steps.md).**
> Latest finding: DQN entry-gating works at maker fees (val Sharpe +1.72 → equity 1.07× over val); fails at taker fees. Production path is Path X (maker-only execution). See RESULTS.md for the full picture.

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
| `models/volatility.py` | Volatility model. Run: `python3 -m models.volatility btc` |
| `models/direction.py` | LightGBM direction model. Run: `python3 -m models.direction btc` |
| `models/direction_dl.py` | CNN-LSTM. Run: `python3 -m models.direction_dl btc cnn_lstm` |
| `models/ensemble.py` | LightGBM + CNN-LSTM weighted ensemble. Run: `python3 -m models.ensemble btc` |
| `backtest/run.py` | Strategy backtest runner. Run: `python3 -m backtest.run btc` |
| `backtest/preds.py` | Cached vol + direction predictions (npz cache, mtime-invalidated) |

Caching rule: save expensive intermediates to `cache/` as `.parquet` or `.npz`. Always check cache before recomputing.

**Prediction cache:** `_vol_preds` and `_dir_preds` in `backtest/preds.py` cache to `.npz`. Invalidated automatically when model file mtime changes. Reduces backtest from ~80s → 9.5s.

## Non-negotiable rules

- **No leakage**: all features use only bars `[t - lookback, t]`. `shift(-n)` for labels only.
- **No random splits**: time-series data is never shuffled. Sequential or walk-forward only.
- **Normalization**: fit scaler on train split only, transform val/test.
- **Cross-asset features**: lag predictor asset by 1+ bars.
- **Rolling windows**: `min_periods=full_window` — NaN early bars, exclude from samples.
- **Embargo**: leave `label_length` bar gap between train end and val/test start.

## Current research state

### Features pipeline — done
191 features across 4 modules, zero NaN after gap masking (MAX_LOOKBACK=1440).
Cached to `cache/btc_features_assembled.parquet`. Load via `features/assembly.py`.

| Module | Features | Key contents |
|---|---|---|
| `features/orderbook.py` | 32 | bucket amounts, imbalance, OFI, velocity, span |
| `features/price.py` | 51 | returns, SMA/EMA, RSI, MACD, VWAP, basis |
| `features/volume.py` | 17 | taker imbalance/net, vol z-score, OBV, OFI |
| `features/market.py` | 30 | OI, funding rate, spread, calendar, sessions |

Splits after gap masking: Train 70,902 (Jul→Oct 2025) / Val 35,451 (Oct→Dec 2025) / Test 35,451 (Dec 2025→Apr 2026).

---

### Volatility model — done (`models/volatility.py`)
Target: `atr_30` ($range/bar over next 30 bars). Model: LightGBM. Cached: `cache/btc_lgbm_atr_30.txt`.

| Target | Val Spearman | Test Spearman |
|---|---|---|
| atr_30 | 0.627 | **0.801** |
| atr_15 | 0.647 | 0.784 |

Output used as: (1) vol gate `vol_pred` (percentile rank), (2) ATR-dynamic TP/SL scaling, (3) `atr_rank` feature for CNN-LSTM two-stage pipeline.

---

### Direction model — done (`models/direction_dl.py`, `models/ensemble.py`)
Labels: `Y_up_H = max(price[t+1:t+H]) / price[t] - 1 > 0.8%`. Horizons: 60 and 100 bars.
Pipeline: ATR-30 rank appended to features (two-stage) → CNN-LSTM SEQ_LEN=30.

| Label | CNN-LSTM AUC (test) |
|---|---|
| up_60 | **0.753** |
| down_60 | 0.707 |
| up_100 | 0.715 |
| down_100 | 0.730 |

Cached: `cache/btc_cnn2s_dir_{up/down}_{60/100}.keras`. Predictions cached: `cache/btc_pred_dir_{col}.npz`.

**Direction model usage rules:**
- Use only with `vol_pred > 0.60` (vol gate active)
- Threshold ≥ 0.75 for up_60, ≥ 0.70 for down_60
- Not for mean-reversion strategies (S2, S3)

---

### Backtest — done (`backtest/run.py`)
Bar-by-bar engine with 1-bar lag, OKX fees 0.08%/side, ATR-dynamic TP/SL, breakeven stop, trail-after-breakeven, time stop. **No regime gate** — all strategies fire freely.

Run: `python3 -m backtest.run btc` (~9.5s with cache)

**Latest results — no regime layer (2026-05-05):**

| Strategy | Val Sharpe | Val Win% | Val Trades | Test Sharpe | Test Win% | Test Trades |
|---|---|---|---|---|---|---|
| S1_VolDir | **+7.02 ✓** | 46% | 98 | -0.81 | 46% | 140 |
| S8_TakerFlow | **+2.42 ✓** | 48% | 99 | -6.12 | 34% | 147 |
| S4_MACDTrend | +0.19 | 39% | 56 | **+0.70 ✓** | 48% | 93 |
| S5/S7/S9/S11/S13 | structural failures (Sharpe < -20 on both splits) |

**Val/test gap remains the core unsolved problem.** S1 hits Sharpe +7.02 on val (Oct–Dec 2025, Nov outage) but -0.81 on test (Dec 2025–Apr 2026, regime mix). Without a regime mechanism, we don't know when to trust the signal.

---

### Phase 2 — regime classifier (TODO, fresh start)

Phase 2 will add a regime-awareness layer to gate strategies. Approach: TBD.
Previous Phase 2 attempt (HMM + extended models) was scrapped — see PLAN.md for fresh design.

### Pending tasks
- **Design Phase 2 regime layer from scratch** — what features, what model, what gate logic
- **Per-strategy param grid search** — optimize TP/SL/thresholds for S1, S8 on val
- **Drop structural failures** — S5/S9/S11/S13 (Sharpe < -20 on both splits)
- **Cross-asset**: run vol + direction models on ETH and SOL

---

## Project structure

```
data/          loader.py, gaps.py
features/      orderbook.py, price.py, volume.py, market.py, assembly.py
models/        splits.py
               volatility.py        ← done (LightGBM ATR)
               direction.py         ← done (LightGBM)
               direction_dl.py      ← done (CNN-LSTM two-stage)
               ensemble.py          ← done (LightGBM + CNN-LSTM)
               evaluate.py          ← confusion matrix comparison
strategy/      agent.py             ← S1–S13 signal functions + DEFAULT_PARAMS
               genetic.py           ← stub (param grid search)
execution/     entry.py             ← MarketEntry, ConfirmEntry, SpreadEntry
               exit.py              ← FixedExit, ATRDynamicExit, ComboExit
               sizing.py            ← FixedFraction, VolScaledSizer
               config.py            ← EXECUTION_CONFIG per strategy
backtest/      engine.py            ← bar-by-bar simulator (all exit mechanisms)
               costs.py             ← OKX fee model
               run.py               ← strategy backtest runner (no regime gate)
               preds.py             ← cached vol + direction predictions
cache/         parquet and npz files (gitignored)
PLAN.md              phased development plan
ARCHITECTURE.md      full system diagram with results
```

## Code style

No docstring walls. Comments only for non-obvious decisions. No inline summaries of what was just done.
