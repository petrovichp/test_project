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

### Volatility model — done (`models/volatility.py`)
Uses assembled features. Results cached at `cache/btc_volatility_eval.parquet`.

| Target | Val Spearman | Test Spearman |
|---|---|---|
| ATR H=15 | 0.647 | **0.784** |
| ATR H=30 | 0.627 | **0.801** |
| Realized vol H=15 | 0.605 | 0.790 |

Walk-forward atr_15: all 6 folds positive (0.57–0.80). Top features: `bb_width`, `dow_sin`, `hour_sin`, `vwap_dev_1440`, `oi_z_1440`.
Val underperforms test — val period (Oct–Dec 2025) covers the Nov outage, a harder regime.
Quantile regression (75/90th pct) runs but coverage undershoots on val, corrects on test.

### Direction model — LightGBM done (`models/direction.py`)
Results cached at `cache/btc_direction_eval.parquet`.

| Label | Val AUC | Test AUC | Gap |
|---|---|---|---|
| down_60 | 0.684 | **0.698** | ✓ 0.015 |
| up_60 | 0.599 | 0.689 | ⚠ 0.091 |
| down_100 | 0.592 | 0.675 | ⚠ 0.083 |
| up_100 | 0.578 | 0.655 | ⚠ 0.077 |

Walk-forward up_60: **6/6 folds > 0.52**, mean AUC 0.66. **Gate to DL stage: PASS.**
Top features: `vwap_1440`, `bb_width`, `fund_mom_480/1440`, `obv_1440`, `oi_z_1440`, `hour_sin/cos`, `taker_net_60`.
down_60 is the most stable model (smallest val/test gap).

### DL models — next
CNN-LSTM hybrid → DeepLOB → Ensemble.

## Project structure

```
data/          loader.py, gaps.py
features/      orderbook.py, price.py, volume.py, market.py, assembly.py
models/        splits.py, volatility.py, direction.py  ← LightGBM done
               direction_dl.py  ← to build (CNN-LSTM, DeepLOB)
strategy/      agent.py, genetic.py  ← empty stubs
backtest/      engine.py, costs.py   ← empty stubs
validation/    walkforward.py        ← empty stub
cache/         parquet and npy files (gitignored)
RESEARCH_PROMPT.md  full research agenda
```

## Code style

No docstring walls. Comments only for non-obvious decisions. No inline summaries of what was just done.
