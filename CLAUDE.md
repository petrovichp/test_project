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

### Volatility model — done (`models/volatility.py`)
Tested 3 target types × 5 horizons using LightGBM on BTC.

| Target | H=15 | H=30 | H=60 | H=100 | H=240 |
|---|---|---|---|---|---|
| ATR | **0.727** | **0.724** | 0.692 | **0.704** | 0.591 |
| Realized vol | 0.696 | 0.674 | 0.628 | 0.617 | 0.448 |
| Price range | 0.614 | 0.554 | 0.493 | 0.445 | 0.265 |

ATR at H=15–100 is the most predictable. Results cached at `cache/btc_volatility_research.parquet`.

**Next steps for vol model:**
- Walk-forward validation on best combinations (ATR H=15, ATR H=30, realized_vol H=15)
- Add OB features (span, imbalance, velocity) — currently excluded
- Quantile regression (predict 75th/90th percentile, more useful for TP/SL sizing)
- Test on ETH and SOL

### Direction model — not yet built
Needs `features/` modules first. See `RESEARCH_PROMPT.md` for full agenda.

## Project structure

```
data/          loader.py, gaps.py
features/      orderbook.py, price.py, volume.py, market.py, assembly.py  ← to build
models/        splits.py, volatility.py, direction.py  ← direction.py empty
strategy/      agent.py, genetic.py  ← empty stubs
backtest/      engine.py, costs.py   ← empty stubs
validation/    walkforward.py        ← empty stub
cache/         parquet and npy files (gitignored)
RESEARCH_PROMPT.md  full research agenda with feature/model details
```

## Code style

No docstring walls. Comments only for non-obvious decisions. No inline summaries of what was just done.
