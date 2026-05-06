# Crypto Trading ML — Project Context

> **For results, status, and next steps see [RESULTS.md](RESULTS.md), [docs/experiments_log.md](docs/experiments_log.md), [docs/next_steps.md](docs/next_steps.md).**
> Latest finding: DQN entry-gating works at OKX maker fees (val Sharpe +1.72); fails at taker fees. Production path is maker-only execution.

## What this project is

Research-first ML system for crypto trading on OKX (BTC/ETH/SOL, 1-minute bars). Goal: validate signal with honest out-of-sample methodology before any production deployment. Live exchange target: OKX (spot + perp).

## Data

**Never touch raw CSVs.** Use preprocessed parquets in `cache/`:

| File | Rows | Description |
|---|---|---|
| `cache/okx_btcusdt_spotpepr_20260425_meta.parquet` | 384k | 29 market/microstructure columns |
| `cache/okx_btcusdt_spotpepr_20260425_ob.parquet` | 384k | 800 OB amount columns + timestamp |
| `cache/okx_ethusdt_spotpepr_20260425_meta.parquet` | — | ETH (same schema) |
| `cache/okx_solusdt_spotpepr_20260425_meta.parquet` | — | SOL (same schema) |

Meta columns: `timestamp, oi_usd, fund_rate, spot_ask_price, spot_bid_price, perp_ask_price, perp_bid_price, span_spot_price, span_perp_price, spot_minute_volume, perp_minute_volume, spot_sell_buy_side_deals, perp_sell_buy_side_deals, spot_spread_bps, spot_imbalance, spot_bid_concentration, spot_ask_concentration, spot_large_bid_count, spot_large_ask_count, perp_spread_bps, perp_imbalance, perp_bid_concentration, perp_ask_concentration, perp_large_bid_count, perp_large_ask_count, taker_sell_buy_ratio, taker_sell, taker_buy, diff_price`

OB encoding: 200 equal-width price bins per side anchored at best bid/ask (bin 0 = closest to mid). Amounts normalized by `spot_ask_price`. `span_spot_price` / `span_perp_price` = dollar range covered.

Loader: `data/loader.py` — `load_meta(ticker)` or `load(ticker, include_ob=True)`.

## Project structure

```
data/        loader.py, gaps.py
features/    orderbook.py, price.py, volume.py, market.py, assembly.py
                                                   → 191 features, cached parquet

models/
  splits.py                 sequential + walk_forward split helpers
  vol_v4.py                 LightGBM ATR-30  (Spearman 0.69 OOS)
  direction_dl_v4.py        4 CNN-LSTMs       (AUC 0.64–0.70 OOS)
  regime_cusum_v4.py        CUSUM+Hurst regime classifier
  dqn_state.py              50-dim state arrays (cached npz)
  dqn_network.py            DQN MLP (50→64→32→10, 5,674 params)
  dqn_replay.py             PER buffer + stratified sampling
  dqn_rollout.py            env-loop driver  (--fee --trade-penalty)
  dqn_selector.py           DQN training loop
  group_a_sweep.py          fee × penalty sweep runner
  walk_forward.py           6-fold validation
  grid_search.py            hyperparameter search
  pnl_predictor.py          supervised PnL regression diagnostic
  diagnostics_ab.py         fee-free walk-forward + optimal-exit oracle
  diagnostics_c.py          5-min timeframe diagnostic
  plot_results.py           single-strategy equity plot
  plot_dqn_results.py       Group A DQN policies plot

strategy/    agent.py             9 strategies + DEFAULT_PARAMS
execution/   entry / exit / sizing / config (per-strategy execution wiring)
backtest/
  engine.py                 bar-by-bar simulator
  costs.py                  OKX fee model
  single_trade.py           numba-jit single-trade simulator (0.7 µs/call, parity-verified)
  preds.py                  cached vol + direction predictions
  run.py                    strategy backtest runner

cache/       parquet, npz, .keras, .pt, .png  (gitignored)
docs/        experiments_log.md, next_steps.md
RESULTS.md   top-level summary
```

## Non-negotiable rules

- **No leakage**: features use only bars `[t − lookback, t]`. `shift(-n)` for labels only.
- **No random splits**: time-series data is never shuffled. Sequential or walk-forward only.
- **Normalization**: fit scaler on train split only, transform val/test.
- **Cross-asset features**: lag predictor asset by 1+ bars.
- **Rolling windows**: `min_periods=full_window` — NaN early bars, exclude from samples.
- **Embargo**: leave `label_length` bar gap between train end and val/test start.
- **Test split is locked**: touch only after val tuning is frozen.
- **Cache rule**: any computation > 10s saves to `cache/*.npz` or `*.parquet`. Always check before recomputing.

## Code style

No docstring walls. Comments only for non-obvious decisions. No inline summaries of what was just done.
