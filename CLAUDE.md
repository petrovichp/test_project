# Crypto Trading ML — Project Context

## Project rules (auto-loaded)

@.claude/rules/data-integrity.md
@.claude/rules/code-style.md
@.claude/rules/experiments.md
@.claude/rules/model-registry.md
@.claude/rules/git.md
@.claude/rules/agent-workflow.md

> **For results, status, and next steps see [RESULTS.md](RESULTS.md), [docs/baselines.md](docs/baselines.md), [docs/voting_ensemble.md](docs/voting_ensemble.md), [docs/capacity_test.md](docs/capacity_test.md), [docs/trade_quality_by_agreement.md](docs/trade_quality_by_agreement.md), [docs/algo_test.md](docs/algo_test.md), [docs/baseline_vote5_audit.md](docs/baseline_vote5_audit.md), [docs/seed_variance.md](docs/seed_variance.md), [docs/ensemble_baseline.md](docs/ensemble_baseline.md), [docs/state_v6_test.md](docs/state_v6_test.md), [docs/experiments_log.md](docs/experiments_log.md), [docs/next_steps.md](docs/next_steps.md), [docs/data_splits.md](docs/data_splits.md), [docs/a2_rule_audit.md](docs/a2_rule_audit.md), [docs/audit_followup_tests.md](docs/audit_followup_tests.md), [docs/fee_sensitivity_vote5.md](docs/fee_sensitivity_vote5.md), [docs/fee_aware_retrain.md](docs/fee_aware_retrain.md), [docs/fee_improvement_proposals.md](docs/fee_improvement_proposals.md).**
> Latest findings (2026-05-08): three frozen baselines (see [docs/voting_ensemble.md](docs/voting_ensemble.md)). **`BASELINE_VOTE5`** (seeds 42/7/123/0/99 plurality): WF **+10.40**, fold-6 +5.20, test +4.19, val +3.53. **`BASELINE_VOTE5_DISJOINT`** (seeds 1/13/25/50/77 plurality, validated voting structurally): WF +10.06, fold-6 **+6.11**, test **+6.45**, val +3.79. **`BASELINE_FULL`** (single seed=42): WF +9.03, val **+7.30**. Plurality voting (discrete vote aggregation, tie → NO_TRADE) is structurally beneficial — Q-averaging produces "third action" drift especially on regime-disagreement bars and was abandoned. DQN exit-timing variants (B/C/B5/C2) do not beat rule-based exits in walk-forward (1/6 folds). Production path is Path X (maker-only execution) → live with rule-based exits.

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
  exit_dqn.py               Group B: 28-dim in-trade state, HOLD/EXIT_NOW + rule-based exits
  group_b_sweep.py          Group B 12-cell runner (3 global × fee + 9 per-strategy)
  exit_dqn_fixed.py         Group B5: 53-dim enriched state, fixed-N episodes, no rule exits
  group_b5_sweep.py         Group B5 27-cell runner (3 windows × 9 strategies)
  group_c_eval.py           Group C1: A4/A2 + B4 (variable-length) composition eval
  group_c2_eval.py          Group C2: A2 + B5 (fixed-window) composition eval
  group_c2_walkforward.py   6-fold walk-forward: A2+rule vs A2+B5 vs no-exit
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

## Project rules

See [.claude/rules/](.claude/rules/) — `data-integrity.md`, `code-style.md`, `experiments.md`, `git.md`, `agent-workflow.md`. These are auto-loaded into Claude's context via `@import` in the header above.
