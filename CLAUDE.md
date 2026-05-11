# Crypto Trading ML — Project Context

## Project rules (auto-loaded)

@.claude/rules/data-integrity.md
@.claude/rules/code-style.md
@.claude/rules/experiments.md
@.claude/rules/model-registry.md
@.claude/rules/git.md
@.claude/rules/agent-workflow.md

> **For the live development plan see [docs/development_plan.md](docs/development_plan.md). For results, status, and historical context see [RESULTS.md](RESULTS.md), [docs/baselines.md](docs/baselines.md), [docs/distill_vote5.md](docs/distill_vote5.md), [docs/cross_asset.md](docs/cross_asset.md), [docs/path_a_c1_results.md](docs/path_a_c1_results.md), [docs/z2_z3_results.md](docs/z2_z3_results.md), [docs/z1_results.md](docs/z1_results.md), [docs/voting_ensemble.md](docs/voting_ensemble.md), [docs/capacity_test.md](docs/capacity_test.md), [docs/trade_quality_by_agreement.md](docs/trade_quality_by_agreement.md), [docs/algo_test.md](docs/algo_test.md), [docs/baseline_vote5_audit.md](docs/baseline_vote5_audit.md), [docs/seed_variance.md](docs/seed_variance.md), [docs/ensemble_baseline.md](docs/ensemble_baseline.md), [docs/state_v6_test.md](docs/state_v6_test.md), [docs/experiments_log.md](docs/experiments_log.md), [docs/data_splits.md](docs/data_splits.md), [docs/a2_rule_audit.md](docs/a2_rule_audit.md), [docs/audit_followup_tests.md](docs/audit_followup_tests.md), [docs/fee_sensitivity_vote5.md](docs/fee_sensitivity_vote5.md), [docs/fee_aware_retrain.md](docs/fee_aware_retrain.md), [docs/fee_improvement_proposals.md](docs/fee_improvement_proposals.md).**
> Latest findings (2026-05-11, post Path C2 + Z2.1):
>
> **Primary baseline (max WF)**: **`VOTE5_v8_H256_DD`** — h=256 Double_Dueling, 12-action space (S11+S13 added via Z3 Step 4), 5 seeds plurality vote, v8_s11s13 state. **WF +12.07, val +6.67, test +4.44, 6/6 folds positive.** Disjoint-pool validation (Path A3) shows val is partly seed-luck; realistic expectations WF ~+10.5, val ~+2-3.
>
> **Cheap-deployment alternative**: **`DISTILL_v8_seed42`** — single DuelingDQN(52,12,256), 48k params, trained via masked CE on VOTE5_v8 plurality labels. **WF +9.99, val +10.41, test +9.35 (highest test in project history), 6/6 folds.** 5× cheaper inference than teacher. Disjoint-pool validation reproduces (s=50: test +9.86); family-mean test ~+7.3 is the honest expected. Voting distilled students *hurts* (same labels → correlated voters); deploy a single net, not a vote. See [docs/distill_vote5.md](docs/distill_vote5.md).
>
> **Cross-asset (Z2.1)**: identical stack on ETH (WF +7.22, val +5.57, test −0.09, 5/6 folds) and SOL (WF +8.24, val +4.16, test +2.19, 6/6 folds). Per-seed greedy val is negative on every cross-asset run — plurality vote is what salvages a deployable policy. Multi-asset deployment viable at ~50-60% of BTC per-asset Sharpe. See [docs/cross_asset.md](docs/cross_asset.md).
>
> **Prior context**: Phase Z1 winner `VOTE5_H256_DD` (test +9.01 was project record before C2 broke it). H256 capacity + DD regularization composes as hypothesized. `BASELINE_VOTE5_H128` exposed as seed-luck (dropped). K=10 plurality doesn't help (more ties → Sharpe ∝ √N collapse). Plurality voting is structurally beneficial; Q-averaging produces "third action" drift on regime-disagreement bars (abandoned). DQN exit-timing variants (B/C/B5/C2) don't beat rule-based exits in walk-forward. Path Z3 Step 5 (combining v8 + v7_basis state) did not compose (feature overlap). Path Z4.3 curriculum learning was NEGATIVE (regime gating biases buffer). Production path: Path X (maker-only execution) → live with rule-based exits using `DISTILL_v8_seed42` for cheap inference or `VOTE5_v8_H256_DD` for max WF.

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
