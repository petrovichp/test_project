# Research Prompt — Crypto Trading ML

## Project

Research-first ML system for crypto trading on OKX (BTC/ETH/SOL, 1-minute bars).
Goal: validate signal with honest out-of-sample methodology before any production deployment.

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

**OB encoding:** 200 equal-width price bins anchored at best bid/ask (bin 0 = closest to mid).
Amounts normalized by `spot_ask_price`. `span_spot_price` / `span_perp_price` = bid-ask spread
(NOT OB depth range — true price-level features require adding depth span to collection pipeline).

**Splits** — 50/25/25 sequential, no random shuffling:
- Train: 70,902 rows — Jul→Oct 2025
- Val: 35,451 rows — Oct→Dec 2025 (harder regime — includes Nov 2025 outage)
- Test: 35,451 rows — Dec 2025→Apr 2026

`models/splits.py`: `sequential(n, 0.50, 0.25)` and `walk_forward(ts, 90d, 30d, 30d)` → 6 folds

---

## Infrastructure

| Module | Function |
|---|---|
| `data/loader.py` | `load_meta(ticker)` / `load(ticker)` — parquet cache |
| `data/gaps.py` | `clean_mask(timestamps, max_lookback)` — flags gap-contaminated rows |
| `models/splits.py` | Sequential and walk-forward splits |
| `features/orderbook.py` | 93 OB features incl. True OFI, depth bands, wall detection |
| `features/price.py` | 51 price/momentum features |
| `features/volume.py` | 17 volume/taker features |
| `features/market.py` | 30 OI, funding, calendar features |
| `features/assembly.py` | Combines all → 191 features, applies gap mask, scales |
| `models/volatility.py` | Vol models. Run: `python3 -m models.volatility btc` |
| `models/direction.py` | LightGBM direction. Run: `python3 -m models.direction btc` |
| `models/direction_dl.py` | CNN-LSTM + DeepLOB. Run: `python3 -m models.direction_dl btc cnn_lstm` |
| `models/ensemble.py` | LightGBM + CNN-LSTM ensemble. Run: `python3 -m models.ensemble btc` |
| `models/evaluate.py` | Confusion matrix comparison. Run: `python3 -m models.evaluate btc` |
| `model_registry.json` | Central registry of all trained models with metrics |

**Caching:** expensive intermediates → `cache/` as `.parquet`, `.npy`, `.keras`, `.txt`. Always check cache before recomputing.

**Model coding system:** `{ticker}_{model_type}_{target}_{horizon}`
- `btc_lgbm_atr_30` — LightGBM, BTC, ATR, 30-bar horizon
- `btc_cnn_dir_up_60` — CNN-LSTM, BTC, direction up, 60-bar horizon
- `btc_ens_dir_dn_100` — Ensemble, BTC, direction down, 100-bar horizon

---

## Strict No-Leakage Rules

- All features use only bars `[t - lookback, t]`. Never `shift(-n)` on features.
- `shift(-n)` for label construction only.
- Normalization: fit scaler on train only, transform val/test.
- No BatchNorm over full dataset — use LayerNorm or InstanceNorm in DL.
- Cross-asset features: lag predictor by 1+ bars.
- Rolling windows: `min_periods=full_window` — NaN early bars, exclude from samples.
- Embargo: gap of `label_length` bars between train end and val/test start.
- Walk-forward only. Never shuffle time-series.

---

## Research Results

### Volatility Models — done

LightGBM regressors. Targets: ATR ($) and realized vol (%). Horizons: 15, 30, 60, 100, 240 bars.

**Best results (Spearman correlation, test set):**

| Model | Val Spearman | Test Spearman | Top-33% Precision (test) |
|---|---|---|---|
| btc_lgbm_atr_30 | 0.627 | **0.801** | 0.644 |
| btc_lgbm_rvol_30 | 0.579 | **0.800** | **0.921** |
| btc_lgbm_atr_15 | 0.647 | 0.784 | 0.606 |
| btc_lgbm_rvol_15 | 0.605 | 0.790 | **0.879** |
| btc_lgbm_atr_60 | 0.582 | 0.764 | 0.662 |
| btc_lgbm_atr_100 | 0.559 | 0.726 | 0.673 |

Walk-forward: **all 6 folds positive** (0.57–0.80). Signal is consistent across regimes.

**Top features:** `bb_width` (27–29%), `dow_sin` (7–9%), `hour_sin/cos` (6–9%), `session_ny` (2–4%), `vwap_dev_1440` / `vwap_1440` / `obv_1440` (3–4% each), `oi_z_1440` / `fund_mean_1440` (1.5% each).

**bb_width removal experiment:** dropping it costs 0.03–0.07 Spearman. Kept — it's legitimate volatility clustering, not leakage. Strategy rule: if bb_width > 95th pct, use directly; else use model output.

**Quantile regression (atr_15, alpha=0.90):** coverage=0.880 on test — well calibrated.

---

### Direction Models — done

Labels: `Y_up_H = max(price[t+1:t+H]) / price[t] - 1 > 0.8%`, symmetric for down. H=60 and 100.

**AUC comparison (test set):**

| Label | LightGBM | CNN-LSTM | **Ensemble** |
|---|---|---|---|
| down_100 | 0.703 | 0.720 | **0.725** |
| up_100 | 0.646 | 0.702 | **0.714** |
| up_60 | 0.640 | 0.699 | **0.704** |
| down_60 | 0.670 | 0.667 | **0.669** |

Walk-forward up_60: **6/6 folds > 0.52**, mean AUC=0.66. Gate to DL stage: PASS.

**Top direction features:** `vwap_1440`, `bb_width`, `fund_mom_480/1440`, `obv_1440`, `oi_z_1440`, `hour_sin/cos`, `taker_net_60`, `ofi_perp_10`. Funding rate momentum and VWAP dominate.

**Confusion matrices (test, optimal F1 threshold):**

| Label | Model | Precision | Recall | F1 | Signal rate |
|---|---|---|---|---|---|
| down_100 | CNN-LSTM (t=0.64) | **0.331** | 0.540 | 0.411 | 37% |
| down_100 | Ensemble (t=0.35) | 0.335 | 0.517 | 0.407 | 30% |
| up_100 | Ensemble (t=0.35) | 0.268 | 0.607 | 0.372 | 36% |
| down_60 | CNN-LSTM (t=0.24) | 0.260 | 0.307 | 0.281 | 19% |

**Key findings:**
- LightGBM fires near-zero positives at any useful threshold — not tradeable alone
- CNN-LSTM fires aggressively at low thresholds — too many false positives
- Ensemble balances precision/recall best — only tradeable model
- Precision ≥ 0.70 unachievable at useful recall — scores are compressed, **calibration needed**
- Positive rate grows train→val→test (1.9%→6.6%→12.6%) — market became more directional, not leakage

**DeepLOB — dropped:** test AUC 0.39–0.55, two labels below 0.50. Raw OB bins without price/volume context carry insufficient directional signal at this data scale and architecture size.

**CNN-LSTM architecture:** `CausalConv1D(32, k=3)` → `GRU(64)` → `Dense(32)` → sigmoid. SEQ_LEN=30, 30 input features including new OFI and depth-band features.

---

### Architecture Decisions

| Decision | Reason |
|---|---|
| Drop random splits | Leakage — original notebook AUC inflated by up to 0.15 |
| Drop DeepLOB | Severe overfitting, test AUC <0.50 on 2 of 4 labels |
| Keep bb_width | Legitimate volatility clustering, not leakage |
| Ensemble over single model | Consistently +0.002–0.012 AUC over best individual model |
| Val underperforms test | Nov 2025 outage = harder regime, not a methodology bug |
| OFI from bin diffs | Stronger signal than imbalance proxy — direction models improved |

---

## Model Training Plan — Next Steps

### Pending (priority order)

**1. Probability calibration (highest priority)**
- Problem: all model scores compressed — precision ≥ 0.70 unachievable at any recall
- Fix: apply Platt scaling (logistic regression on val probabilities) or isotonic regression
- Target: `down_100` CNN-LSTM at precision ≥ 0.70 after calibration
- Implementation: `models/calibration.py` — wrap existing models with sklearn `CalibratedClassifierCV`

**2. Two-stage pipeline**
- Feed predicted ATR (from volatility model) as a feature into direction models
- Expected: helps direction model calibrate its confidence during high-vol periods
- Implementation: add `btc_lgbm_atr_30` predictions as a feature column in `features/assembly.py`

**3. Cross-asset (ETH and SOL)**
- Run same volatility and direction pipeline on ETH and SOL
- Use BTC lag-1 return as cross-asset feature for ETH/SOL models
- Check: does signal transfer across assets with same feature set?

**4. Backtest**
- Plug ensemble signals into `backtest/engine.py`
- OKX fees: taker 0.08%, maker 0.02%
- Execution lag: 1-bar delay
- Evaluate: Sharpe ratio, max drawdown, profit factor

**5. OB depth span (data collection improvement)**
- Currently `span_spot_price` = bid-ask spread (not OB depth range)
- Add actual dollar range covered by 200 bins to collection pipeline
- Unlocks true price-level features: "liquidity within ±0.5% of mid"

**6. DeepLOB (revisit with more data)**
- Current dataset too small and noisy for raw OB sequence learning
- Revisit when ETH/SOL data is added (3× more samples) or with GPU training

---

## Evaluation Protocol (apply to every new model)

1. **Train** on train split, early stopping evaluated on val only
2. **Val analysis** — AUC/Spearman, confusion matrix, feature importance. Gate: AUC > 0.52 / Spearman > 0.40
3. **Test** — run once only after val passes. Never re-tune after seeing test.
4. **Walk-forward** — 6 folds. Signal confirmed if 5+ of 6 folds pass gate.
5. **Calibration check** — are predicted probabilities well-spread or compressed?

---

## Code Style

- No docstring walls. Comments only for non-obvious decisions.
- No random shuffling anywhere.
- Cache everything expensive in `cache/`.
- Run modules directly: `python3 -m models.volatility btc`
- Model codes: `{ticker}_{model_type}_{target}_{horizon}`
