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

**Two-stage pipeline:** `btc_lgbm_atr_30` ATR rank prediction permanently included as feature.
Improved AUC by +0.039–0.050 on up_60 and down_60.

**AUC comparison (test set, two-stage ensemble):**

| Label | LightGBM | CNN-LSTM | **Ensemble** | vs no ATR |
|---|---|---|---|---|
| up_60 | 0.644 | 0.753 | **0.754** | +0.043 |
| down_60 | 0.681 | 0.707 | **0.708** | +0.039 |
| up_100 | 0.690 | 0.715 | **0.719** | +0.005 |
| down_100 | 0.701 | 0.730 | **0.733** | +0.008 |

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

### Completed experiments

**Probability calibration** (`models/calibration.py`) — **no improvement**
- Isotonic regression on val predictions: AUC unchanged, precision negligibly affected
- Root cause: calibration remaps output scale but cannot improve the model's ranking ability
- Conclusion: need more signal, not calibration

**Two-stage pipeline** (`models/two_stage.py`, now in `models/ensemble.py`) — **genuine improvement**
- ATR rank feature: +0.039–0.050 AUC on up_60 and down_60
- Now permanently included in ensemble as default
- CNN models saved as `btc_cnn2s_dir_*`, ensemble as `btc_ens2s_dir_*`

### Pending (priority order)

**1. Cross-asset (ETH and SOL)**
- Run same volatility and direction pipeline on ETH and SOL
- Use BTC lag-1 return as cross-asset feature for ETH/SOL models
- Check: does signal transfer across assets with same feature set?

**2. Backtest engine — implement and run all strategies**
- Build `backtest/engine.py`: simulate bar-by-bar execution, OKX fees (taker 0.08%, maker 0.02%), 1-bar lag
- Run Strategies 1–6 individually on test set, evaluate Sharpe / max drawdown / profit factor
- RL agent (Strategy 7) requires backtest engine as simulation environment

**3. OB depth span (data collection improvement)**
- Currently `span_spot_price` = bid-ask spread (not OB depth range)
- Add actual dollar range covered by 200 bins to collection pipeline
- Unlocks true price-level features: "liquidity within ±0.5% of mid"

**4. DeepLOB (revisit with more data)**
- Current dataset too small and noisy for raw OB sequence learning
- Revisit when ETH/SOL data is added (3× more samples) or with GPU training

---

## Trading Strategy Plan

All signals are already computed in the feature pipeline (`features/assembly.py`).
Technical indicators available: `bb_pct_b`, `bb_width`, `macd_hist`, `macd`, `macd_signal`,
`rsi_6`, `rsi_14`, `sma_20/50/200`, `ema_12/26`, `ret_sma_200`, `vwap_1440`, `vwap_dev_1440`.
ML signals: `vol_pred` (ATR rank), `p_up_60`, `p_dn_60`, `p_up_100`, `p_dn_100`.

### Strategy 1 — Volatility-Filtered Direction
Use vol model as regime gate, direction ensemble for side, technical indicators to avoid overextended entries.
```
Entry long : vol_pred > 0.65 AND p_up_60 > 0.50 AND bb_pct_b < 0.70 AND rsi_14 < 65
Entry short: vol_pred > 0.65 AND p_dn_60 > 0.50 AND bb_pct_b > 0.30 AND rsi_14 > 35
TP = entry ± 1.5 × atr_30   |   SL = entry ∓ 0.8 × atr_30   |   Time stop: 60 bars
Position size ∝ vol_pred
```
Best for: volatile trending sessions (NY hours 13–21 UTC).

### Strategy 2 — Funding Rate Mean-Reversion
Extreme funding creates carry pressure that reverses. MACD confirms weakening momentum.
```
Entry short: fund_rate > 95th pct AND fund_mom_480 > 0 AND macd_hist < 0 AND rsi_14 > 60
Entry long : fund_rate < 5th pct  AND fund_mom_480 < 0 AND macd_hist > 0 AND rsi_14 < 40
TP: funding returns to ±1σ   |   SL: 2% adverse move   |   Max hold: 240 bars
```
Best for: low-frequency (~3–5 trades/day), late Asia session when funding imbalances build.

### Strategy 3 — Bollinger Band Mean-Reversion
Price at BB extreme + OFI normalising + low vol regime = mean-reversion setup.
```
Entry long : bb_pct_b < 0.05 AND ofi_perp_10_r15 > 0 AND taker_imb_5 > -0.2
             AND vol_pred < 0.50 AND vwap_dev_1440 < -0.005
Entry short: bb_pct_b > 0.95 AND ofi_perp_10_r15 < 0 AND taker_imb_5 < 0.2
             AND vol_pred < 0.50 AND vwap_dev_1440 > 0.005
TP: bb_pct_b returns to 0.50 (midline)   |   SL: bb_pct_b hits 0.00 or 1.00   |   Time stop: 30 bars
```
Key: only fires in tight-band regimes (low `bb_width`). Wide band = trend continuation, skip.

### Strategy 4 — MACD + SMA Trend Following
SMA stack confirms bullish/bearish structure, MACD histogram expanding, vol supports trend.
```
Entry long : macd_hist > 0 AND macd_hist expanding AND price > sma_50 > sma_200
             AND ret_sma_200 > 0.002 AND vol_pred > 0.60 AND p_up_60 > 0.45
Entry short: macd_hist < 0 AND macd_hist expanding AND price < sma_50 < sma_200
             AND ret_sma_200 < -0.002 AND vol_pred > 0.60 AND p_dn_60 > 0.45
TP = 2 × atr_30   |   SL: price crosses sma_50 OR 1 × atr_30   |   Trailing stop at breakeven
```
Best for: strong trending sessions with high OI buildup (`oi_z_1440` elevated).

### Strategy 5 — OFI Momentum Scalp
True OFI spike signals imminent price move. Short hold, tight exits.
```
Entry long : ofi_perp_10_r15 > +2σ AND ofi_perp_10 > 0 AND taker_net_15 > 0 AND rsi_6 < 70
Entry short: ofi_perp_10_r15 < -2σ AND ofi_perp_10 < 0 AND taker_net_15 < 0 AND rsi_6 > 30
TP = 0.8 × atr_30   |   SL = 0.4 × atr_30   |   Hard time stop: 15 bars
Exit immediately if OFI reverses sign
```
Warning: 0.16% round-trip fees — viable only with maker orders or avg move > 0.5%.

### Strategy 6 — Two-Signal High-Precision
Multiple independent signals must agree. Low frequency (~3–5% of bars), highest precision.
```
Entry long : p_up_60 > 0.55 AND p_dn_60 < 0.30 AND macd_hist > 0
             AND 45 < rsi_14 < 65 AND ofi_perp_10_r15 > 0 AND vol_pred > 0.55
Entry short: p_dn_60 > 0.55 AND p_up_60 < 0.30 AND macd_hist < 0
             AND 35 < rsi_14 < 55 AND ofi_perp_10_r15 < 0 AND vol_pred > 0.55
TP = 2 × atr_30   |   SL = 1 × atr_30   |   Time stop: 60 bars
```
Expected precision ~0.45–0.55 based on confusion matrix analysis (10% base rate → 4–5× better than random).

### Strategy 7 — RL Meta-Agent (Strategy Selector)
An RL agent observes the full market state and selects which strategy (or flat) is most appropriate for the current regime. Acts as an adaptive meta-controller that switches between Strategies 1–6.

**State space (196-dim):**
- 191 assembled features (technical + microstructure + derivatives)
- ML outputs: `vol_pred`, `p_up_60`, `p_dn_60`, `p_up_100`, `p_dn_100`

**Action space (8 discrete):**
```
0 = Flat   1 = S1 Long   2 = S1 Short   3 = S2 (Funding)
4 = S3 (BB reversion)    5 = S4 (MACD trend)
6 = S5 (OFI scalp)       7 = S6 (Two-signal)
```

**Reward:**
```
r_t = pnl_t - 0.0008 × |trade_t| - 0.2 × drawdown_t - 0.1 × overtrading_penalty_t
```

**Architecture:** LSTM(128) over 30-bar history → Dense(64, 32) → Policy head (Softmax 8) + Value head (Dense 1)
**Algorithm:** PPO (Proximal Policy Optimization) — handles non-stationarity of market data
**Training:** Walk-forward, 90-day train / 30-day eval windows, simulated execution with fees

**What the agent learns:**
- High vol + strong trend + high OI → select Strategy 4 (MACD trend)
- High vol + direction signal → select Strategy 1 (vol-filtered)
- Extreme funding + weakening price → select Strategy 2 (funding reversion)
- Low vol + price at BB extreme → select Strategy 3 (BB mean-reversion)
- Multiple signals aligned → select Strategy 6 (two-signal)
- Ambiguous state → Flat (preserve capital)

### Implementation Roadmap

| Step | Task | Dependency |
|---|---|---|
| 1 | Build `backtest/engine.py` with OKX fees + 1-bar lag | — |
| 2 | Backtest Strategies 1–4 individually on test set | Step 1 |
| 3 | Identify per-regime best strategy from backtest | Step 2 |
| 4 | Build RL environment wrapping backtest engine | Step 1 |
| 5 | Train PPO agent on walk-forward splits | Steps 3, 4 |
| 6 | Evaluate RL agent vs best individual strategy | Step 5 |

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
