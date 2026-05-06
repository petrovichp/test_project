# Detailed Experiments Log

Companion to [RESULTS.md](../RESULTS.md). Per-experiment numbers, configurations, and per-fold breakdowns.

---

## Phase 1+2

### Vol model v4 — LightGBM ATR-30

- **Module:** [models/vol_v4.py](../models/vol_v4.py)
- **Train:** bars [1,440, 101,440), 100,000 rows, full 191-feature set
- **Holdout:** last 5% of vol-train (early-stopping)
- **Output:** `cache/btc_lgbm_atr_30_v4.txt`, `cache/btc_pred_vol_v4.npz`

| Metric | Value |
|---|---|
| Spearman in-sample | +0.877 |
| **Spearman OOS (RL period 283k bars)** | **+0.690** |
| Best iteration | 103 |
| ATR train median | 30.49 |

### Direction CNN-LSTM v4 — 4 models (up/down × 60/100)

- **Module:** [models/direction_dl_v4.py](../models/direction_dl_v4.py)
- **Architecture:** Conv1D(32, k=3, causal) → GRU(64) → Dropout(0.3) → Dense(32) → Dense(1, sigmoid)
- **SEQ_LEN:** 30 + vol-rank-v4 as 31st channel (two-stage)
- **Train:** dir-train [1,440, 91,440), 90,000 rows
- **Holdout:** dir-holdout [91,440, 101,440), 10,000 rows (early-stopping `val_auc` patience=5)

| Label | AUC train | AUC holdout | AUC RL OOS |
|---|---|---|---|
| up_60 | 0.9015 | 0.7319 | **0.7032** |
| down_60 | 0.9710 | 0.8186 | **0.6799** |
| up_100 | 0.8540 | 0.7361 | **0.6359** |
| down_100 | 0.9000 | 0.5918 | **0.6418** |

vs v3 baseline: −0.05 to −0.09 AUC. Modest degradation from longer OOS span (4× v3) and across-gap windows.

### CUSUM regime v4

- **Module:** [models/regime_cusum_v4.py](../models/regime_cusum_v4.py)
- **Threshold fit:** Vol-train (100k bars)
- **Labels:** all bars [1,440, 384,614)

| Threshold | Value |
|---|---|
| CUSUM+ p75 | +12.825 |
| CUSUM− p25 | −12.634 |
| Hurst p65 / p35 | +0.521 / +0.469 |
| bb_width p30 | 0.00150 |

**State distribution per split:**

| Split | calm | trend_up | trend_down | ranging | chop |
|---|---|---|---|---|---|
| vol-train | 30.0% | 9.4% | 7.7% | 23.8% | 29.1% |
| DQN-train | 21.1% | 7.7% | 7.5% | 31.2% | 32.5% |
| DQN-val | 8.2% | 9.0% | 11.1% | 32.5% | 39.1% |
| DQN-test | 17.4% | 6.8% | 8.5% | 33.0% | 34.2% |

**KW gate (DQN-train fwd-30):** PASS, p = 2.21e-20.

### State arrays + standardization

- **Module:** [models/dqn_state.py](../models/dqn_state.py)
- **Output:**
  - `cache/btc_dqn_standardize_v5.json` (median + IQR per feature, fit on Vol-train)
  - `cache/btc_dqn_state_{train,val,test}.npz`

**State layout (50 dims):**

```
0      vol_pred                       standardized
1      atr_pred_norm                  standardized
2-6    regime one-hot (5 states)
7-15   signal flags (9 strategies, ∈ {-1, 0, +1})
16     bb_width                       standardized
17     fund_rate_z                    standardized
18     last_trade_pnl_pct             stateful (filled by training loop)
19     current_dd_from_peak           stateful

20-25  log_return       lags [60,30,15,5,1,0]
26-31  taker_net_60_z   lags [60,30,15,5,1,0]
32-37  ofi_perp_10      lags [60,30,15,5,1,0]
38-43  vwap_dev_240     lags [60,30,15,5,1,0]
44-49  log_volume_z     lags [60,30,15,5,1,0]
```

**Action mask coverage (≥1 strategy active):** train 29.75%, val 33.58%, test 26.43%.

**Per-strategy signal activity on DQN-train:**

| Strategy | Active bars | % |
|---|---|---|
| S1_VolDir | 26,328 | 14.63% |
| S7_OIDiverg | 11,644 | 6.47% |
| S8_TakerFlow | 12,873 | 7.15% |
| S10_Squeeze | 5,254 | 2.92% |
| S6_TwoSignal | 3,751 | 2.08% |
| S4_MACDTrend | 3,362 | 1.87% |
| S3_BBRevert | 1,211 | 0.67% |
| S2_Funding | 777 | 0.43% |
| S12_VWAPVol | 4 | 0.00% |

---

## Phase 3 — DQN attempts (initial failure)

### Phase 3.1 — Single-trade simulator

- **Module:** [backtest/single_trade.py](../backtest/single_trade.py)
- **Architecture:** numba `@njit(cache=True)` — mirrors [backtest/engine.py:142-213](../backtest/engine.py#L142-L213)

| Test | Result |
|---|---|
| Parity vs `engine.run()` | **50/50 trades match within 1e-9 PnL** |
| Speed | **0.7 µs/call** (target <100 µs) |
| 540k DQN steps overhead | 0.4s |

### Phase 3.2 — Random-policy rollout

- **Module:** [models/dqn_rollout.py](../models/dqn_rollout.py)
- **Network:** [models/dqn_network.py](../models/dqn_network.py) — 50→64→32→10 MLP, 5,674 params
- **Buffer:** [models/dqn_replay.py](../models/dqn_replay.py) — circular numpy + PER

**Random rollout diagnostics:**

| Metric | Value |
|---|---|
| Throughput | 18,920 transitions/s |
| Action 0 (NO_TRADE) | 92.07% |
| Trade actions | 7.93% |
| Win rate | 32.5% |
| Mean trade PnL | −0.235% |
| Final equity | 0.107 (random policy with fees) |

### Phase 3.3 — DQN training (baseline)

- **Module:** [models/dqn_selector.py](../models/dqn_selector.py)
- **Hyperparams:** Adam lr 1e-3, batch 128, γ 0.99, ε 1.0→0.05 over 80k, target sync 1k, PER α=0.6 β=0.4→1.0
- **Reward at training:** trade_pnl − 2×TAKER_FEE (taker fee 0.0008)

| Run | val Sharpe | Best step | Trades on val | Win% | Loss range |
|---|---|---|---|---|---|
| Baseline | **−9.19** | 25,000 | 274 | 39.4% | 0.0024→0.0001 (collapse) |
| Path A (×100 rew + stratified PER) | **−5.87** | 65,000 | 217 | 42.9% | 0.005→0.026 (healthy) |
| Path C (binary {NO_TRADE, S1}) | **−4.70** | 20,000 | 162 | 45.0% | 0.003→0.008 |

All failed Phase 3 gate (>0.5 required).

---

## Group D — Failure diagnostics

### D1 — Supervised PnL predictor

- **Module:** [models/pnl_predictor.py](../models/pnl_predictor.py)
- **Approach:** per-strategy LightGBM regressing trade PnL from 50-dim state + 4 direction probs
- **Training:** DQN-train fires per strategy

**Spearman of predicted vs actual trade PnL:**

| Strategy | n train | val Spearman | test Spearman |
|---|---|---|---|
| S7_OIDiverg | 11,644 | **+0.084** | −0.036 |
| S8_TakerFlow | 12,870 | +0.035 | +0.075 |
| S10_Squeeze | 5,254 | +0.006 | +0.038 |
| S4_MACDTrend | 3,362 | −0.047 | +0.017 |
| S3_BBRevert | 1,211 | −0.076 | −0.017 |
| S1_VolDir | 26,328 | **−0.114** | −0.149 |
| S6_TwoSignal | 3,751 | −0.148 | −0.087 |

**Conclusion:** state has near-zero predictive power for residual trade PnL beyond what strategies already use. (For reference, direction model AUC 0.70 → Spearman ≈ 0.30. Our predictor maxes at 0.084 — 4× weaker.)

### D2 — Hyperparameter grid search

- **Module:** [models/grid_search.py](../models/grid_search.py)
- **Grid:** ~50 combos per strategy across (signal threshold, tp_pct, sl_pct)
- **Total combos:** ~250 across 5 strategies

| Strategy | def val | best val | def test | best test | tr v/t |
|---|---|---|---|---|---|
| S1_VolDir | −6.50 | −1.84 ★ | −4.41 | **+0.10 ★** | 124/62 |
| S4_MACDTrend | −4.78 | −2.48 ★ | −4.12 | −1.99 ★ | 81/52 |
| S6_TwoSignal | −7.18 | −4.87 ★ | **+1.48** | −1.43 ✗ | 102/73 |
| S7_OIDiverg | −26.05 | −14.77 ★ | −32.15 | −21.68 ★ | 219/198 |
| S8_TakerFlow | −10.77 | −1.12 ★ | −5.34 | −3.80 ★ | 93/82 |

**Notable:** S6 default test +1.48 → grid-best test −1.43 (overfitting). Best grid-tuned test is S1 +0.10.

### D3 — Walk-forward (6 folds × 5 strategies × 3 modes)

- **Module:** [models/walk_forward.py](../models/walk_forward.py)
- **Folds:** 6 contiguous ~47k-bar slices over RL period

**Stability gate (≥4/6 folds Sharpe>0 AND mean>0):** **0/75 combinations pass.**

**Per-fold mean Sharpe across all combos:**

| Fold | Period | Mean | Median |
|---|---|---|---|
| 1 | Sep 19 → Oct 22 2025 | −8.10 | −4.85 |
| 2 | Oct 22 → Dec 15 2025 | −7.62 | −4.31 |
| 3 | Dec 15 → Jan 17 2026 | −8.46 | −4.62 |
| 4 | Jan 17 → Feb 19 2026 | **−10.36** | −11.38 |
| 5 | Feb 19 → Mar 23 2026 | −8.09 | −5.91 |
| 6 | Mar 23 → Apr 25 2026 | −8.24 | −5.31 |

Every fold negative — not a "bad single window" issue.

---

## Path 1 — Root cause diagnostics

### Path 1a — Fee-free walk-forward

- **Module:** [models/diagnostics_ab.py](../models/diagnostics_ab.py)

| Strategy | with-fee mean | with-fee pos/6 | fee-free mean | fee-free pos/6 |
|---|---|---|---|---|
| S1_VolDir | −5.66 | 0/6 | **+2.55** | 5/6 |
| S4_MACDTrend | −3.79 | 0/6 | **+1.70** | 5/6 |
| S6_TwoSignal | −5.06 | 1/6 | **+0.41** | 4/6 |
| S7_OIDiverg | −28.52 | 0/6 | **+2.90** | 6/6 |
| S8_TakerFlow | −7.43 | 0/6 | **+3.97** | 6/6 |
| **Grand mean** | **−10.09** | — | **+2.31** | — |

**Δ = +12.40 Sharpe from fee removal alone.**

### Path 1b — Optimal-exit oracle (60-bar lookahead)

| Strategy | Oracle with fee | Oracle fee-free | Oracle win rate (fee-free) | Mean trade PnL fee-free |
|---|---|---|---|---|
| S1_VolDir | +7.79 | **+43.32** | 82.7% | +0.300% |
| S4_MACDTrend | +10.30 | **+23.46** | 87.1% | +0.396% |
| S6_TwoSignal | +5.64 | +30.33 | 86.0% | +0.294% |
| S7_OIDiverg | +10.25 | +42.69 | 88.9% | +0.312% |
| S8_TakerFlow | −1.03 | +38.57 | 83.4% | +0.240% |
| **Grand mean** | **+6.59** | **+35.68** | — | — |

Oracle vs actual gap (with fee): +6.59 − (−10.09) = **+16.68 Sharpe of latent value from better exits**.

### Path 1c — 5-minute timeframe

| | with fee | fee-free |
|---|---|---|
| 1-min grand mean | −10.09 | +2.31 |
| 5-min grand mean | **−6.20** | **+2.33** |
| Δ | +3.89 | +0.02 |

5-min cadence improves with-fee Sharpe by reducing trade frequency (fewer fee events). Fee-free Sharpe is essentially identical → no extra alpha from longer timeframe.

---

## Group A — Fee × penalty sweep

- **Module:** [models/group_a_sweep.py](../models/group_a_sweep.py)
- **Total runtime:** ~25 minutes for 7 cells

**Best step / training trajectory:**

| Cell | Best step | Best val Sharpe | Trades | Win% | Equity | Max DD |
|---|---|---|---|---|---|---|
| A0 | 65,000 | −5.87 | 217 | 42.9% | 0.763 | −26.0% |
| A1 | 35,000 | +5.81 | 287 | 56.4% | 1.290 | −8.1% |
| A2 | 65,000 | **+7.30** | 251 | 55.0% | 1.398 | −6.3% |
| A3 | 10,000 | +5.82 | 326 | 52.5% | 1.311 | −7.6% |
| A4 | 35,000 | **+1.72** | 241 | 50.6% | 1.072 | −7.2% |
| A5 | 25,000 | −0.95 | 275 | 46.5% | 0.948 | −10.5% |
| A6 | 70,000 | −5.38 | 231 | 39.8% | 0.775 | −26.6% |

**Action distribution at best step (% of validation steps):**

| Cell | NO_TRADE | S1 | S2 | S3 | S4 | S6 | S7 | S8 | S10 | S12 |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 97.6 | 1.0 | 0.0 | 0.0 | 0.1 | 0.2 | 0.4 | 0.5 | 0.2 | 0.0 |
| A1 | 96.6 | 1.1 | 0.0 | 0.1 | 0.1 | 0.1 | 0.8 | 0.6 | 0.5 | 0.0 |
| A2 | 97.6 | 0.9 | 0.1 | 0.1 | 0.1 | 0.1 | 0.5 | 0.4 | 0.3 | 0.0 |
| A3 | 93.6 | 1.9 | 0.1 | 0.1 | 0.3 | 0.2 | 1.3 | 1.1 | 1.4 | 0.0 |
| A4 | 97.4 | 1.0 | 0.0 | 0.1 | 0.1 | 0.1 | 0.4 | 0.5 | 0.3 | 0.0 |
| A5 | 95.2 | 1.9 | 0.1 | 0.2 | 0.2 | 0.2 | 1.1 | 0.8 | 0.5 | 0.0 |
| A6 | 98.2 | 0.6 | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 | 0.3 | 0.2 | 0.0 |

DQN concentrates 95–98% on NO_TRADE; trade allocation is on S1, S7, S8, S10. S2, S3, S4, S6, S12 collectively ≤2%.

**Penalty interaction with fee:**

| | penalty=0 | penalty=0.001 | penalty=0.003 |
|---|---|---|---|
| fee=0 | +5.81 | **+7.30 (best)** | +5.82 |
| fee=0.0004 | +1.72 | −0.95 | n/a |
| fee=0.0008 | −5.87 | −5.38 | n/a |

At fee=0 mild penalty helps (DQN focuses on highest-confidence trades). At fee=0.0004 penalty hurts (over-restriction). At taker fee no penalty is enough to fix the policy.

---

## Cross-experiment comparisons

### Best result per fee level

| Fee | Approach | Best result |
|---|---|---|
| 0.0008 (taker) | DQN entry-gate (Group A0/A6) | val Sharpe −5.87 |
| 0.0004 (maker) | **DQN entry-gate (A4, no penalty)** | **val Sharpe +1.72** |
| 0 (oracle) | DQN entry-gate (A2 with 0.1% penalty) | **val Sharpe +7.30** |

### Approach comparison at fee=0.0008

| Approach | val Sharpe |
|---|---|
| Passive (free-firing) | −10.09 (Path 1a grand mean) |
| CUSUM gate | −3.13 (walk-forward grand mean) |
| Grid-tuned single strategy | +0.10 best test (S1) |
| DQN gating (Phase 3) | −9.19 |
| DQN gating (Path A) | −5.87 |
| DQN gating (Path C binary) | −4.70 |
| **DQN gating (Group A0)** | **−5.87** |

All approaches negative under taker fees.

### Approach comparison at fee=0

| Approach | val Sharpe |
|---|---|
| Passive (free-firing) | +2.31 (Path 1a fee-free grand mean) |
| **DQN entry-gating (A1)** | **+5.81** |
| **DQN entry-gating + 0.1% penalty (A2)** | **+7.30** |
| DQN entry-gating + 0.3% penalty (A3) | +5.82 |
| Oracle (perfect exit, fee-free) | +35.68 |

DQN adds +3.5 to +5 Sharpe over passive. Oracle ceiling +28 above DQN — exit-timing RL (Group B) is the next lever.
