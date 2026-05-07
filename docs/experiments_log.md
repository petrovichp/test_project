# Detailed Experiments Log

Companion to [RESULTS.md](../RESULTS.md). Per-experiment numbers, configurations, and per-fold breakdowns.

For all train/val/test split boundaries (predictive models, RL, walk-forward folds): see [data_splits.md](data_splits.md).

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

DQN adds +3.5 to +5 Sharpe over passive. Oracle ceiling +28 above DQN — but exit-timing RL (Group B) failed to close that gap (see below).

---

## Group B — Exit-timing DQN

- **Module:** [models/exit_dqn.py](../models/exit_dqn.py)  +  [models/group_b_sweep.py](../models/group_b_sweep.py)
- **Total runtime:** ~12 min (3 global cells ~2.5 min, 9 per-strategy cells ~9.5 min)

### Formulation

Within each in-trade bar, a 28-dim state is built and the DQN chooses HOLD / EXIT_NOW. Rule-based exits (TP / SL / BE / trail / time-stop) stay active in parallel; whichever fires first ends the trade. Sparse terminal reward (realized PnL net of fees), γ=1.0 within episode (bounded ≤240 bars). Buffer fills via HOLD-biased random exploration (10% EXIT_NOW) so warmup contains full trade trajectories with rule-fired terminals.

Network: 28 → 64 → 32 → 2 (4,002 params). Buffer 80k, batch 128, LR 1e-3, 80k grad steps max with 16k early-stop patience.

### B1-B3 — Global exit DQN across fee levels

Single shared exit policy for all entries (sequential, first-firing strategy at each bar).

| Cell | Fee | Baseline (rule-only) | RL exit | ΔSharpe | Best step | RLexit% | Trades |
|---|---|---|---|---|---|---|---|
| B1 | 0.0008 (taker) | −14.911 | **−22.459** | **−7.55** | 4,000 | 33.5% | 531 |
| B2 | 0.0004 (maker) | −6.810 | **−11.042** | **−4.23** | 20,000 | 31.6% | 512 |
| B3 | 0.0000 (fee-free) | +3.793 | **+2.270** | **−1.52** | 4,000 | 54.8% | 683 |

All three cells negative. Pooling 9 strategies' entries into one exit DQN actively hurts even at fee=0 — strategies have heterogeneous exit signatures the shared policy cannot resolve.

### B4 — Per-strategy exit DQN at maker fee

One DQN per entry strategy (9 sub-runs), each trained only on that strategy's entries.

| Cell | Strategy | Baseline | RL exit | ΔSharpe | Trades | RLexit% |
|---|---|---|---|---|---|---|
| B4_S0 | S1_VolDir    | −4.656  | −4.050  | **+0.606** | 227 | 5.3% |
| B4_S1 | S2_Funding   | −4.467  | −3.510  | **+0.957** | 54  | 40.7% |
| B4_S2 | S3_BBRevert  | −22.407 | −21.403 | **+1.004** | 205 | 12.7% |
| B4_S3 | S4_MACDTrend | −4.228  | **−2.659** | **+1.568** | 119 | 42.9% |
| B4_S4 | S6_TwoSignal | −7.301  | −7.414  | −0.112     | 154 | 33.8% |
| B4_S5 | S7_OIDiverg  | −9.725  | −9.809  | −0.085     | 508 | 27.2% |
| B4_S6 | S8_TakerFlow | −5.055  | **−3.086** | **+1.970** | 245 | 38.8% |
| B4_S7 | S10_Squeeze  | −7.875  | −8.201  | −0.326     | 374 | 36.1% |
| B4_S8 | S12_VWAPVol  | +3.216  | +3.216  | +0.000 (n=1) | 1 | — |

**6/9 positive Δ; best +1.97 (S8_TakerFlow), mean ≈ +0.6.** Per-strategy exit DQNs reliably extract a small lift, but none lift any single strategy into profitable territory at maker fee.

### Decision (per [next_steps.md](next_steps.md) gate)

The +4-Sharpe gate ("captures ≥30% of actual-vs-oracle gap") was not cleared:
- Best B4 lift +1.97 << +4
- B1-B3 negative

Per-strategy exit DQN does provide a real, small improvement (mean +0.6, 6/9 positive). Pooling strategies into a single shared exit DQN is actively harmful. Future exit improvements would require a different formulation (e.g., per-bar dynamic SL placement rather than binary HOLD/EXIT, or signal-driven exit thresholds inside the strategies themselves). The +28 Sharpe oracle gap remains largely unclaimed; most of it likely lives in *intra-bar entry timing* (which the DQN cannot see at 1-min resolution) rather than exit selection.

**Group C handling:** C2 (joint hierarchical training, ~3-5 days) is dropped — joint training only pays off if both stages independently produce strong lifts. C1 (sequential composition of A4 entry + B4 exits, ~hours of code) ran — see Group C section below.

**Production implication:** A4 entry DQN with rule-based exits is the deployable baseline. B4 per-strategy exits add a small optional lift in isolation but **do not transfer when stacked on A4** (see Group C1 below).

---

## Group C1 — A4 entry + B4 per-strategy exits (sequential composition)

- **Module:** [models/group_c_eval.py](../models/group_c_eval.py)
- **Total runtime:** ~4 seconds (no retraining; reuses A4 + 9× B4 policies)

### Result (internal comparison, same simulator)

| Split | Rule-only (A4 + rule) | Combined (A4 + B4) | ΔSharpe | Δ equity | RL exit % |
|---|---|---|---|---|---|
| val  | −5.698 (eq 0.759) | −5.763 (eq 0.752) | **−0.07** | −0.66% | 19.3% |
| test | −3.765 (eq 0.884) | −4.654 (eq 0.850) | **−0.89** | −3.47% | 23.9% |

### Per-strategy attribution (test split)

| Strategy | rule-only n / meanPnL | combined n / meanPnL | combined ΔmeanPnL |
|---|---|---|---|
| S1_VolDir    | 72 / −0.035% | 85 / −0.149% | **−0.114%** |
| S4_MACDTrend | 10 / +0.330% | 11 / +0.354% | +0.024% |
| S7_OIDiverg  | 51 / −0.165% | 71 / −0.076% | **+0.089%** |
| S8_TakerFlow | 56 / +0.065% | 64 / +0.036% | −0.029% |

The B4 exit policy fires too aggressively on S1_VolDir entries (most-traded strategy) — the early exits truncate winners on a strategy that A4 has high confidence in.

### Methodology note — simulator difference

| Evaluator | val Sharpe | test Sharpe | val trades | val mean duration |
|---|---|---|---|---|
| `dqn_selector.evaluate_policy` (Group A original, uncapped lookahead) | **+1.715** | **−1.650** | 241 | uncapped |
| `group_c_eval.evaluate_combined` (240-bar cap to match B4 training) | −5.698 | −3.765 | 326 | ≤240 |

The C1 evaluator caps trade lookahead at 240 bars because B4's state vector includes `n_bars_in_trade / 240` and its policy expects a bounded horizon. This truncates legitimate long-running A4 trades. Within either evaluator the Δ is internally fair, but absolute Sharpe differs. **A4's reported +1.72 is the production-relevant number; the C1 baseline numbers exist only to make the Δ comparison consistent.**

### Decision

C1 confirms that B4 per-strategy exit policies **do not transfer to A4-selected entries**. The training distribution mismatch (sequential first-firing vs A4's selective ~3% picks) is the root cause: B4 learned to bail early on noisy mean trades, but A4 picks higher-quality entries that reward longer holding.

To make joint entry+exit RL work, the exit DQN would need to be retrained on A4's selected entry distribution — which is what C2 (joint hierarchical training, ~3-5 days) was designed to do. **C2 stays dropped** — given C1's negative result (B4 transfer doesn't even break even), the marginal expected value of C2 is low relative to the production-readiness work that's now overdue (A4 walk-forward + seed variance, then Path X).

### Important secondary finding — A4 val/test gap

Group A's headline number was A4 **val** Sharpe +1.72. Re-running through the original simulator on the locked test split gives **−1.65**. A4 has a real val/test degradation that wasn't surfaced in Group A's writeup. **A4 is not yet validated on test** — the "Reduced scope" mini-validation (walk-forward across 6 RL folds + seed variance) is now a hard prerequisite to any production move.

---

## Group B4_fee0 — per-strategy exit DQN at fee=0

To test the fee-drag hypothesis ("does RL exit help when fees aren't a factor?"), 9 per-strategy exit DQNs retrained at fee=0 on each strategy's entries. Same code path as B4 (Group B); only the `--fee 0` flag differs.

| Cell | Strategy | Baseline (fee=0) | RL exit | ΔSharpe |
|---|---|---|---|---|
| B4_fee0_S0 | S1_VolDir    | −0.548 | **+3.389** | **+3.94** |
| B4_fee0_S1 | S2_Funding   | +2.167 | **+5.724** | **+3.56** |
| B4_fee0_S2 | S3_BBRevert  | −1.567 | −0.464     | +1.10 |
| B4_fee0_S3 | S4_MACDTrend | +0.542 | **+4.741** | **+4.20 ★ clears +4 gate** |
| B4_fee0_S4 | S6_TwoSignal | −2.432 | +0.062     | +2.49 |
| B4_fee0_S5 | S7_OIDiverg  | +6.087 | +5.505     | −0.58 |
| B4_fee0_S6 | S8_TakerFlow | +1.960 | +2.936     | +0.98 |
| B4_fee0_S7 | S10_Squeeze  | +2.165 | +3.014     | +0.85 |
| B4_fee0_S8 | S12_VWAPVol  | +3.216 | +3.216     | 0 (n=1) |

**7/9 positive, mean Δ ≈ +1.84, best +4.20 (S4_MACDTrend) — *clears* the +4-Sharpe gate that Group B failed at maker fee.** Confirms the fee-drag hypothesis: RL exit-timing is a real, learnable signal — it just gets eaten by 0.08% round-trip costs at maker fee on 1-min trades.

---

## Group C1_fee0 — A2 entry + B4_fee0 exits at fee=0

The cleanest test of "does RL exit stack on RL entry when fees aren't dragging".

### Internal comparison (same evaluator, 240-bar cap)

| Split | Rule-only (A2 + rule exits) | Combined (A2 + B4_fee0 exits) | ΔSharpe | Δ equity |
|---|---|---|---|---|
| val  | +3.876 (eq 1.181) | +3.928 (eq 1.172) | **+0.05** | −0.94% |
| test | +6.979 (eq 1.244) | +4.280 (eq 1.143) | **−2.70** | −10.06% |

### A2 baseline through original simulator (uncapped, apples-to-apples with Group A reporting)

| Split | A2 alone, fee=0 | Trades | Win % | Equity | Max DD |
|---|---|---|---|---|---|
| val  | **+7.295** | 251 | 55.0% | 1.398 | −6.31% |
| test | **+3.776** | 185 | 55.1% | 1.127 | −9.69% |

A2 reproduces its reported +7.30 on val and **generalizes to +3.78 on the locked test split** with equity 1.13× — a real, deployable signal at fee=0.

### Per-strategy attribution (test split, C1_fee0 evaluator)

| Strategy | rule-only meanPnL | combined meanPnL | Δ |
|---|---|---|---|
| S1_VolDir    | +0.133% | +0.095% | −0.04% |
| **S4_MACDTrend** | **+0.507%** | +0.144% | **−0.36%** (huge drop) |
| S7_OIDiverg  | −0.043% | −0.014% | +0.03% |
| S8_TakerFlow | +0.093% | +0.043% | −0.05% |
| **S10_Squeeze**  | **+0.122%** | +0.002% | **−0.12%** |

The strategies that A2 trades best on (S4_MACDTrend, S10_Squeeze) get hurt the most. RL exits truncate winners on high-quality entries.

### Verdict

C1_fee0 confirms what C1 already showed at maker fee: **B4 per-strategy exit policies do not transfer to RL-gated entries even at fee=0**. The transfer pathology is *structural*, not fee-related. Each individual policy works on its own training distribution but they don't compose.

**Why:** B4 trains on the "sequential first-firing" entry distribution (~30% bar coverage). A2 picks ~3% selective entries with much higher per-trade edge. B4's policy learned to bail early on average noisy trades; on A2's high-quality entries the early-bail destroys winners.

This is exactly the use-case for **C2 (joint hierarchical training)** — where the exit DQN sees the entry DQN's actual selected entries during training. C2 would cost ~3-5 days. We're not doing it now because:

1. Production-readiness work (A4 walk-forward + seed variance, Path X maker execution) has higher expected value
2. A2's standalone fee-free result (val +7.30, test +3.78) is already strong; production deployment via maker fees would shift conditions back toward fee-free regime, where A2 alone is the deployable target
3. Adding C2 is a future enhancement, not a blocker

### Updated headline takeaways

- **The trading signal works** (A2 val +7.30, test +3.78 fee-free, equity 1.13× over locked test split)
- **RL exit-timing is real** (B4_fee0 clears +4 gate on best strategy, mean +1.84)
- **Variable-length exit DQN doesn't compose** (C1/C1_fee0 both fail to transfer; C1 maker Δ -0.07/-0.89; C1_fee0 fee-free Δ +0.05/-2.70)
- **Fixed-window exit DQN does compose on test** (see Group B5 + C2 below)

---

## Group B5 + C2 — fixed-window exit DQN with enriched state

C1's failure motivated a redesign. Two structural problems with B4 were identified: variable episode length (rule-fired terminals dominated the buffer) and the rule-vs-DQN race in credit assignment. Group B5 fixes both.

### Design changes

| | B4 | **B5** |
|---|---|---|
| Episode length | 1–240 bars (rule-determined) | **fixed N ∈ {60, 120, 240}** |
| Rule-based exits during training | TP/SL/BE/trail/time-stop active | **disabled** — only DQN's EXIT_NOW or window-edge can terminate |
| State dim | 28 | **53** |
| Network | 28→64→32→2 (4,002 params) | 53→96→48→2 (10,114 params) |

### B5 state vector (53 dims)

```
─── In-trade scalars (8) ───
 0  unrealized_pnl_pct                   clip ±10
 1  bars_in_trade / N
 2  bars_remaining / N
 3  entry_direction                      ±1
 4  max_unrealized_pnl_so_far            clip 0..10
 5  min_unrealized_pnl_so_far            clip -10..0
 6  bars_since_peak / N
 7  realized_vol_in_trade

─── Cyclic time (2) ───
 8  hour_of_day_sin
 9  hour_of_day_cos

─── PRICE PATH (20) — last 20 bars cum-return-from-entry × 100 ───
10..29  padded with 0 for bars before entry

─── VOLATILITY WINDOW (10) — last 10 bars |log_return| standardized ───
30..39

─── Entry-time static (3) ───
40  vol_pred at entry         (sliced from base state[entry][0])
41  bb_width at entry         (sliced from base state[entry][16])
42  regime_id at entry / 4.0

─── Current market aggregates (10) ───
43..48  log_return × 6 lags    (sliced from base state[t][20:26])
49..52  taker_net_60_z × 4 lags (sliced from base state[t][28:32])
```

### B5 per-strategy results at fee=0 (27 cells = 3 windows × 9 strategies)

#### Window N=120 bars (2 hours)

| Cell | Strategy | Baseline (always-HOLD-to-N) | B5 RL | ΔSharpe | RL exit % |
|---|---|---|---|---|---|
| B5_fix120_fee0_S0 | S1_VolDir    | −2.374 | +1.618 | +3.992 | 82.4% |
| B5_fix120_fee0_S1 | S2_Funding   | +0.661 | +2.405 | +1.744 | 74.5% |
| B5_fix120_fee0_S2 | S3_BBRevert  | −3.192 | −2.369 | +0.823 | 63.0% |
| B5_fix120_fee0_S3 | S4_MACDTrend | −1.952 | +0.178 | +2.131 | 54.3% |
| **B5_fix120_fee0_S4** | **S6_TwoSignal** | **−4.737** | +0.164 | **+4.901** | 76.9% |
| B5_fix120_fee0_S5 | S7_OIDiverg  | +3.338 | **+4.346** | +1.008 | 4.0% |
| B5_fix120_fee0_S6 | S8_TakerFlow | −1.936 | +1.106 | +3.042 | 16.0% |
| B5_fix120_fee0_S7 | S10_Squeeze  | −0.969 | −0.784 | +0.186 | 57.3% |
| B5_fix120_fee0_S8 | S12_VWAPVol  | +3.216 | +3.216 | 0      | n=1 |

8/9 positive Δ, mean Δ **+1.98**, max Δ +4.90.

#### Window N=240 bars (4 hours) ★ best

| Cell | Strategy | Baseline | B5 RL | ΔSharpe | RL exit % |
|---|---|---|---|---|---|
| B5_fix240_fee0_S0 | S1_VolDir    | −2.893 | +1.204 | +4.098 | 51.8% |
| B5_fix240_fee0_S1 | S2_Funding   | −3.337 | +0.349 | +3.686 | 73.1% |
| B5_fix240_fee0_S2 | S3_BBRevert  | −6.833 | −4.681 | +2.151 | 78.8% |
| B5_fix240_fee0_S3 | S4_MACDTrend | −0.367 | +1.562 | +1.929 | 73.7% |
| B5_fix240_fee0_S4 | S6_TwoSignal | −4.618 | −0.252 | +4.366 | 88.8% |
| B5_fix240_fee0_S5 | S7_OIDiverg  | +3.745 | **+5.905** | +2.159 | 13.3% |
| B5_fix240_fee0_S6 | S8_TakerFlow | −4.909 | +1.364 | +6.273 | 61.0% |
| **B5_fix240_fee0_S7** | **S10_Squeeze** | **−3.546** | **+3.142** | **+6.689** | 29.2% |
| B5_fix240_fee0_S8 | S12_VWAPVol  | +3.216 | +3.216 | 0      | n=1 |

**9/9 positive Δ, mean Δ +3.48**, max Δ **+6.69 (S10_Squeeze)**, best abs Sharpe +5.91 (S7_OIDiverg).

#### Summary across windows

| Window | n positive Δ | mean Δ | max Δ | best abs Sharpe |
|---|---|---|---|---|
| N=60 | 8/9 | +2.29 | +6.25 (S1_VolDir) | +6.02 (S2_Funding) |
| N=120 | 8/9 | +1.98 | +4.90 (S6_TwoSignal) | +4.35 (S7_OIDiverg) |
| **N=240** | **9/9** | **+3.48** | **+6.69 (S10_Squeeze)** | +5.91 (S7_OIDiverg) |

---

## Group C2 — A2 entry + B5 fixed-window exits stacked

The actual production-relevant test: stack the trained B5 per-strategy exit policies on top of A2's selective entry policy at fee=0.

### Result table

| Configuration | val Sharpe | val equity | test Sharpe | test equity | test max DD | test win % |
|---|---|---|---|---|---|---|
| A2 alone + rule-based exits (production target) | **+7.295** | 1.398 | +3.776 | 1.127 | −9.69% | 56.1% |
| A2 + always-HOLD-to-60 (no exits) | −3.918 | 0.840 | −1.046 | 0.965 | −16.03% | 52.2% |
| C2_fix60 (A2 + B5_fix60 RL exits) | +1.693 | 1.064 | −1.219 | 0.957 | −14.15% | 56.6% |
| A2 + always-HOLD-to-120 | −6.816 | 0.730 | +2.564 | 1.089 | — | 54.2% |
| C2_fix120 (A2 + B5_fix120 exits) | −2.478 | 0.870 | +3.717 | 1.129 | — | 65.2% |
| A2 + always-HOLD-to-240 | −3.604 | 0.833 | +5.245 | 1.217 | −5.74% | 56.5% |
| **C2_fix240 (A2 + B5_fix240 exits)** | **−1.426** | 0.927 | **+8.329** | **1.343** | **−4.29%** | **67.0%** |

### Reading

- **Test side**: C2_fix240 is the best result the project has produced. **+8.33 Sharpe vs A2-alone +3.78**, equity 1.343× vs 1.127×, lower drawdown, higher win rate.
- **Val side**: C2_fix240 is −1.43, much worse than A2-alone +7.30. The val/test inversion is unusual.
- **Window matters**: N=240 dominates N=120 dominates N=60 in this composition. Strategies need longer horizons to fully express edge — short windows force exits before trends complete.
- **RL exit rate**: C2_fix240 fires RL_EXIT on 87 of 206 trades (42%) on test, captures most of A2-baseline-without-rules' improvement and extracts additional alpha on top.

### Why fixed-window helped (vs C1's failure)

1. **No rule-fired terminals in buffer** — the DQN owns every terminal decision. Sparse-reward credit assignment is no longer confounded by rules
2. **Right-tail visibility** — the DQN observes what happens when you hold a trade all the way to bar N including catastrophic losing tails. It learns to cut losses *because* it sees them
3. **Strategy-config independence** — B5 doesn't read `EXECUTION_CONFIG` thresholds. Its policy depends only on in-trade state, so it generalizes across entry distributions
4. **Richer state** — the price-path window and trajectory scalars give the DQN direct visibility into the trade's profit history, not just a current snapshot

### Why val/test asymmetry exists

- B5 per-strategy policies were val-best-checkpointed *per strategy* on the dense entry distribution, not on the composition
- Composition val (C2_val) is therefore genuinely OOS for the stacked system — it's not a selection artifact
- The B5 policies that work well on per-strategy val (e.g., S7_OIDiverg val +5.91) don't necessarily transfer to A2's selective subset of S7 entries on val
- On test, the patterns evidently align — B5's exits and A2's entries compose well

### Verdict and forward path

See § Walk-forward below — the walk-forward result reverses the apparent C2_fix240 breakthrough.

---

## Walk-forward validation — A2 entry across 6 RL folds (DECISIVE)

- **Module:** [models/group_c2_walkforward.py](../models/group_c2_walkforward.py)
- **Total runtime:** ~15 seconds (no retraining; uses existing A2 + B5_fix240 policies)
- **Folds:** 6 contiguous ~47,195-bar slices over the full RL period (Sep 2025 → Apr 2026)
- **Purpose:** verify whether C2_fix240's single-shot test +8.33 was a real signal or window-specific

### Per-fold results

| Fold | In-sample? | Date range | A2 + rule (Sharpe / eq) | A2 + B5 (Sharpe / eq) | A2 + no-exit (Sharpe / eq) | Δ vs rule |
|---|---|---|---|---|---|---|
| 1 | yes | 2025-09-20 → 10-22 | **+13.08** / 1.711 | +7.12 / 1.343 | +1.35 / 1.042 | −5.96 |
| 2 | yes | 2025-10-22 → 12-15 | **+14.82** / 2.228 | +0.33 / 0.998 | −5.03 / 0.698 | **−14.49** |
| 3 | yes | 2025-12-15 → 01-17 | **+6.17** / 1.212 | +1.47 / 1.041 | −3.46 / 0.901 | −4.71 |
| 4 | yes | 2026-01-17 → 02-19 | **+9.34** / 1.632 | +1.84 / 1.072 | +5.06 / 1.280 | −7.51 |
| 5 | partial | 2026-02-19 → 03-24 | **+8.14** / 1.432 | −0.61 / 0.960 | −3.24 / 0.856 | −8.75 |
| 6 | OOS (test) | 2026-03-24 → 04-25 | +2.46 / 1.069 | **+5.40** / 1.172 | +4.41 / 1.150 | **+2.94** |

### Aggregate

| Configuration | Mean Sharpe | Median | Folds positive |
|---|---|---|---|
| **A2 + rule-based** | **+9.00** | +8.74 | **6/6** ✓ |
| A2 + B5_fix240 (C2) | +2.59 | +1.65 | 5/6 |
| A2 + always-HOLD-to-240 (no-exit) | −0.15 | — | 3/6 |
| **Δ (B5 − rule)** | **−6.41** | −6.74 | **1/6 (fold 6 only)** |

### Reading

1. **A2 + rule-based dominates**: 6/6 folds positive, mean +9.00 Sharpe, equity 1.07× to 2.23× per ~32-day fold. Rule-based exits combined with A2's selective entries form the deployable system.

2. **C2_fix240 is positive in 5/6 folds (mean +2.59)** — so the policy is *not broken*, it just doesn't beat rule-based. The single positive Δ comes from fold 6 (the test split), where rule-based was uncharacteristically weak.

3. **Fold 6 is anomalous, not C2_fix240**: A2 + rule-based scored only +2.46 in fold 6 vs +6 to +15 in folds 1-5. In that stressed regime, B5's earlier exits cut larger losses. The original test +8.33 result was real but overrated as a structural improvement — it was a fold-specific advantage.

4. **No-exit baseline (always-HOLD-to-240) confirms exits matter**: mean −0.15, only 3/6 folds positive. Letting trades run to bar 240 unmanaged is bad; *some* exit policy is required.

5. **In-sample folds 1-3 do NOT show systematically lower B5 performance than out-of-sample folds 5-6**, ruling out simple overfitting as the explanation. B5's underperformance vs rule-based is structural — binary HOLD/EXIT_NOW cannot replicate rule-based TP capture and trail-after-breakeven mechanics.

### Why rule-based wins (mechanistically)

Rule-based exits have three structural advantages B5 can't match:

1. **TP capture**: rule TP at 1.5–3% (ATR-scaled per strategy) locks in trend-mode wins. B5's binary HOLD/EXIT_NOW with bounded 240-bar window tends to exit *before* TP fires (it doesn't know about TP)
2. **Trail-after-breakeven**: rule trail ratchets SL up with peak price, locking partial profit. B5 has no equivalent ratchet mechanism — it can only choose to exit at a bar, not adjust the SL
3. **Per-strategy tuning**: each rule TP/SL/trail/be is sized to the strategy's signal characteristic via `EXECUTION_CONFIG`. B5 trains on bar-level state without strategy-specific exit shaping

On strategies that produce TP-friendly trade trajectories (the majority, in folds 1–5), rule-based wins decisively. B5 only wins when the regime shifts and TP rarely fires (fold 6) — then early loss-cutting beats waiting for rules.

### Decision

**Deployment target: A2 entry + rule-based exits.** Walk-forward 6/6 positive, mean Sharpe +9.00, robust across the full RL period.

C2_fix240 (A2 + B5 RL exits) is closed as a production candidate but retains optional value as a regime-stress fallback or future research direction (joint hierarchical training).

---

## Trail-after-breakeven extension to mean-reversion strategies (negative result, 2026-05-07)

### Hypothesis

Currently 5/9 strategies have `trail_after_breakeven=True` in [execution/config.py](../execution/config.py): S1, S4, S6, S8, S10 (the trend strategies). The other 4 — **S2_Funding, S3_BBRevert, S7_OIDiverg, S12_VWAPVol** (mean-reversion) — only have `breakeven` (SL moves to entry once profitable) but no subsequent trail. Enabling trail-after-BE for those 4 *might* lock in additional partial profits and lift Sharpe.

### Experiment

Set `trail_after_breakeven=True` for the four mean-reversion strategies. Trail distance auto-defaults to each strategy's `sl_pct` (S2=0.5%, S3=0.4%, S7=0.5%, S12=0.6%) per `ComboExit.plan()` logic. Re-ran walk-forward across 6 RL folds.

### Result — degradation

| Fold | Baseline (no TAB on mean-rev) | With TAB on S2/S3/S7/S12 | Δ |
|---|---|---|---|
| 1 | +13.079 | +13.084 | +0.005 |
| 2 | **+14.820** | **+12.844** | **−1.976** |
| 3 | +6.172 | +6.220 | +0.048 |
| 4 | +9.343 | +9.551 | +0.208 |
| 5 | **+8.139** | **+7.108** | **−1.031** |
| 6 | +2.457 | +2.568 | +0.111 |
| **Aggregate** | **+9.001** | **+8.562** | **−0.439** |

Folds 1, 3, 4, 6: roughly neutral (±0.2). Folds 2 and 5 lose 1-2 Sharpe each, dragging the mean down by 0.44.

### Why it hurt

The four affected strategies are all **mean-reversion** in nature. Their trade trajectory is fundamentally different from trend strategies:

- **Trend strategy (e.g., S1_VolDir):** price climbs monotonically, peaks, retraces. Trail-after-BE locks in the peak — wins.
- **Mean-reversion strategy (e.g., S3_BBRevert):** price moves toward a *target* (e.g., Bollinger band center, fair-value VWAP). Trajectory is non-monotonic — price often overshoots in the entry direction, retraces past breakeven, then continues to the reversion target. The trail prematurely exits during the retrace **before the reversion completes**.

Concrete example for S3_BBRevert:
- Entry at lower BB at $100 (long, expecting reversion to mean $101)
- Price moves to $100.50 (+0.5% — BE triggers, SL moves to $100, trail activates at 0.4%)
- Price retraces to $100.10 → trail-SL = max($100, $100.50 × 0.996) = $100.098
- Price drops to $99.95 → hits trail-SL at $100.098 → **exits at +0.10%**
- ...but had the trade held, price would continue mean-reversion to $101 → **+1.0%**

The trail mechanism assumes "post-profit retrace = bad signal" (correct for trend), but for mean-reversion a small retrace is normal price discovery, not a thesis reversal. The original config decision to leave mean-reversion strategies without trail was principled.

### Decision

**Reverted.** `execution/config.py` restored to original 5/9 `trail_after_breakeven=True` distribution. Lesson preserved here for the record.

### Lesson for future tuning

Exit mechanics need to match the strategy's signal type:

| Strategy type | Best exit mechanism |
|---|---|
| **Trend / momentum** (signal continues) | TP + breakeven + **trail-after-BE** |
| **Mean-reversion** (signal targets a level) | TP + breakeven (no trail) + time-stop |

This is already encoded in the existing config — confirmed correct by walk-forward.
