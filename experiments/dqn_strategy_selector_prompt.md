# Training Prompt — DQN Strategy Selector (v5)

## Goal
Train a small DQN that, at each bar, selects **which strategy to fire** (or no-trade) using market context + regime labels + volatility signal + per-strategy signal activity + a **multi-scale window** of recent price/flow/volume dynamics. Replaces the hand-coded `REGIME_GATE` dict.

**Success criterion:** test-split Sharpe > 1.0, ≥4 of 6 walk-forward folds positive, beats CUSUM-gate baseline (S4 +3.13).

---

## Key changes from v4

1. **Add a sparse multi-scale window** to 5 features (log_return, taker_net_60_z, ofi_perp_10, vwap_dev_240, log_volume_z). Sampled at lags `[60, 30, 15, 5, 1, 0]` per feature → 30 new dims.
2. **State vector grows from 23 → 50 dims**, network params from ~4k → ~5.7k. Still tiny, still CPU.
3. The 3 single-value market features that were windowed (`taker_net_60_z`, `ofi_perp_10`, `vwap_dev_240`) are removed from the static block to avoid duplication.
4. Adds `log_return` and `log_volume_z` as new feature derivations — computed from the existing features parquet, no new ML training.

---

## Inherited from v4 (unchanged)

1. **Treat bars as a continuous numerical sequence**. Real-time gaps between consecutive bars are ignored — features computed via index-based rolling.
2. **Drop only the first 1,440 warmup bars**. After that, features parquet has zero NaN.
3. **No T1/T2/T3 tiering, no imputation**. Full 191-feature set used by upstream models.
4. **Vol-train = 100k bars** (bars 1.4k–101.4k). RL period = 283k bars.

---

## Why this is now simple

The current feature engineering uses `pd.rolling(window=N, min_periods=N).func()`. Pandas operates on the *sequence index*, not on timestamps — a 60-bar window means "the previous 60 entries" regardless of whether those entries are 60 seconds apart, 5 minutes apart, or split by a 24-hour outage. So **features are already computable on every bar** after the initial warmup.

Verified empirically: 99.6% of rows in `cache/btc_features_assembled.parquet` are fully valid (all 191 columns non-NaN). The 0.4% NaN is concentrated in the first 1,440 bars (the longest window's warmup).

`data/gaps.py:clean_mask` was applied separately — a *semantic* filter that says "skip bars whose lookback window straddles a real-time gap because the feature value will represent an unusual time interval". For the DQN, we **opt out** of this filter and accept that some bars' features represent across-gap time periods. The DQN learns to handle whatever pattern is there.

---

## Data scope: 384,614 bars → 383,174 usable

```
Drop warmup        bars      0 → 1,440         Jul 4 → ~Jul 5 2025  (1,440 bars,  ≈1 day)

Vol-train          bars  1,440 → 101,440       Jul 5 → ~Sep 20 2025   100,000 bars (~77 days)
                                                ↑ vol LightGBM retrained on full 191-feature set

DQN-train          bars 101,440 → 281,440      ~Sep 20 → ~Feb 5 2026  180,000 bars (~138 days)
DQN-val            bars 281,440 → 332,307      ~Feb 5  → ~Mar 16 2026  50,867 bars (~40 days)
DQN-test           bars 332,307 → 384,614      ~Mar 16 → Apr 25 2026   52,307 bars (~40 days)
```

(Date ranges approximate — bars are ~1303/day average; computed from real timestamps on the actual cleaned bars.)

---

## State space — 50 dimensions

### Static block (20 dims)

| Group | Features | Dim |
|---|---|---|
| Vol signal | `vol_pred, atr_pred_norm` | 2 |
| Regime one-hot (CUSUM, 5 states) | `is_calm, is_trend_up, is_trend_down, is_ranging, is_chop` | 5 |
| Active signal flags | `s1, s2, s3, s4, s6, s7, s8, s10, s12` | 9 |
| Static market context | `bb_width, fund_rate_z` | 2 |
| Recent perf | `last_trade_pnl_pct, current_dd_from_peak` | 2 |

### Windowed block (30 dims) — sparse multi-scale

For each of 5 features, take values at lags `[60, 30, 15, 5, 1, 0]` (6 positions per feature):

| Feature | Definition | Source |
|---|---|---|
| `log_return` | `log(price[t-k] / price[t-k-1])` | derived from `perp_ask_price` |
| `taker_net_60_z` | rolling-60 taker net flow z-score | features parquet |
| `ofi_perp_10` | order-flow imbalance, 10-bar | features parquet |
| `vwap_dev_240` | (price − vwap_240) / vwap_240 | features parquet |
| `log_volume_z` | `(log(perp_minute_volume) − log_v_med) / log_v_iqr` | derived from `perp_minute_volume` |

Encoding per feature: `[lag60, lag30, lag15, lag5, lag1, lag0]` ordered oldest → newest.

### Full layout (indices 0–49)

```
INDEX     FEATURE                                    RANGE
─────     ──────────────────────────────────────    ──────────────
  0       vol_pred                                    ~ N(0, 1)
  1       atr_pred_norm                               ~ N(0, 1)
  2-6     regime one-hot (5)                          {0, 1}
  7-15    signal flags (9)                            {-1, 0, +1}
  16      bb_width                                    ~ N(0, 1)
  17      fund_rate_z                                 ~ N(0, 1)
  18      last_trade_pnl_pct                          ~ N(0, 1)
  19      current_dd_from_peak                        ≤ 0, scaled

  20-25   log_return @ lags [60, 30, 15, 5, 1, 0]    ~ N(0, 1)
  26-31   taker_net_60_z @ lags [60, 30, 15, 5, 1, 0] ~ N(0, 1)
  32-37   ofi_perp_10 @ lags [60, 30, 15, 5, 1, 0]   ~ N(0, 1)
  38-43   vwap_dev_240 @ lags [60, 30, 15, 5, 1, 0]  ~ N(0, 1)
  44-49   log_volume_z @ lags [60, 30, 15, 5, 1, 0]  ~ N(0, 1)
```

All numerical features standardized using **vol-train (bars 1,440 → 101,440) median + IQR**. **Same statistics applied across all lag positions** of a windowed feature (not per-lag). Standardization stats cached as `cache/btc_dqn_standardize_v5.json`.

### Lag boundary handling

For bars where a lag would index before the warmup boundary (bar 1,440), pad with zero (post-standardization). This affects only ~60 bars at the very start of vol-train and is rare (DQN-train starts at bar 101,440 — well past any padding).

Implementation note: build the state arrays in vectorized form using `np.lib.stride_tricks.sliding_window_view` with offsets for the 6 lag positions, then index per feature.

---

## Action space

10 discrete actions, unchanged:
```
0     = NO_TRADE
1..9  = S1, S2, S3, S4, S6, S7, S8, S10, S12
```

Per-bar action mask:
```
valid_actions[t] = {0} ∪ {k : strategy_k_signal[t] != 0}
```

`Q-values[~valid_actions] = -1e9` before argmax/softmax.

No additional NaN-related masking — all features valid on bars 1,440+.

---

## Reward function

Trade-level delayed reward via `backtest/engine.py`:
```
r_t = trade.pnl_pct − 2·TAKER_FEE     (when trade closes)
r_t = 0                               (NO_TRADE bars or position open)
γ   = 0.99
```

Use **Huber loss**, n-step return (n = trade duration), prioritized experience replay (PER) with α=0.6, β=0.4→1.0.

---

## Network architecture

### Topology

A 3-layer feed-forward MLP, deliberately small (~5.7k params):

```
INPUT  state vector  ────────────────────► (B, 50)   batch B, 50-dim state
                                                     (20 static + 30 windowed)
                                                          │
                                                          ▼
                                ┌──────────────────────────────────────────┐
                                │  Linear:  50 → 64                        │
                                │           Kaiming init (gain=ReLU)       │
                                │  ReLU                                    │
                                └──────────────────────────────────────────┘
                                                          │
                                                          ▼
                                ┌──────────────────────────────────────────┐
                                │  Linear:  64 → 32                        │
                                │  ReLU                                    │
                                └──────────────────────────────────────────┘
                                                          │
                                                          ▼
                                ┌──────────────────────────────────────────┐
                                │  Linear:  32 → 10                        │
                                │  (logits — no activation)                │
                                └──────────────────────────────────────────┘
                                                          │
                                                          ▼
OUTPUT  Q-values  ───────────────────────► (B, 10)   one Q per action
```

```python
class DQN(nn.Module):
    def __init__(self, state_dim=50, n_actions=10, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_actions)
        # Kaiming-uniform default fits ReLU; no special init needed
        # No BatchNorm, no Dropout — small net, off-policy sampling

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)              # raw logits (Q-values), shape (B, n_actions)
```

**Parameter count**

| Layer | Params |
|---|---|
| `fc1`: 50 × 64 + 64 | 3,264 |
| `fc2`: 64 × 32 + 32 | 2,080 |
| `fc3`: 32 × 10 + 10 | 330 |
| **Total** | **5,674** |

CPU is fine (~0.07 ms per forward pass on a single bar). GPU offers no benefit at this scale.

### Why this size and shape

| Choice | Rationale |
|---|---|
| **MLP, not LSTM/CNN** | The state vector already includes recent-history compressed signals (`taker_net_60_z`, `vwap_dev_240`, `last_trade_pnl_pct`). A recurrent layer adds optimization difficulty without new information. |
| **3 layers** | Need at least 2 hidden layers to learn nonlinear interactions between regime × signal-active flags × vol_pred. A third layer adds expressivity for cross-feature gating logic. |
| **64→32 hidden widths** | With 22 inputs and 10 outputs, fan-in/fan-out of ~2× per layer is conservative and prevents over-parameterization on ~180k training transitions. |
| **No BatchNorm** | Off-policy RL has non-stationary state distributions (ε decays, replay sampling shifts). BN's running statistics get poisoned. |
| **No Dropout** | Q-learning targets are noisy enough already; dropout adds variance without a clear regularization win on a 4k-param net. |
| **No double DQN, dueling, distributional** | Vanilla DQN is the right baseline. Add complexity only if vanilla converges and underperforms the CUSUM baseline. |
| **ReLU** | Standard; gradient flow is fine at this depth. |

### Two networks: online + target

Standard DQN pair:

```
online_net   ── forward + backprop happens here ──► weights updated every step
                                                          │
                                                          │  hard copy every 1000 grad steps
                                                          ▼
target_net   ── forward only, no_grad ──────────────► weights frozen between syncs
```

Why two:
- The target network supplies stable Q-value estimates for the Bellman target. Without it, the online network "chases its own tail" and diverges.
- Hard copy (not Polyak soft updates) — simpler and works fine at this scale.

### Forward pass with action masking

Masking happens **after** the network outputs Q-values, before any argmax / max / softmax. This is critical for both inference and Bellman targets:

```python
def masked_argmax(net, state, valid_actions_mask):
    """Inference path. valid_actions_mask: (B, 10) bool."""
    with torch.no_grad():
        q = net(state)                             # (B, 10) logits
        q = q.masked_fill(~valid_actions_mask, -1e9)
        return q.argmax(dim=-1)                    # (B,)

def masked_max(target_net, state_next, valid_mask_next):
    """Bellman target path. Used inside the loss."""
    with torch.no_grad():
        q_next = target_net(state_next)
        q_next = q_next.masked_fill(~valid_mask_next, -1e9)
        return q_next.max(dim=-1).values           # (B,)
```

Masking with `-1e9` (not `-inf`): avoids NaN propagation if the gradient ever reaches the masked entries (it shouldn't, but defensive).

### Loss function

**n-step Bellman target with Huber loss.**

Each replay-buffer transition stores a sequence of intermediate rewards `[r_0, r_1, ..., r_{n-1}]` collected over the trade's life (n = trade duration in bars; 1 if NO_TRADE). The target is:

```
G_n = r_0 + γ r_1 + γ² r_2 + ... + γ^{n-1} r_{n-1}
target = G_n + γ^n · max_{a' ∈ valid'} Q_target(s', a')
```

For terminal transitions (end of split), drop the bootstrap:
```
target = G_n
```

The loss compares the online network's `Q(s, a)` against this target:

```python
def n_step_loss(online_net, target_net, batch, γ):
    s, a, rewards_seq, lengths, s_next, valid_next, done, is_weights = batch
    # rewards_seq: padded to max length; lengths: actual n per sample
    # s, s_next: (B, 22). a: (B,). is_weights: PER importance-sampling weights.

    q_sa = online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)        # (B,)

    # Build target G_n + γ^n max Q'(s')
    discounts = γ ** torch.arange(rewards_seq.size(1))                # (T,)
    G_n = (rewards_seq * discounts).sum(dim=1)                        # (B,) — already padded with 0s past length
    
    bootstrap = torch.zeros(B)
    not_terminal = ~done
    if not_terminal.any():
        with torch.no_grad():
            q_next = target_net(s_next).masked_fill(~valid_next, -1e9)
            bootstrap[not_terminal] = q_next[not_terminal].max(dim=-1).values * (γ ** lengths[not_terminal].float())

    target = G_n + bootstrap
    td_error = q_sa - target
    loss = (is_weights * F.huber_loss(q_sa, target, reduction='none', delta=1.0)).mean()

    # Update PER priorities outside this fn:
    # buffer.update_priorities(idx, td_error.abs().detach().cpu().numpy() + 1e-6)
    return loss, td_error.detach()
```

**Why Huber, not MSE:** trade PnL distribution has fat tails (rare 3%+ winners, occasional -1.5% gappy fills). MSE squares those into massive gradients that destabilize training. Huber is linear past `delta=1.0`, capping the gradient magnitude.

**Why n-step:** trades close anywhere from 1 to 120 bars after entry. A 1-step Bellman would credit the entry action with only the next-bar PnL, ignoring the bulk of the trade outcome. n-step propagates the actual realized PnL back to the entry decision.

### Replay buffer schema

```
Each transition (one entry per DQN-train bar):

  state          : (22,)   float32   — pre-decision state
  action         : ()      int8      — chosen action 0..9
  rewards_seq    : (n,)    float32   — per-bar realized PnL during trade life
                                       (length n = 1 for NO_TRADE, τ for trade)
  state_next     : (22,)   float32   — state at trade close (or next bar if NO_TRADE)
  valid_actions_next: (10,) bool     — mask at state_next
  done           : ()      bool      — true if trade hits end of split
  priority       : ()      float32   — abs(TD error) + small ε  (PER)
```

Storage: numpy arrays in a circular buffer (capacity 80k). Variable-length `rewards_seq` stored as list-of-arrays or fixed-width (pad with 0, store length separately).

### Optimizer + training schedule

```
optimizer       : Adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
lr scheduler    : none (constant)
gradient clip   : max_norm=10.0  (defensive — Huber bounds individual TD errors but batches can spike)
weight decay    : 0  (with PER + replay buffer, regularization comes from data scale)
```

### Hyperparameter cheat-sheet

| Param | Value | Notes |
|---|---|---|
| `state_dim` | 50 | fixed (20 static + 30 windowed) |
| `n_actions` | 10 | fixed |
| `hidden` | 64 (then 32) | enlarge to 128 if undertraining; shrink to 32 if overfitting |
| `γ` | 0.99 | trades take τ ≤ 120 bars; γ^120 ≈ 0.30, ample propagation |
| `target_sync_steps` | 1000 | stable; could try Polyak τ=0.005 if instability |
| `lr` | 1e-3 | Adam default; reduce to 5e-4 if loss oscillates |
| `batch_size` | 128 | balances variance vs. gradient quality |
| `buffer_size` | 80,000 | covers ~half of DQN-train transitions |
| `ε` schedule | 1.0 → 0.05 over 80k steps | linear; explores ~50% of training time |
| `n_epochs` | 3 | through 180k DQN-train bars → ~540k steps |
| `warmup_steps` | 5,000 | random actions before first gradient update |
| `huber_delta` | 1.0 | matches typical reward magnitude (~0.01 = 1%) |
| `grad_clip_norm` | 10.0 | cuts off rare exploding gradients |
| `PER α` | 0.6 | how much priority is used (0 = uniform, 1 = pure priority) |
| `PER β` | 0.4 → 1.0 over training | importance-sampling correction |

---

## Training procedure

```
buffer_size:      80_000          (4× v1 — much more RL data available)
batch_size:       128
γ:                0.99
lr:               1e-3 (Adam)
ε start → end:    1.0 → 0.05 over 80_000 steps (linear)
target_sync:      every 1000 gradient steps
n_epochs:         3 passes through DQN-train (~180k bars × 3 = 540k steps)
warmup_steps:     5000
loss:             Huber on n-step Bellman target
replay sampling:  prioritized (PER)
early stopping:   monitor DQN-val Sharpe every 5000 steps; stop after 25k no-improvement; save best
```

---

## File structure

```
models/
  vol_v4.py                  ← LightGBM ATR-30, fit on Vol-train (bars 1,440–101,440)
  direction_dl_v4.py         ← 4 CNN-LSTMs, fit on Dir-train (bars 1,440–91,440),
                                early-stopped on Dir-holdout (91,440–101,440)
  regime_cusum_v4.py         ← CUSUM+Hurst, thresholds fit on Vol-train, labels on full
  dqn_state.py               ← Phase 1C standardize + Phase 2D state arrays (50-dim)
  dqn_selector.py            ← (Phase 3) network + training loop
backtest/
  preds.py                   ← (existing) cached vol/dir/regime predictions for older v3 backtest
  dqn_eval.py                ← (Phase 4) test evaluation
cache/
  ── upstream models (Phase 1) ──
  btc_lgbm_atr_30_v4.txt             ← vol model
  btc_pred_vol_v4.npz                ← atr + rank, bars [1,440, 384,614)
  btc_cnn2s_dir_{up,down}_{60,100}_v4.keras  ← 4 direction CNN-LSTM models
  btc_pred_dir_{up,down}_{60,100}_v4.npz     ← direction probabilities, bars [1,440, 384,614)
  btc_regime_cusum_v4.parquet        ← 5-state labels, bars [1,440, 384,614)
  btc_regime_cusum_v4_thresholds.json
  btc_dqn_standardize_v5.json        ← median + IQR per feature, fit on Vol-train
  ── DQN state (Phase 2) ──
  btc_dqn_state_train.npz            ← 180,000 × 50-dim state + 10-dim valid_actions
  btc_dqn_state_val.npz              ← 50,867 × 50
  btc_dqn_state_test.npz             ← 52,307 × 50
  ── DQN training & eval (Phase 3+) ──
  btc_dqn_policy.pt                  ← trained DQN weights (best DQN-val Sharpe)
  btc_dqn_results.parquet            ← per-bar action + per-trade PnL on test
```

---

## Implementation phases

Five sequential phases. Each phase has a **gate** — a check that must pass before starting the next phase. If a gate fails, fix-in-place rather than rolling forward.

```
  Phase 1            Phase 2          Phase 3          Phase 4         Phase 5
 ───────────       ───────────      ───────────      ───────────     ───────────
 Upstream    ───▶  State arrays ──▶ DQN training ──▶ Locked test ──▶ Walk-forward
 models                                              evaluation     robustness
 (A,B,C)            (D)              (E)              (F)             (G)
 ~6 min            ~1 min           ~30 min train    ~5 min          ~30 min
                                    (~3 d to code)
```

---

### Phase 1 — Upstream models & standardization (~10 min runtime)

**Goal:** produce all per-bar inputs the DQN consumes that aren't the network itself: vol predictions, regime labels, direction probabilities (used inside strategy signals only), and the standardization statistics.

**Bar-chunk legend** (raw 384,614-bar parquet, no `clean_mask`):

| Chunk | Bar range | Bar count | Approx. dates | Used as |
|---|---|---|---|---|
| Warmup | `[0, 1,440)` | 1,440 | Jul 4 → ~Jul 5 2025 | dropped (NaN window) |
| Dir-train | `[1,440, 91,440)` | 90,000 | ~Jul 5 → ~Sep 12 2025 | CNN-LSTM training |
| Dir-holdout | `[91,440, 101,440)` | 10,000 | ~Sep 12 → ~Sep 19 2025 | CNN-LSTM early-stop holdout |
| Vol-train | `[1,440, 101,440)` | 100,000 | ~Jul 5 → ~Sep 19 2025 | LightGBM vol fit + CUSUM threshold fit + standardize stats |
| RL period | `[101,440, 384,614)` | 283,174 | ~Sep 19 2025 → Apr 25 2026 | DQN train + val + test (split below) |
| └ DQN-train | `[101,440, 281,440)` | 180,000 | ~Sep 19 2025 → ~Feb 5 2026 | DQN online training |
| └ DQN-val | `[281,440, 332,307)` | 50,867 | ~Feb 5 → ~Mar 16 2026 | DQN early-stop / hyperparam selection |
| └ DQN-test | `[332,307, 384,614)` | 52,307 | ~Mar 16 → Apr 25 2026 | locked, single-shot evaluation |

Note: Dir-train and Dir-holdout together equal Vol-train; the CNN-LSTM uses the last 10k of vol-train as early-stop holdout while the LightGBM vol model fits on all 100k. This means vol-rank for bars `[91,440, 101,440)` (the dir-holdout range) is in-sample relative to the vol model — a minor leak that inflates dir-holdout AUC slightly. The DQN-test evaluation (bars 332,307+) uses fully OOS vol rank, so the locked metric is honest.

**Steps**

- **A. Vol model v4** (~5 sec runtime, ~3 sec fit)
  - Load `btc_features_assembled.parquet` (384,614 rows × 191 features, no `clean_mask`).
  - Drop first 1,440 warmup rows.
  - Fit `StandardScaler` + LightGBM ATR-30 on **Vol-train (`[1,440, 101,440)`)**, with last 5% used as early-stop holdout inside training.
  - Predict on **RL period (`[101,440, 384,614)`)**.
  - Save model `cache/btc_lgbm_atr_30_v4.txt` and predictions `cache/btc_pred_vol_v4.npz` (atr + rank, full bars `[1,440, 384,614)`).

- **A.5. Direction CNN-LSTM v4** (~4 min runtime, 4 models × ~40s each)
  - Reuse `models.direction_dl.SEQ_FEATURES` (30 channels) + vol-rank as 31st channel ("two-stage" architecture, matching old `cnn2s_*` topology).
  - Fit `StandardScaler` on **Dir-train (`[1,440, 91,440)`)**.
  - Train CNN-LSTM (Conv1D → GRU → Dense, SEQ_LEN=30) on **Dir-train**, early-stop on **Dir-holdout (`[91,440, 101,440)`)**, `val_auc` patience=5.
  - Train 4 separate models for `up_60`, `down_60`, `up_100`, `down_100` (label = `>0.8%` move within H bars).
  - Predict on **bars `[1,440, 384,614)`** (full 383k arrays cached for downstream consumption; only RL portion is OOS).
  - Save models `cache/btc_cnn2s_dir_{up,down}_{60,100}_v4.keras` and predictions `cache/btc_pred_dir_{up,down}_{60,100}_v4.npz`.
  - Direction predictions are **used by strategy signal logic (S1, S4, S6)** but are **not** entries in the 50-dim DQN state vector.

- **B. CUSUM regime v4** (~0.4 sec runtime)
  - Compute Hurst (60-bar) and CUSUM (60-bar) on raw 1-bar log-returns; bb_width comes from features parquet.
  - Fit 5 percentile thresholds on **Vol-train (`[1,440, 101,440)`)**: CUSUM+ p75, CUSUM- p25, Hurst p65/p35, bb_width p30. Use `np.nanpercentile` (rolling-window NaN at sequence head).
  - Label all bars in **`[1,440, 384,614)`**.
  - Save `cache/btc_regime_cusum_v4.parquet` and `cache/btc_regime_cusum_v4_thresholds.json`.

- **C. Standardization stats** (rolled into Phase 2 step D, ~15 sec)
  - Compute `log_return` from `perp_ask_price` (1-bar log diff) and `log_volume_z` raw values from `perp_minute_volume` (log first).
  - For each numerical state feature (static + windowed), compute median + IQR on **Vol-train (`[1,440, 101,440)`)**.
    - For windowed features: use the lag-0 series only (single statistic shared across all 6 lag positions).
  - Cache as `cache/btc_dqn_standardize_v5.json`.

**Deliverables (Phase 1)**

- `cache/btc_lgbm_atr_30_v4.txt` — vol model fit on Vol-train
- `cache/btc_pred_vol_v4.npz` — atr + rank, bars `[1,440, 384,614)`
- `cache/btc_cnn2s_dir_{up,down}_{60,100}_v4.keras` — 4 CNN-LSTM models fit on Dir-train, early-stopped on Dir-holdout
- `cache/btc_pred_dir_{up,down}_{60,100}_v4.npz` — direction probabilities, bars `[1,440, 384,614)`
- `cache/btc_regime_cusum_v4.parquet` — regime labels for bars `[1,440, 384,614)`, thresholds fit on Vol-train
- `cache/btc_regime_cusum_v4_thresholds.json` — fitted thresholds
- `cache/btc_dqn_standardize_v5.json` — median + IQR per state feature, fit on Vol-train

**Gate to Phase 2**

- Vol RL-period Spearman ≥ 0.65 (informational; ensures vol model didn't degrade vs. v3).
- All 4 direction models RL-period AUC > 0.55 (binary direction has signal).
- CUSUM Kruskal-Wallis on DQN-train forward returns: p < 0.01.
- All standardize keys present in JSON; no NaN/inf in median or IQR.

---

### Phase 2 — Per-bar state arrays (~1 min runtime)

**Goal:** materialize the 50-dim state and 10-dim valid-actions mask for every bar in DQN-train/val/test as cached npz, so training loops do **zero feature computation**.

**Step**

- **D. Build state arrays**
  - Compute the 5 windowed source series for all bars 1.4k–384.6k:
    - `log_return`, `taker_net_60_z`, `ofi_perp_10`, `vwap_dev_240`, `log_volume_z`.
  - For each target bar t in 101.4k–384.6k:
    - Static block (20 dims): vol_pred, atr_pred_norm, regime one-hot, signal flags, bb_width, fund_rate_z, recent perf.
    - Windowed block (30 dims): for each of 5 features, gather 6 lag positions `[60, 30, 15, 5, 1, 0]`. Pad with 0 if any lag underflows (rare past warmup).
    - Apply standardization (median, IQR) per feature.
    - Concatenate → 50-dim float32.
  - Build 10-dim binary `valid_actions` mask per bar.
  - Slice to train/val/test boundaries; cache to three npz files.
  - Vectorized: use `np.lib.stride_tricks.sliding_window_view` with 6 lag offsets — single pass, ~0.5s for 283k bars.

**Deliverables**

- `cache/btc_dqn_state_train.npz`  (~180k × 50 + 180k × 10)
- `cache/btc_dqn_state_val.npz`    (~50.9k × 50 + …)
- `cache/btc_dqn_state_test.npz`   (~52.3k × 50 + …)

**Gate to Phase 3**

- Per-split arrays have correct shape and dtype.
- Each numeric column post-standardization: |mean| < 0.5, std ∈ [0.5, 2.0] on DQN-train (sanity, not exact normality — IQR-scaled).
- Valid-actions mask: action 0 (NO_TRADE) always 1; non-zero count of bars with ≥1 strategy active matches expectation from current backtest (~30–60% of bars on DQN-train).

---

### Phase 3 — DQN training (~30 min runtime, ~3 days to implement)

**Goal:** train the policy; pick best by DQN-val Sharpe.

**Step**

- **E. Training loop**
  - MLP (50→64→32→10), replay buffer (PER, 80k), n-step Bellman, Huber loss, action masking, ε-greedy.
  - **Single-trade simulator** wrapping `backtest.engine.run`. **Critical:** must execute one trade in <1 ms; calling the full engine per action would be glacial.
  - Track: train loss, DQN-val Sharpe @5k steps, action distribution per fold of training.
  - Early stopping: stop after 25k steps with no DQN-val Sharpe improvement; save best weights.

**Deliverables**

- `models/dqn_state.py`, `models/dqn_selector.py`, `backtest/dqn_eval.py` (single-trade simulator lives here or `backtest/single_trade.py`).
- `cache/btc_dqn_policy.pt` (best-val weights).
- Training log with loss curve, DQN-val Sharpe trajectory, action histogram over time.

**Gate to Phase 4**

- Best DQN-val Sharpe > 0.5 (informational; if <0, the architecture or reward shaping is broken — debug before evaluating on test).
- Action distribution: action 0 frequency 50–95% (i.e. policy actually trades sometimes).
- Loss curve: not diverging; Q-value magnitudes bounded (max |Q| < 10).

**Stop condition**

If best DQN-val Sharpe < 0 after one full training run with reasonable hyperparameters → stop here, do not run Phase 4. Either: (a) re-examine reward shaping & state vector, or (b) accept that strategy gating isn't learnable from this representation and pivot back to signal redesign.

---

### Phase 4 — Locked test evaluation (~5 min)

**Goal:** single-shot, no-tuning evaluation on DQN-test (Mar 16 → Apr 25 2026).

**Step**

- **F. Test evaluation**
  - Load best-val DQN weights.
  - Run policy on DQN-test, simulate trades through engine.
  - Report: Sharpe, Calmar, MaxDD, action distribution, per-strategy attribution.

**Deliverables**

- `cache/btc_dqn_results.parquet` (per-bar action + per-trade PnL).
- Summary table comparable to existing `backtest/run.py` output.

**Gate to Phase 5**

- DQN-test Sharpe > 1.0 (acceptance criterion). If failed, walk-forward is unlikely to redeem the result; consider Phase 5 a diagnostic only.
- DQN-test Sharpe ≥ best CUSUM-gate single strategy (S4 +3.13). If below, the DQN is a downgrade — note this clearly and proceed to Phase 5 to confirm it's not a single-fold artifact.

---

### Phase 5 — Walk-forward robustness (~30 min)

**Goal:** confirm Phase 4 result isn't a lucky test split.

**Step**

- **G. Walk-forward**
  - 6 overlapping ~50k-bar folds within RL period (101.4k–384.6k).
  - Refit DQN per fold; report per-fold Sharpe distribution.

**Deliverables**

- Per-fold Sharpe table, mean ± std.
- Decision: ship policy / iterate state design / abandon RL gating.

**Final acceptance**

- ≥4 of 6 folds with Sharpe > 0.5 (required), ≥5 of 6 (stretch).

---

## Time budget summary

| Phase | Runtime | Implementation effort | Gate |
|---|---|---|---|
| 1 — upstream (vol + dir + regime + standardize) | ~10 min (vol 5s, dir 4 min, regime 0.4s, standardize 15s) | small (each module ~1 file) | RL-period AUC > 0.55 (dir), Spearman ≥ 0.65 (vol), KW p < 0.01 (regime) |
| 2 — state arrays | ~1 min | ~half day (vectorized window builder is the only new code) | shape & sanity stats |
| 3 — DQN training | ~30 min | **~3 days** (network + buffer + n-step + single-trade sim) | val Sharpe > 0.5 |
| 4 — test eval | ~5 min | ~1 hour | test Sharpe > 1.0 |
| 5 — walk-forward | ~30 min | ~1 hour (loop wrapping Phase 3) | ≥4/6 folds > 0.5 |

**End-to-end clean run:** ~70 min wall-clock once code is stable. **Total dev effort:** ~3 working days.

---

## Acceptance criteria

| Metric | Required | Stretch |
|---|---|---|
| DQN-test Sharpe (single eval) | > 1.0 | > 2.0 |
| Walk-forward folds Sharpe > 0.5 | ≥ 4/6 | ≥ 5/6 |
| Test Sharpe ≥ best CUSUM-gate single strategy | yes (vs S4 +3.13) | beat by ≥ 0.5 |
| Action distribution | not 100% one strategy | balanced |
| Action 0 (no-trade) frequency | < 95% | 50–80% |

---

## Pitfalls

1. **Don't peek at DQN-test until final evaluation.** Hyperparameter tuning on DQN-val only.
2. **Hard mask Q-values** for non-firing strategies. No soft penalty.
3. **Watch buffer composition.** Most transitions have r=0. Use prioritized replay so non-zero rewards aren't drowned.
4. **Use Huber loss** for stable Q-targets given high reward variance.
5. **Don't reward-shape no-trade.** Reward is exactly 0 when action=0.
6. **Across-gap features.** A 60-bar lookback may span 4 hours of real time when there's a 1-hour gap inside it. The DQN can learn this, but: standardization stats from vol-train should be robust (use IQR not std) so a few outliers don't poison the scaling.
7. **Distribution shift train→test.** Vol-train is Jul–Sep 2025. DQN-test is Mar–Apr 2026. The Nov 2025 outage is in DQN-train (~Sep 20 → ~Feb 5). Walk-forward will surface any shift sensitivity.

---

---

## Overall architecture diagram (v5)

```
═══════════════════════════════════════════════════════════════════════════════
                    DQN STRATEGY SELECTOR — v5 ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                       RAW DATA  (384,614 bars, no clean_mask)                │
│                       Jul 4 2025 → Apr 25 2026  ·  BTC 1-min                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            FEATURE ASSEMBLY  (existing, 191 cols, index-based rolling)       │
│                       NaN only in first 1,440 warmup bars                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │  drop bars 0–1,439 (warmup)
                                      ▼
                          ┌─────────────────────────┐
                          │  USABLE: 383,174 bars   │
                          │  bar 1,440 → 384,614    │
                          └────────────┬────────────┘
                                       │
        ┌──────────────────────────────┼─────────────────────────────────┐
        ▼                              ▼                                 ▼
   ┌────────┐                ┌──────────────────┐              ┌─────────────────┐
   │ Vol-tr │   100k bars    │   DQN-train      │   180k bars  │ DQN-val/test    │
   │ ~77 d  │                │   ~138 days      │              │ ~40 d + ~40 d   │
   └───┬────┘                └────────┬─────────┘              └────────┬────────┘
       │                              │                                 │
       ▼                              ▼                                 ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║  ── upstream models, fitted on Vol-train (bars 1.4k–101.4k) ──       ║
       ║                                                                       ║
       ║  ┌──────────────────────────┐    ┌─────────────────────────────────┐ ║
       ║  │  Vol LightGBM (atr_30)   │    │  CUSUM regime classifier        │ ║
       ║  │  191 feats → vol_pred    │    │  Hurst + CUSUM + bb_width →     │ ║
       ║  │                          │    │  5 states                       │ ║
       ║  │  Cached: btc_pred_vol_v4 │    │  Cached: btc_regime_cusum_v4    │ ║
       ║  └──────────────────────────┘    └─────────────────────────────────┘ ║
       ║                                                                       ║
       ║  ── strategy signals + windowed series (per bar) ──                   ║
       ║  9 strategy signals (existing rules, ∈ {-1, 0, +1})                   ║
       ║  5 windowed series:  log_return, taker_net_60_z, ofi_perp_10,         ║
       ║                       vwap_dev_240, log_volume_z                      ║
       ║                                                                       ║
       ║  Standardization stats (median, IQR) computed on Vol-train ONLY      ║
       ╚══════════════════════════════════════════════════════════════════════╝
                                       │
                                       ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║                    PER-BAR STATE VECTOR  (50 dim)                     ║
       ║                                                                       ║
       ║  ┌── Static block (20) ─────────────────────────────────────────┐    ║
       ║  │  Vol         vol_pred, atr_pred_norm                      2  │    ║
       ║  │  Regime      one-hot 5 states                             5  │    ║
       ║  │  Sig active  s1..s12  (9 strategies, {-1,0,+1})           9  │    ║
       ║  │  Static ctx  bb_width, fund_rate_z                        2  │    ║
       ║  │  Recent perf last_pnl, current_dd                         2  │    ║
       ║  └──────────────────────────────────────────────────────────────┘    ║
       ║                                                                       ║
       ║  ┌── Windowed block (30) ─ lags [60, 30, 15, 5, 1, 0] each ──┐       ║
       ║  │  log_return       × 6 lags                              6  │       ║
       ║  │  taker_net_60_z   × 6 lags                              6  │       ║
       ║  │  ofi_perp_10      × 6 lags                              6  │       ║
       ║  │  vwap_dev_240     × 6 lags                              6  │       ║
       ║  │  log_volume_z     × 6 lags                              6  │       ║
       ║  └──────────────────────────────────────────────────────────────┘    ║
       ║                                                                       ║
       ║  All numerical → standardize:  (x − median) / IQR                     ║
       ║  Same stats applied across all lag positions of a windowed feature    ║
       ║                                                                       ║
       ║                            + ACTION MASK (10 dim)                     ║
       ║   valid_actions = {0:NO_TRADE} ∪ {k : strategy_k_signal != 0}        ║
       ╚══════════════════════════════════════════════════════════════════════╝
                                       │
                                       ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║                          DQN  (~5,674 params)                         ║
       ║                                                                       ║
       ║          state (50)                     Q-values (10)                 ║
       ║             │                              ▲                          ║
       ║             ▼                              │                          ║
       ║        ┌─────────┐    ┌─────────┐    ┌─────────┐                     ║
       ║        │ Linear  │───▶│ Linear  │───▶│ Linear  │                     ║
       ║        │ 50→64   │    │ 64→32   │    │ 32→10   │                     ║
       ║        │ ReLU    │    │ ReLU    │    │ (logits)│                     ║
       ║        └─────────┘    └─────────┘    └─────────┘                     ║
       ║                                              │                        ║
       ║                                              ▼                        ║
       ║                    Q[~valid_actions] = -1e9                          ║
       ║                              │                                        ║
       ║                              ▼                                        ║
       ║              ε-greedy:  argmax Q   (or random masked)                 ║
       ╚══════════════════════════════════════════════════════════════════════╝
                                       │
                                       │   action a ∈ {0..9}
                                       ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║                  EXECUTION & REWARD  (existing engine)                ║
       ║                                                                       ║
       ║   if a == 0:   advance 1 bar, reward = 0                              ║
       ║                                                                       ║
       ║   if a != 0:   strategy_a fires → enter trade                         ║
       ║                  EXECUTION_CONFIG[strategy_a] handles TP/SL/BE/sizing ║
       ║                  trade runs τ bars in backtest/engine.py              ║
       ║                                                                       ║
       ║                  on close:                                            ║
       ║                    r = pnl_pct − 2·TAKER_FEE  (Huber)                 ║
       ║                    advance τ bars, next state at t+τ                  ║
       ╚══════════════════════════════════════════════════════════════════════╝
                                       │
                                       │   transition (s, a, r-seq, s')
                                       ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║          REPLAY BUFFER  (PER, α=0.6, β=0.4→1.0, size 80k)             ║
       ║                                                                       ║
       ║   sample batch (128) → n-step Bellman target                         ║
       ║       G   = Σ γⁱ rᵢ                                                  ║
       ║       tgt = G + γⁿ · max Q_target(s')[valid']                        ║
       ║       loss= Huber(Q(s)[a], tgt)  weighted by IS                       ║
       ║                                                                       ║
       ║   target_net synced every 1000 grad steps                             ║
       ║   ε:  1.0 → 0.05 over 80k steps                                       ║
       ║   Adam lr 1e-3 · 3 epochs through DQN-train                           ║
       ║   early stop: best DQN-val Sharpe @ 5k step intervals                 ║
       ╚══════════════════════════════════════════════════════════════════════╝
                                       │
                                       │   final policy
                                       ▼
       ╔══════════════════════════════════════════════════════════════════════╗
       ║                EVALUATION  (DQN-test, locked, single shot)            ║
       ║                                                                       ║
       ║   Run trained policy over 52k bars (Mar 16 → Apr 25 2026)             ║
       ║   Report:  Sharpe · Calmar · MaxDD · action distribution              ║
       ║            per-strategy attribution · forced-NO_TRADE %               ║
       ║                                                                       ║
       ║   Baseline to beat:  S4 with CUSUM gate, test Sharpe = +3.13         ║
       ║                                                                       ║
       ║   Walk-forward: 6 folds × refit · per-fold Sharpe distribution        ║
       ╚══════════════════════════════════════════════════════════════════════╝
```

---

## What this experiment will resolve

A passing DQN means: **strategy gating is the bottleneck**, and (vol + regime + signal-activity) carries enough information to learn the gate.

A failing DQN means **the strategies themselves are the bottleneck** — no gate over current 9 strategies has consistent edge. Stop investing in RL gating; rebuild signal logic.

---

## Out of scope

- Continuous action / sizing
- Multi-asset state (BTC only)
- Recurrent network (LSTM-DQN)
- Imitation pretraining from CUSUM gate
- Distributional / dueling / Rainbow DQN
- Retraining direction CNN-LSTM (cached preds used by S1/S4/S6 internally; not a DQN input)
- Re-engineering features to be gap-aware (we explicitly opt out — bars are a sequence)
