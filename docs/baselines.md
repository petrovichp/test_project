# Baselines — reference systems for all future experiments

> Refreshed 2026-05-11 after Phase Z3 Step 4: **`VOTE5_v8_H256_DD` is the new primary baseline** (12-action: S11_Basis + S13_OBDiv added). Reproducible end-to-end from the data files in `cache/` and the commands below. All future experiments report deltas vs `VOTE5_v8_H256_DD` (strongest) and may include the others for context.

## Quick comparison

| Metric | ⭐ **VOTE5_v8_H256_DD** | **VOTE5_v7basis_H256_DD** | **VOTE5_H256_DD** | **VOTE5** | **VOTE5_H256** | **VOTE5_DD** |
|---|---|---|---|---|---|---|
| Type | h=256 + DD, **12-action** (S11/S13 added) | h=256 + DD, basis+funding state | h=256 + Double_Dueling | h=64 plurality | h=256 plurality | h=64 + DD plurality |
| State version | v8_s11s13 (52-dim) | v7_basis (55-dim) | v5 (50-dim) | v5 | v5 | v5 |
| Constituents | 5 seeds: 42,7,123,0,99 | same | same | same | same | same |
| **Walk-forward mean Sharpe** | **+12.07** | +11.66 | +11.05 | +10.40 | +11.86 | +6.80 |
| Walk-forward folds positive | **6/6** | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Fold 6 Sharpe | +4.44 | +2.31 | **+8.23** | +5.20 | +0.41 | +4.58 |
| DQN-val Sharpe | **+6.67** | +6.05 | +3.21 | +3.53 | +3.32 | +6.12 |
| DQN-test Sharpe | +4.44 | +2.90 | **+9.01** | +4.19 | +1.21 | +5.91 |
| Trades (full RL period) | 1,416 | 1,182 | 1,372 | 1,122 | — | 1,437 |
| Profile | **best WF + best val + 6/6 folds** | val-robustness alternative | best test + fold-6 | balanced h=64 | aggregate WF (weak fold-6) | val regularization (h=64) |

**All baselines beat BTC buy-and-hold on val (+7.2%) and test (+8.6%).**

### Headline: VOTE5_v8_H256_DD

`VOTE5_v8_H256_DD` (Phase Z3 Step 4, 2026-05-11) is the **primary deployable system**. It stacks three independently-validated improvements on top of the original VOTE5 baseline:

- **H256 capacity** (from Z1.1): lifts WF aggregate
- **Double_Dueling regularization** (from Z1.1): recovers fold-6 robustness
- **S11_Basis + S13_OBDiv action expansion** (from Z3 Step 4): unique signal types not in original 9 strategies

**Reproduction**:
```bash
# Build expanded state cache (signals + valid_actions + state vector)
python3 -m models.build_state_v8

# Train 5 seeds
for SEED in 42 7 123 0 99; do
  python3 -m models.dqn_selector btc \
    --tag VOTE5_v8_H256_DD_seed${SEED} \
    --state-version v8_s11s13 \
    --hidden 256 --algo double_dueling \
    --fee 0.0 --trade-penalty 0.001 --seed ${SEED}
done

# Evaluate
python3 -m models.eval_z2_z3
```

Wall time: ~10 min total (1.5 min state build + 5 × ~2 min training).

---

### Previously promoted: VOTE5_H256_DD

`VOTE5_H256_DD` (Phase Z1.1, 2026-05-10) was the primary baseline before Step 4. It stacked two improvements:

- **H256 capacity**: lifts WF mean Sharpe (+11.86 alone vs +10.40 vanilla)
- **Double_Dueling regularization**: recovers fold-6 robustness (H256 alone fold-6 +0.41 → stacked **+8.23**)

The combination wins on every metric except val Sharpe (where DD-alone retains a small edge):

| | beats vanilla VOTE5 by | beats H256 alone by | beats DD alone by |
|---|---:|---:|---:|
| WF | +0.65 | −0.81 | +4.25 |
| test | **+4.82** | **+7.80** | +3.10 |
| fold-6 | **+3.03** | **+7.82** | +3.65 |

**Test +9.01 is the highest test Sharpe in the entire project.** Source: [docs/z1_results.md](z1_results.md).

### Reproduction

```bash
for SEED in 42 7 123 0 99; do
  python3 -m models.dqn_selector btc \
    --tag VOTE5_H256_DD_seed${SEED} \
    --hidden 256 --algo double_dueling \
    --fee 0.0 --trade-penalty 0.001 --seed ${SEED}
done
# Then evaluate as K=5 plurality:
python3 -m models.eval_z1_full
```

Wall time: 5 × ~2 min training. Network: `DuelingDQN(50, 10, hidden=256)` with Double DQN target update.

### ~~Dropped: BASELINE_VOTE5_H128~~

Previously listed as the "late-period winner". **Exposed as seed-luck in Phase Z1.4**: the original pool's fold-6 +10.70 / test +10.59 collapse to **fold-6 −4.82** / test +5.30 in a disjoint seed pool. Mean across both pools: WF +8.55, fold-6 +2.94. Removed from baseline rotation. See [docs/z1_results.md](z1_results.md) §Z1.4.

### Other baselines retained

- `BASELINE_VOTE5` (h=64 plurality, vanilla DQN): historical primary; still useful as a lightweight reference for state-vector / strategy-space experiments.
- `BASELINE_VOTE5_H256` (h=256 plurality, vanilla): high WF aggregate but weak fold-6; kept for capacity-axis comparisons.
- `BASELINE_VOTE5_DISJOINT` (h=64 plurality, seeds 1/13/25/50/77): structural validation of plurality voting.
- `BASELINE_VOTE5_DD` (h=64 plurality, DD): regularization-axis reference; best val of any K=5 (+6.12).
- `BASELINE_FULL` (single seed=42): strongest single-seed val (+7.30); not deployable solo (variance ±2.17).
- `BASELINE_LEAN` (single seed=42, S6/S7/S8 ablated): regime-alternative.

`BASELINE_VOTE5` is documented separately in [voting_ensemble.md](voting_ensemble.md). The remainder of this doc covers `BASELINE_FULL` and `BASELINE_LEAN` (both single-seed) — `BASELINE_VOTE5` shares the same training spec but uses 5 seeds aggregated by plurality voting.

---

## Common foundation (shared by both baselines)

### Data

All data lives in `cache/` as parquet/npz. Never touch raw CSVs.

| File | Contents |
|---|---|
| `cache/okx_btcusdt_spotpepr_20260425_meta.parquet` | 384,614 bars × 29 columns of market/microstructure data (BTC, 1-min) |
| `cache/btc_features_v4.parquet` | 191 engineered features (orderbook, price, volume, market, microstructure) |
| `cache/btc_pred_vol_v4.npz` | LightGBM ATR-30 predictions + `atr_train_median = 0.001834` |
| `cache/btc_pred_dir_v4.npz` | 4 CNN-LSTM direction predictions (P_up/P_down × 60/100 horizons) |
| `cache/btc_dqn_state_train.npz` | 50-dim states, 180,000 bars (DQN-train) |
| `cache/btc_dqn_state_val.npz` | 50,867 bars (DQN-val) |
| `cache/btc_dqn_state_test.npz` | 52,307 bars (DQN-test, locked) |

### Splits ([docs/data_splits.md](data_splits.md) is canonical)

| Split | Bar range | Count | Date range | Span |
|---|---|---|---|---|
| Vol-train (predictive) | [1,440, 101,440) | 100,000 | 2025-07-05 → 2025-09-19 | 76 d |
| Dir-train | [1,440, 91,440) | 90,000 | 2025-07-05 → 2025-09-12 | 69 d |
| Dir-holdout (early-stop) | [91,440, 101,440) | 10,000 | 2025-09-12 → 2025-09-19 | 6 d |
| **DQN-train (RL)** | [101,440, 281,440) | 180,000 | 2025-09-19 → 2026-02-12 | 146 d |
| **DQN-val (RL early-stop)** | [281,440, 332,307) | 50,867 | 2026-02-12 → 2026-03-20 | 35 d |
| **DQN-test (RL locked)** | [332,307, 384,614) | 52,307 | 2026-03-20 → 2026-04-25 | 36 d |

Walk-forward uses 6 contiguous folds of ~47,195 bars each across the full RL period (`[101,440, 384,614)`).

### State (50-dim) — same for both baselines

Built by [models/dqn_state.py](../models/dqn_state.py). Per-bar state vector:

```
 [0..3]   direction predictions (up/down × 60/100)
 [4..6]   vol prediction, vol percentile, ATR-z
 [7..15]  9 binary signal flags from strategies S1..S12
 [16]     hour-of-day sin
 [17]     hour-of-day cos
 [18]     equity (1.0-anchored)
 [19]     drawdown from peak (clipped [-20, 0])
 [20]     last-trade PnL %
 [21..29] 9 per-strategy rank features
 [30..49] 20 orderbook + microstructure features
```

`valid_actions` mask (10-dim bool): NO_TRADE always valid; strategy k valid if `signals[t, k]` is True. The mask is checked before any argmax/max — no leakage of disabled actions.

### Network architecture

```python
DQN(state_dim=50, n_actions=10, hidden=64)
    fc1: Linear(50, 64) + ReLU
    fc2: Linear(64, 32) + ReLU
    fc3: Linear(32, 10)          # raw Q-values
```

[models/dqn_network.py](../models/dqn_network.py). 5,674 params total. No BatchNorm, no Dropout. Kaiming-uniform default init. Action masking applied AFTER forward via `q.masked_fill(~valid, -1e9)`.

### Training scheme — hyperparameters (identical for both baselines)

| Param | Value | Source |
|---|---|---|
| optimizer | Adam, lr 1e-3 | `LR = 1e-3` in [models/dqn_selector.py:38](../models/dqn_selector.py#L38) |
| discount γ | 0.99 | `GAMMA = 0.99` |
| batch size | 128 | `BATCH_SIZE = 128` |
| replay buffer | 80,000 transitions | `BUFFER_SIZE = 80_000` |
| warmup (random policy) | 5,000 transitions | `WARMUP_STEPS = 5_000` |
| total grad steps | 200,000 (early-stop active) | `TOTAL_GRAD_STEPS = 200_000` |
| early-stop patience | 25,000 grad steps | `EARLY_STOP_PATIENCE = 25_000` |
| rollout refresh | M=2,000 transitions every 500 grad steps | `REFRESH_M`, `REFRESH_EVERY` |
| target sync | every 1,000 grad steps (hard copy) | `TARGET_SYNC_EVERY = 1_000` |
| validation | every 5,000 grad steps | `VAL_EVERY = 5_000` |
| ε schedule | 1.0 → 0.05 linearly over 80,000 steps | `EPS_DECAY_STEPS = 80_000` |
| PER α / β | α=0.6 fixed; β: 0.4 → 1.0 over training | `PER_ALPHA`, `PER_BETA_*` |
| Huber δ | 1.0 | `HUBER_DELTA = 1.0` |
| grad clip | 10.0 | `GRAD_CLIP = 10.0` |
| reward scale (buffer-only) | ×100 | `REWARD_SCALE = 100.0` |
| stratified PER | True (50/50 NO_TRADE vs trade per batch) | `USE_STRATIFIED_PER = True` |
| n-step Bellman | 1-step (next-bar target) | rollout `step` is bar-by-bar |
| seed | 42 | `--seed 42` (default) |

### Reward / episode logic (identical for both)

Episodes are bar-by-bar, equity-based:

- At each bar `t`, the policy emits action `a` from the masked space.
- `a == NO_TRADE`: equity unchanged, reward = 0; advance one bar.
- `a == strategy k`: a synthetic trade is opened with the strategy's entry/exit/sizing config (see Execution config below). The trade runs until TP/SL/breakeven/trail/time-stop fires. Reward = trade PnL fraction. Equity is multiplied by (1 + PnL). Cursor advances to bar after exit.
- `--trade-penalty 0.001` adds a fixed −0.001 reward per trade entry (in buffer only — eval uses raw PnL).
- `--fee 0.0` — fee-free training. **The fee=0 setting is the assumption that maker-only execution is achievable** (see [docs/next_steps.md](next_steps.md) Path X). All Sharpe figures are at fee=0.

### Exit logic — rule-based ([execution/config.py](../execution/config.py))

Per-strategy ATR-scaled TP/SL with breakeven and conditional trailing:

| Strategy | base TP | base SL | Breakeven | Trail after BE | Time-stop |
|---|---|---|---|---|---|
| S1_VolDir (trend) | 2.0% | 0.7% | 0.5% | yes | none |
| S2_Funding (MR) | 0.8% | 0.5% | 0.3% | no | 60 bars |
| S3_BBRevert (MR) | 0.8% | 0.4% | 0.2% | no | 30 bars |
| S4_MACDTrend (trend) | 2.5% | 0.8% | 0.6% | yes | none |
| S6_TwoSignal (trend) | 2.5% | 0.8% | 0.5% | yes | none |
| S7_OIDiverg (MR-ish) | 1.0% | 0.5% | 0.3% | no | 45 bars |
| S8_TakerFlow (trend) | 1.5% | 0.6% | 0.4% | yes | none |
| S10_Squeeze (trend) | 3.0% | 0.8% | 0.8% | yes | 120 bars |
| S12_VWAPVol | 1.5% | 0.6% | 0.3% | no | 30 bars |

TP/SL scale linearly with `predicted_atr / atr_train_median`. Sizing is volatility-scaled: `target_risk_pct ÷ atr_pred` notional fraction.

---

## BASELINE_FULL — A2 with all 9 strategies

### Reproduction

```bash
# Prereq: state arrays already cached from prior pipeline runs.
# Train:
python3 -m models.dqn_selector btc \
        --tag BASELINE_FULL \
        --fee 0.0 --trade-penalty 0.001 --seed 42

# Walk-forward (6 folds):
python3 -m models.group_c2_walkforward \
        --policy-tag BASELINE_FULL --no-b5 \
        --out-tag BASELINE_FULL
```

### Spec

| Field | Value |
|---|---|
| **Action mask** | none (all 10 actions valid where signals fire) |
| **Effective action space** | NO_TRADE + S1, S2, S3, S4, S6, S7, S8, S10, S12 |
| **Train command** | `python3 -m models.dqn_selector btc --tag BASELINE_FULL --fee 0.0 --trade-penalty 0.001 --seed 42` |
| **Equivalent prior tag** | `A2` |
| **Policy file** | `cache/btc_dqn_policy_BASELINE_FULL.pt` (md5: `92395edb…`) |
| **Training history** | `cache/btc_dqn_train_history_BASELINE_FULL.json` |
| **Best DQN-val Sharpe** | +7.295 at grad-step 65,000 |
| **Total grad steps** | 200,000 (no early stop) |
| **Wall time** | 144 s |
| **Net params** | 5,674 |

### Performance

| Slice | Sharpe | Equity | Trades | Win % | Max DD |
|---|---|---|---|---|---|
| DQN-val | +7.295 | 1.398 | 251 | 55.0% | −6.3% |
| DQN-test | +3.666 | 1.127 | 185 | 53.0% | — |
| Walk-forward (mean across 6 folds) | **+9.034** | various 1.07× to 2.23× | ~150/fold | 50–65% | various |

Per-fold walk-forward Sharpe: `[13.03, 14.82, 6.29, 9.56, 8.17, 2.33]` — 6/6 folds positive.

Action distribution on val: NO_TRADE 97.6%, S1 0.9%, S6 0.5%, S7 0.4%, S8 0.3%, others <0.3% each. The DQN is highly selective (~2% bar coverage).

---

## BASELINE_LEAN — A2 with S6/S7/S10 ablated

### Reproduction

```bash
python3 -m models.dqn_selector btc \
        --tag BASELINE_LEAN \
        --fee 0.0 --trade-penalty 0.001 --seed 42 \
        --ablate-actions 5,6,8

# Walk-forward (6 folds):
python3 -m models.group_c2_walkforward \
        --policy-tag BASELINE_LEAN \
        --ablate-actions 5,6,8 --no-b5 \
        --out-tag BASELINE_LEAN
```

### Spec

| Field | Value |
|---|---|
| **Action mask** | actions {5, 6, 8} forced invalid (during training AND eval) |
| **Effective action space** | NO_TRADE + S1, S2, S3, S4, S8 + (S12 effectively unused, 0 trades) |
| **Train command** | `python3 -m models.dqn_selector btc --tag BASELINE_LEAN --fee 0.0 --trade-penalty 0.001 --seed 42 --ablate-actions 5,6,8` |
| **Equivalent prior tag** | `A2_no_s6_s7_s10` |
| **Policy file** | `cache/btc_dqn_policy_BASELINE_LEAN.pt` (md5: `db390b88…`) |
| **Training history** | `cache/btc_dqn_train_history_BASELINE_LEAN.json` |
| **Best DQN-val Sharpe** | +4.662 at grad-step 20,000 |
| **Total grad steps** | 45,000 (early-stop) |
| **Wall time** | 85 s |
| **Net params** | 5,674 |

### Performance

| Slice | Sharpe | Equity | Trades | Win % | Max DD |
|---|---|---|---|---|---|
| DQN-val | +4.662 | 1.233 | 224 | 53–60% | — |
| DQN-test | **+5.192** | 1.195 | 181 | 54–62% | — |
| Walk-forward (mean across 6 folds) | +6.756 | various | 123–247/fold | 50–62% | −3% to −13% |

Per-fold walk-forward Sharpe: `[6.36, 9.24, 8.49, 9.28, 4.07, 3.09]` — 6/6 folds positive.

Per-fold direction: trade count higher than FULL (123–247 vs ~150), win rate higher (54–62% vs 50–65%), but per-trade alpha lower → smaller wins, more frequent.

---

## Why two baselines

`BASELINE_LEAN` was found in the audit follow-up tests ([audit_followup_tests.md](audit_followup_tests.md) Test 6b). It under-performs on the WF aggregate but **wins on the locked DQN-test split** and on WF folds 3 and 6. Pattern: it does better in the most recent ~80 days of data (folds 5–6 + DQN-test all sit in early 2026).

Two interpretations, neither falsified yet:
1. *Regime shift hypothesis*: the early-2026 regime favors a smaller action space; FULL over-trades on noise.
2. *Overfitting hypothesis*: FULL is overfit to the early WF folds; LEAN's reduced capacity generalizes better forward in time.

Future experiments compare to **both** baselines. An improvement that beats FULL on aggregate WF but loses to LEAN on test is an improvement only if regime stability holds. An improvement that beats both on both is a real win.

## Files / artefacts (frozen)

| File | Purpose |
|---|---|
| `cache/btc_dqn_policy_BASELINE_FULL.pt` | trained network weights (FULL) |
| `cache/btc_dqn_policy_BASELINE_LEAN.pt` | trained network weights (LEAN) |
| `cache/btc_dqn_train_history_BASELINE_FULL.json` | full training history (history of val Sharpe per checkpoint) |
| `cache/btc_dqn_train_history_BASELINE_LEAN.json` | full training history |
| `cache/btc_groupC2_walkforward_verify_baseline.json` | FULL walk-forward (6 folds) |
| `cache/btc_groupC2_walkforward_retrain_no_s6_s7_s10.json` | LEAN walk-forward (6 folds) |
| [models/dqn_selector.py](../models/dqn_selector.py) | training loop |
| [models/dqn_network.py](../models/dqn_network.py) | DQN class |
| [models/dqn_state.py](../models/dqn_state.py) | state-array builder |
| [models/dqn_rollout.py](../models/dqn_rollout.py) | environment rollout (Numba single-trade simulator) |
| [models/group_c2_walkforward.py](../models/group_c2_walkforward.py) | walk-forward runner |
| [execution/config.py](../execution/config.py) | per-strategy entry/exit/sizing config |

The `A2.pt` and `A2_no_s6_s7_s10.pt` originals are kept in `cache/` for backwards compatibility with prior result JSONs.
