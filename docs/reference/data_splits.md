# Data Splits — Train / Val / Test for All Models

Single source of truth for split boundaries. All values are hard-coded as constants in the model modules and never overridden at runtime.

**Source data:** BTC USDT spot+perp, 1-min cadence, 384,614 bars, 2025-07-04 → 2026-04-25 (~9.7 months).

---

## Sequential split layout

```
bar 0 ─────────────────────────────────────────────────────────────────── 384,614
│Warmup│←── Vol-train ──────→│  ←──── DQN-train ────→│ DQN-val │ DQN-test │
│      │←── Dir-train ──→│DH│
0     1,440             91k 101k                  281k     332k          384k
                                                                          ↑
                                                                    locked test
```

`Dir-train` and `Vol-train` both start at the end of `Warmup` (bar 1,440). The direction model further splits its tail into a 10,000-bar `Dir-holdout` for early-stopping. Vol model uses LightGBM's `early_stopping_rounds` on the tail of `Vol-train`.

---

## Predictive-model splits (Phase 1)

| Split | Bar range | Count | % of total | Date range | Span | Use |
|---|---|---|---|---|---|---|
| Warmup | [0, 1,440) | 1,440 | 0.4% | 2025-07-04 → 2025-07-05 | 1 d | dropped (NaN window for rolling features) |
| **Vol-train** | [1,440, 101,440) | 100,000 | 26.0% | 2025-07-05 → 2025-09-19 | 76 d | **LightGBM vol model fit + CUSUM thresholds + state-standardize stats** |
| **Dir-train** | [1,440, 91,440) | 90,000 | 23.4% | 2025-07-05 → 2025-09-12 | 69 d | **CNN-LSTM direction training (4 models: up/down × 60/100)** |
| **Dir-holdout** | [91,440, 101,440) | 10,000 | 2.6% | 2025-09-12 → 2025-09-19 | 6 d | **CNN-LSTM early-stop** (val_auc patience=5) |

Notes:
- All standardize stats (median + IQR per feature) are fit on Vol-train only and applied via `transform`-only on later splits.
- CUSUM regime thresholds (CUSUM+ p75, CUSUM− p25, Hurst p65/p35, bb_width p30) come from the same Vol-train slice.
- The 6-bar embargo between train and val/test is implicit via the rolling-feature warmup window (`min_periods=full_window`).

**Constants:** [models/dqn_state.py:52-55](../models/dqn_state.py#L52-L55), [models/vol_v4.py](../models/vol_v4.py), [models/direction_dl_v4.py](../models/direction_dl_v4.py), [models/regime_cusum_v4.py](../models/regime_cusum_v4.py)

---

## RL splits (Phase 2 — entry & exit DQN)

| Split | Bar range | Count | % of total | Date range | Span | Use |
|---|---|---|---|---|---|---|
| **DQN-train** | [101,440, 281,440) | 180,000 | 46.8% | 2025-09-19 → 2026-02-12 | 146 d | **Group A entry DQN + Group B exit DQN training** |
| **DQN-val** | [281,440, 332,307) | 50,867 | 13.2% | 2026-02-12 → 2026-03-20 | 35 d | **early-stop + best-checkpoint selection** (the val Sharpe numbers in RESULTS.md) |
| **DQN-test** | [332,307, 384,614) | 52,307 | 13.6% | 2026-03-20 → 2026-04-25 | 36 d | **LOCKED — single-shot eval only** (the test Sharpe numbers in RESULTS.md) |

Notes:
- `DQN-train` starts at the END of `Vol-train` (bar 101,440). No overlap with predictive-model training.
- DQN-val and DQN-test are each ~5 weeks of trading. Test is touched once after val tuning is frozen — never used for selection.
- All Group A / Group B / Group C runs use exactly these splits (no random cross-validation, no shuffling).

**Constants:** [models/dqn_state.py:53-55](../models/dqn_state.py#L53-L55) — `WARMUP=1440`, `VOL_TRAIN_E=101_440`, `DQN_TRAIN_E=281_440`, `DQN_VAL_E=332_307`.

---

## Walk-forward folds (D3, Path 1a, Path 1b diagnostics)

The "RL period" — the full span available for walk-forward stress-testing — is `[101,440, 384,614)`, **283,174 bars** (~217 calendar days). Split into **6 contiguous folds of ~47,195 bars each (~32 days):**

| Fold | Bar range | Count | Date range |
|---|---|---|---|
| fold 1 | [101,440, 148,635) | 47,195 | 2025-09-19 → 2025-10-22 |
| fold 2 | [148,635, 195,830) | 47,195 | 2025-10-22 → 2025-12-15 |
| fold 3 | [195,830, 243,025) | 47,195 | 2025-12-15 → 2026-01-17 |
| fold 4 | [243,025, 290,220) | 47,195 | 2026-01-17 → 2026-02-19 |
| fold 5 | [290,220, 337,415) | 47,195 | 2026-02-19 → 2026-03-23 |
| fold 6 | [337,415, 384,614) | 47,199 | 2026-03-23 → 2026-04-25 |

Date spans differ slightly because the source data has occasional gaps (missing bars don't advance the wall clock). Bar counts are uniform.

**Constants:** [models/walk_forward.py:23-26](../models/walk_forward.py#L23-L26) — `RL_START_REL=100_000`, `RL_END_REL=383_174`, `N_FOLDS=6` (relative-to-WARMUP indexing internally).

---

## Summary in one line per model

| Model | Train | Val (early-stop) | Test (locked) |
|---|---|---|---|
| Vol LightGBM | 100k bars (Jul 5 – Sep 19) | tail of train (early-stopping rounds) | full RL period (OOS Spearman) |
| Direction CNN-LSTM × 4 | 90k bars (Jul 5 – Sep 12) | 10k bars (Sep 12 – Sep 19) | full RL period (OOS AUC) |
| CUSUM regime | thresholds fit on Vol-train | n/a (rule-based labeling) | n/a |
| **DQN entry (Group A)** | **180k bars (Sep 19 – Feb 12)** | **50,867 bars (Feb 12 – Mar 20)** | **52,307 bars (Mar 20 – Apr 25)** |
| **DQN exit (Group B/C)** | same DQN-train | same DQN-val | same DQN-test |

---

## Non-negotiable rules (re-stated)

These are also in [CLAUDE.md](../CLAUDE.md). The split layout exists to enforce them:

- **No leakage**: features use only bars `[t − lookback, t]`. `shift(-n)` for labels only.
- **No random splits**: time-series data is never shuffled. Sequential or walk-forward only.
- **Normalization fit on train only**: scaler median/IQR fitted on Vol-train; transform-only on val/test.
- **Cross-asset features**: lag predictor asset by 1+ bars (prevents lookahead in joint feature pipelines).
- **Rolling windows**: `min_periods=full_window` — NaN early bars, exclude from training samples.
- **Embargo**: implicit via the WARMUP=1,440 bar gap and rolling-feature `min_periods` requirement at every split boundary.
- **Test split is locked**: touched only after val tuning is frozen. The Group A / Group B / Group C numbers reported in [RESULTS.md](../RESULTS.md) are the result of single-shot test evaluation.

---

## How to verify these numbers yourself

```python
import pandas as pd
meta = pd.read_parquet('cache/raw/okx_btcusdt_spotpepr_20260425_meta.parquet')
ts = pd.to_datetime(meta['timestamp'].values, unit='s')

WARMUP, VOL_TRAIN_E, DIR_TRAIN_E = 1_440, 101_440, 91_440
DQN_TRAIN_E, DQN_VAL_E = 281_440, 332_307

for name, a, b in [
    ('Warmup',      0,             WARMUP),
    ('Vol-train',   WARMUP,        VOL_TRAIN_E),
    ('Dir-train',   WARMUP,        DIR_TRAIN_E),
    ('Dir-holdout', DIR_TRAIN_E,   VOL_TRAIN_E),
    ('DQN-train',   VOL_TRAIN_E,   DQN_TRAIN_E),
    ('DQN-val',     DQN_TRAIN_E,   DQN_VAL_E),
    ('DQN-test',    DQN_VAL_E,     len(meta)),
]:
    print(f'{name:<12} [{a:>7,}, {b:>7,})  {b-a:>7,}  '
          f'{ts[a]:%Y-%m-%d} → {ts[b-1]:%Y-%m-%d}')
```
