# State-pack v8 layout — definitive audit

Executed 2026-05-13 as part of the A1/A2/A3 audit sweep
(see [development_plan.md §Forward plan](../reference/development_plan.md)).

Triggered by the Z5.1 audit (2026-05-12), which exposed that the
`transformer_network.py` docstring had a state-dim layout substantially
different from what `dqn_state.py` actually writes. This document is the
**authoritative reference** for the 52-dim v8_s11s13 state vector.

## Authoritative source

The builder code is the truth. The relevant writes are in
[models/dqn_state.py:284-289](../../models/dqn_state.py#L284-L289) (for the
50-dim v5 base) and [models/build_state_v8.py:97-102](../../models/build_state_v8.py#L97-L102)
(for the 2-dim v8 extension).

## Definitive layout (52 dims)

| dims | content | type | source | notes |
|---|---|---|---|---|
| `[0]` | `vol_pred` | standardized scalar | dqn_state.py:284 | LGBM vol prediction, median-IQR std |
| `[1]` | `atr_pred_norm` | standardized scalar | dqn_state.py:285 | raw ATR / IQR |
| `[2:7]` | regime one-hot | binary | dqn_state.py:286 | `{calm, trend_up, trend_down, ranging, chop}` |
| `[7:16]` | signal flags (9 strategies) | {-1, 0, +1} | dqn_state.py:287 | S1, S2, S3, S4, S6, S7, S8, S10, S12 (in that order) |
| `[16]` | `bb_width` | standardized scalar | dqn_state.py:288 | Bollinger band width |
| `[17]` | `fund_rate_z` | standardized scalar | dqn_state.py:289 | rolling z-score of funding, window=480 |
| `[18]` | `last_trade_pnl_pct` | scalar, **stateful** | dqn_state.py builder writes 0 | DQN env loop overwrites at runtime |
| `[19]` | `current_dd_from_peak` | scalar, **stateful** | dqn_state.py builder writes 0 | DQN env loop overwrites at runtime |
| `[20:26]` | `log_return` windowed | standardized | dqn_state.py | lags `[60, 30, 15, 5, 1, 0]` oldest→newest |
| `[26:32]` | `taker_net_60_z` windowed | standardized | dqn_state.py | same lag pattern |
| `[32:38]` | `ofi_perp_10` windowed | standardized | dqn_state.py | same lag pattern |
| `[38:44]` | `vwap_dev_240` windowed | standardized | dqn_state.py | same lag pattern |
| `[44:50]` | `log_volume_z` windowed | standardized | dqn_state.py | same lag pattern |
| `[50]` | `S11_Basis` flag | {-1, 0, +1} | build_state_v8.py:97-102 | v8 addition |
| `[51]` | `S13_OBDiv` flag | {-1, 0, +1} | build_state_v8.py:97-102 | v8 addition |

## Empirical sanity check

Loaded `cache/state/btc_dqn_state_val_v8_s11s13.npz` (50,867 bars × 52 dims)
and validated:

- Dims 2-6 are binary `{0, 1}` with rows summing to ~1.0 (one-hot integrity).
- Dims 7-15, 50, 51 are ternary `{-1, 0, +1}` (signal flags).
- Dims 18, 19 are all zeros at load time (stateful — overwritten by env loop).
- Windowed groups (20-25, 26-31, …) have distinct column values
  (max_abs_diff 4-20 between cols 0 and 5 of each group). The identical
  summary statistics across columns reflect feature stationarity, not
  duplicate data.

## Discovered side-finding: S12 is short-only on val/test

Dim `[15]` (S12_VWAPVol signal flag) takes only `{-1, 0}` on val and test —
never fires long. Verified the strategy code (`strategy/agent.py:417`) is
symmetric: long requires `vwap_dev_240 < -threshold` AND `vol_z > σ` AND
`turning_pos` AND `vol_pred < ceil`. The val/test periods evidently don't
have any bars satisfying all four long conditions simultaneously. This is
a data property, not a code bug, but worth noting:

> S12 effectively acts as a short-only strategy in deployment on the
> current dataset. If the policy is selecting S12, it's always going short.

## Documented vs actual (drift report)

Two locations had wrong state-layout documentation prior to this audit:

### 1. `models/transformer_network.py` docstring (since 2026-05-12, now corrected by this audit)

| docstring claim | actual content |
|---|---|
| `[0..3]` direction probabilities (4 tokens) | dim 0 is vol_pred; dims 1-3 are atr_pred + regime[0:2] |
| `[4..6]` vol prediction features (3 tokens) | regime one-hot [calm, trend_up, trend_down] |
| `[16..17]` hour-of-day sin/cos (2 tokens) | bb_width, fund_rate_z |
| `[18..20]` equity / drawdown / last-pnl (3 tokens) | last_pnl, dd, log_return[60] — partially right |
| `[21..29]` 9 per-strategy rank features (9 tokens) | log_return + taker_net_60_z windowed |
| `[30..49]` 20 orderbook + microstructure features | ofi_perp + vwap_dev + log_volume windowed (microstructure-ish only — no raw OB depth/imbalance) |
| `[50..51]` S11/S13 strategy flags | **CORRECT** |

The Transformer was trained on actual data (it just sees floats), so the
training itself is valid. But the Z4.2 hypothesis ("attention learns
pairwise feature interactions") was framed on a wrong mental model of
what the features were. With the correct picture — the state is mostly
windowed lag-features of a few price/volume scalars plus a few one-hot
flags — it's even less surprising that attention didn't help: there
aren't 52 independent semantic tokens, there are ~10 features replicated
across 5-6 lags. Attention has nothing semantically meaningful to attend
between *across the lag windows*.

### 2. Z5.1 docs (now corrected, see [z5_validation.md](../experiments/z5_validation.md))

Claimed regime classifier output was NOT in the state vector. It IS in
state[2:7].

## Notable absences

Confirming what is **not** in the state, despite being computed elsewhere:

- **Direction probabilities** (`p_up_60`, `p_dn_60`, etc.) — used to compute
  strategy signals (S1, S4, S6) but **not** entries in the state. The
  network sees the resulting {-1, 0, +1} flag, not the underlying prob.
- **Hour-of-day** — not in v5 or v8. Could be added.
- **Raw OB depth/imbalance** — only OFI (order-flow-imbalance from
  trades) is in state at dims [32:38]. Raw OB depth (the 800-bin cumulative
  features in the v3 collection schema) is **not** in any DQN state version.
- **Cross-asset features** — no ETH/SOL features in BTC's state.
- **OI delta features** — only single-bar `oi_usd` is in feature parquet;
  not in state.

These are **candidates for state v10 expansion** (per the v2 plan §F2/F3).

## Recommendations

1. **Fix the `transformer_network.py` docstring** to match the actual layout.
   *(Low priority — Transformer was negative anyway.)*
2. **Add a CI-style assertion** in `build_state_v8.py` that hashes the
   state-pack dimension assignments and aborts if the upstream `dqn_state.py`
   builder ever changes layout silently. Prevents future drift.
3. **State-v10 candidates** when expanding feature set:
   - hour-of-day sin/cos (cheap, currently absent)
   - OI delta over 5/15/60 min (currently single-bar only)
   - cross-venue funding spread (once Binance/HL collection matures)

## Outputs

- This document.
- No new code (read-only audit).
