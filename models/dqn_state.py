"""
DQN v5 state-array builder.

Phases 1C + 2D combined: compute median+IQR standardize stats on vol-train,
then materialize per-bar 50-dim state vectors + 10-dim valid-actions masks
for DQN-train, DQN-val, DQN-test splits.

State layout (50 dims) — see experiments/dqn_strategy_selector_prompt.md:
  Static (20):
     0  vol_pred                      standardized
     1  atr_pred_norm                 standardized (raw atr / iqr)
     2-6  regime one-hot              {calm, trend_up, trend_down, ranging, chop}
     7-15 signal flags                {-1, 0, +1} for S1, S2, S3, S4, S6, S7, S8, S10, S12
     16  bb_width                     standardized
     17  fund_rate_z                  standardized (rolling z-score, win=480)
     18  last_trade_pnl_pct           STATEFUL (filled with 0 here; DQN loop overwrites)
     19  current_dd_from_peak         STATEFUL (filled with 0 here; DQN loop overwrites)
  Windowed (30) — lags [60, 30, 15, 5, 1, 0], oldest-to-newest:
     20-25 log_return                  standardized
     26-31 taker_net_60_z              standardized
     32-37 ofi_perp_10                 standardized
     38-43 vwap_dev_240                standardized
     44-49 log_volume_z                standardized

Outputs:
  cache/btc_dqn_standardize_v5.json
  cache/btc_dqn_state_train.npz   — state (N×50), valid_actions (N×10), aux info
  cache/btc_dqn_state_val.npz
  cache/btc_dqn_state_test.npz

Direction predictions: loaded from cache/btc_pred_dir_{col}_v4.npz (full 383k
bar arrays from direction_dl_v4.py). These ARE used to compute strategy
signals (S1/S4/S6 depend on p_up_60/p_dn_60), but they are NOT entries in the
50-dim DQN state vector — the network sees only the per-strategy {-1,0,+1}
flag that results from the strategy's signal logic.

Run: python3 -m models.dqn_state [ticker]
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader     import load_meta
from strategy.agent  import STRATEGIES, DEFAULT_PARAMS

CACHE        = ROOT / "cache"
WARMUP       = 1440
VOL_TRAIN_E  = 101_440
DQN_TRAIN_E  = 281_440
DQN_VAL_E    = 332_307
LAGS         = [60, 30, 15, 5, 1, 0]
WIN_FEATS    = ["log_return", "taker_net_60_z", "ofi_perp_10", "vwap_dev_240", "log_volume_z"]
STATIC_NUM   = ["vol_pred", "atr_pred_norm", "bb_width", "fund_rate_z"]
STRAT_KEYS   = ["S1_VolDir", "S2_Funding", "S3_BBRevert", "S4_MACDTrend",
                "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow",
                "S10_Squeeze", "S12_VWAPVol"]
REGIME_STATES = ["calm", "trend_up", "trend_down", "ranging", "chop"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _med_iqr(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    return med, max(iqr, 1e-8)


def _std(x, m, q, clip: float = 10.0):
    """Median/IQR standardize, clipped to [-clip, clip] to keep DQN inputs bounded.
    Across-gap bars produce extreme log_return / vwap_dev outliers (rare 50–500σ
    events). Clipping at ±10 lets the network see "this is extreme" without
    saturating downstream activations."""
    z = ((x - m) / q).astype(np.float32)
    return np.clip(z, -clip, clip)


# ── strategy-input dataframe ─────────────────────────────────────────────────

# columns the strategies in strategy/agent.py touch (subset of features parquet
# + meta-only fields). Matches backtest/run.py:_STRAT_COLS but for new
# (no-clean-mask) bar layout.
_STRAT_COLS = [
    "bb_pct_b", "bb_width", "rsi_6", "rsi_14", "macd_hist",
    "ofi_perp_10_r15", "ofi_perp_10", "taker_imb_5", "taker_net_15",
    "fund_rate", "fund_mom_480", "ret_sma_200", "vwap_dev_1440",
    "sma_50", "sma_200",
    "oi_price_div_15", "taker_net_30", "taker_net_60",
    "taker_imb_30", "ret_15", "vwap_dev_240",
    "vol_z_spot_60", "spot_imbalance", "perp_imbalance",
    # meta-only:
    "spot_large_bid_count", "spot_large_ask_count",
    "perp_large_bid_count", "perp_large_ask_count",
    "diff_price",
]


def _build_strategy_df(pq_use: pd.DataFrame, meta_use: pd.DataFrame,
                        price_use: np.ndarray, atr_use: np.ndarray,
                        rank_use: np.ndarray, dir_preds: dict) -> pd.DataFrame:
    """Build the per-bar inputs for all 9 strategies.

    dir_preds: {'up_60': arr, 'down_60': arr, 'up_100': arr, 'down_100': arr}
               each array length matches `pq_use` (bars WARMUP → end).
    """
    df = pd.DataFrame({
        "price":     price_use,
        "atr_pred":  atr_use,
        "vol_pred":  rank_use,            # rank used as gate; matches existing convention
    })
    for c in _STRAT_COLS:
        if c in pq_use.columns:
            df[c] = pq_use[c].values
        elif c in meta_use.columns:
            df[c] = meta_use[c].values
        else:
            df[c] = 0.0
    df["p_up_60"]   = dir_preds["up_60"]
    df["p_dn_60"]   = dir_preds["down_60"]
    df["p_up_100"]  = dir_preds["up_100"]
    df["p_dn_100"]  = dir_preds["down_100"]
    return df


# ── windowed features ────────────────────────────────────────────────────────

def _windowed_series(price: np.ndarray, taker_net_60: np.ndarray,
                       ofi_perp_10: np.ndarray, vwap_dev_240: np.ndarray,
                       perp_volume: np.ndarray) -> dict:
    """Compute the 5 raw windowed source series — already aligned to bars
    WARMUP→end. taker_net_60_z and log_volume_z use the full-series rolling
    statistic windows; standardization stats applied later."""
    log_return    = np.diff(np.log(np.maximum(price, 1e-8)), prepend=0.0).astype(np.float32)
    log_return    = np.nan_to_num(log_return, nan=0.0, posinf=0.0, neginf=0.0)

    # taker_net_60_z: rolling-480 z-score to mirror strategy_8's approach
    s  = pd.Series(taker_net_60)
    sd = s.rolling(480, min_periods=120).std().bfill().fillna(1.0).values + 1e-8
    taker_net_60_z = np.nan_to_num((s.values / sd), nan=0.0).astype(np.float32)

    # log_volume: log(perp_minute_volume). Replace 0/neg with tiny.
    log_vol = np.log(np.maximum(perp_volume, 1e-3)).astype(np.float32)
    log_vol = np.nan_to_num(log_vol, nan=0.0)

    return {
        "log_return":     log_return,
        "taker_net_60_z": taker_net_60_z,
        "ofi_perp_10":    np.nan_to_num(ofi_perp_10.astype(np.float32), nan=0.0),
        "vwap_dev_240":   np.nan_to_num(vwap_dev_240.astype(np.float32), nan=0.0),
        "log_volume_z":   log_vol,        # standardized later
    }


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  DQN STATE BUILDER v5 — {ticker.upper()}\n{'='*70}")

    # ── load source data ─────────────────────────────────────────────────────
    pq    = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    meta  = load_meta(ticker)
    assert (pq["timestamp"].values == meta["timestamp"].values).all()
    print(f"  feat parquet: {pq.shape}  meta: {meta.shape}")

    # vol preds (full bars WARMUP→end)
    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_full  = vol["atr"]            # length 383,174
    rank_full = vol["rank"]
    n_full    = len(atr_full)
    print(f"  vol preds: atr.shape={atr_full.shape}  spearman OOS={vol['spearman_oos']:.3f}")

    # NaN check — vol-train portion may have NaN if ok_tr filtered some rows
    nan_atr = np.isnan(atr_full).sum()
    nan_rk  = np.isnan(rank_full).sum()
    print(f"  vol preds NaN: atr={nan_atr}  rank={nan_rk}  (forward-fill)")
    atr_full  = pd.Series(atr_full).ffill().bfill().values.astype(np.float32)
    rank_full = pd.Series(rank_full).ffill().bfill().values.astype(np.float32)

    # regime labels
    rg = pd.read_parquet(CACHE / f"{ticker}_regime_cusum_v4.parquet")
    assert len(rg) == n_full and (rg["timestamp"].values == pq["timestamp"].values[WARMUP:]).all()
    regime_names = rg["state_name"].values
    regime_oh    = np.zeros((n_full, 5), dtype=np.float32)
    for i, s in enumerate(REGIME_STATES):
        regime_oh[:, i] = (regime_names == s).astype(np.float32)

    # slice full arrays past warmup
    pq_use   = pq.iloc[WARMUP:].reset_index(drop=True)
    meta_use = meta.iloc[WARMUP:].reset_index(drop=True)
    ts_use   = pq_use["timestamp"].values
    price_use= meta_use["perp_ask_price"].values

    # ── direction preds v4 ───────────────────────────────────────────────────
    dir_preds = {}
    for col in ["up_60", "down_60", "up_100", "down_100"]:
        d = np.load(CACHE / f"{ticker}_pred_dir_{col}_v4.npz")
        a = d["preds"].astype(np.float32)
        assert len(a) == n_full, f"dir_{col} length {len(a)} != {n_full}"
        nan_n = np.isnan(a).sum()
        if nan_n:
            a = pd.Series(a).bfill().fillna(0.5).values.astype(np.float32)
        dir_preds[col] = a
        print(f"  dir {col:<8}  loaded {len(a):,} bars  AUC RL={float(d['auc_rl']):.4f}")

    # ── strategy signals ─────────────────────────────────────────────────────
    print("\n  Computing strategy signals (direction preds from v4 CNN-LSTM) ...")
    df_strat = _build_strategy_df(pq_use, meta_use, price_use, atr_full, rank_full, dir_preds)
    sig_arr  = np.zeros((n_full, len(STRAT_KEYS)), dtype=np.int8)
    for i, key in enumerate(STRAT_KEYS):
        fn, _ = STRATEGIES[key]
        s, _, _ = fn(df_strat, DEFAULT_PARAMS[key])
        sig_arr[:, i] = s.astype(np.int8)
        n_long  = (s > 0).sum()
        n_short = (s < 0).sum()
        n_any   = ((s != 0)).sum()
        print(f"    {key:<14}  long={n_long:>6,}  short={n_short:>6,}  any={n_any:>6,}  ({n_any/n_full*100:.1f}%)")

    # ── windowed source series ───────────────────────────────────────────────
    print("\n  Building windowed source series ...")
    src = _windowed_series(
        price       = price_use,
        taker_net_60= pq_use["taker_net_60"].values,
        ofi_perp_10 = pq_use["ofi_perp_10"].values,
        vwap_dev_240= pq_use["vwap_dev_240"].values,
        perp_volume = meta_use["perp_minute_volume"].values,
    )

    # ── fund_rate_z ──────────────────────────────────────────────────────────
    fund_z = ((meta_use["fund_rate"] -
               meta_use["fund_rate"].rolling(480, min_periods=120).mean()) /
              (meta_use["fund_rate"].rolling(480, min_periods=120).std() + 1e-12)).fillna(0).values.astype(np.float32)

    # ── Phase 1C: standardize stats from vol-train slice (0 → VOL_TRAIN_E-WARMUP) ─
    n_vt = VOL_TRAIN_E - WARMUP                   # 100,000
    print(f"\n  Phase 1C — fitting median/IQR on vol-train slice (n={n_vt:,}) ...")
    stats = {}

    # static numeric
    static_arrays = {
        "vol_pred":      rank_full,
        "atr_pred_norm": atr_full,
        "bb_width":      pq_use["bb_width"].values.astype(np.float32),
        "fund_rate_z":   fund_z,
    }
    for k, a in static_arrays.items():
        m, q = _med_iqr(a[:n_vt])
        stats[k] = {"median": m, "iqr": q}
        print(f"    {k:<16}  med={m:>+10.4f}  iqr={q:>+10.4f}")

    # windowed (use lag-0 series → its own raw values on vol-train)
    for k, a in src.items():
        m, q = _med_iqr(a[:n_vt])
        stats[k] = {"median": m, "iqr": q}
        print(f"    {k:<16}  med={m:>+10.4f}  iqr={q:>+10.4f}")

    out_json = CACHE / f"{ticker}_dqn_standardize_v5.json"
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"  → {out_json.name}")

    # ── apply standardization ────────────────────────────────────────────────
    static_std = {}
    for k, a in static_arrays.items():
        s = stats[k]
        static_std[k] = _std(a, s["median"], s["iqr"])
    src_std = {}
    for k, a in src.items():
        s = stats[k]
        src_std[k] = _std(a, s["median"], s["iqr"])

    # ── Phase 2D: build per-bar state arrays ────────────────────────────────
    print(f"\n  Phase 2D — building 50-dim state arrays ...")

    # only bars in DQN-train..DQN-test contribute to the cached arrays.
    # but we compute states for ALL bars first (vectorized), then slice.
    n_bars = n_full
    state  = np.zeros((n_bars, 50), dtype=np.float32)

    # static block
    state[:, 0]    = static_std["vol_pred"]
    state[:, 1]    = static_std["atr_pred_norm"]
    state[:, 2:7]  = regime_oh
    state[:, 7:16] = sig_arr.astype(np.float32)
    state[:, 16]   = static_std["bb_width"]
    state[:, 17]   = static_std["fund_rate_z"]
    # 18, 19 left zero (stateful fields filled by DQN training loop)

    # windowed block — vectorized via shift
    base_idx = 20
    for i, fname in enumerate(WIN_FEATS):
        a = src_std[fname]
        for j, lag in enumerate(LAGS):
            col = base_idx + i * 6 + j
            if lag == 0:
                state[:, col] = a
            else:
                shifted = np.zeros_like(a)
                shifted[lag:] = a[:-lag]      # bar t gets value from bar t-lag; head zero-padded
                state[:, col] = shifted

    # ── valid actions mask ───────────────────────────────────────────────────
    # 10 actions: 0=NO_TRADE, 1..9 = STRAT_KEYS in order
    valid = np.zeros((n_bars, 10), dtype=np.bool_)
    valid[:, 0] = True                              # NO_TRADE always valid
    for i in range(9):
        valid[:, i + 1] = sig_arr[:, i] != 0

    # ── slice to splits ──────────────────────────────────────────────────────
    splits = [
        ("train", n_vt,                     DQN_TRAIN_E - WARMUP),
        ("val",   DQN_TRAIN_E - WARMUP,     DQN_VAL_E   - WARMUP),
        ("test",  DQN_VAL_E   - WARMUP,     n_full),
    ]
    for nm, a, b in splits:
        sub_state = state[a:b]
        sub_valid = valid[a:b]
        sub_ts    = ts_use[a:b]
        sub_price = price_use[a:b]
        sub_sig   = sig_arr[a:b]
        np.savez(
            CACHE / f"{ticker}_dqn_state_{nm}.npz",
            state         = sub_state,
            valid_actions = sub_valid,
            ts            = sub_ts.astype(np.int64),
            price         = sub_price.astype(np.float32),
            signals       = sub_sig,                     # raw {-1,0,+1} per strategy
            atr           = atr_full[a:b].astype(np.float32),
            rank          = rank_full[a:b].astype(np.float32),
            regime_id     = np.array([REGIME_STATES.index(s) for s in regime_names[a:b]], dtype=np.int8),
        )
        print(f"    {nm:<6}  bars {a:>7,}–{b:>7,}  ({b-a:,})  → btc_dqn_state_{nm}.npz")

    # ── sanity / gate checks ────────────────────────────────────────────────
    print(f"\n  Phase 2 gate checks (DQN-train slice):")
    a, b = n_vt, DQN_TRAIN_E - WARMUP
    sub  = state[a:b]
    print(f"    state shape:       {sub.shape}  dtype={sub.dtype}")
    # numeric standardize sanity per dim group
    def _stat(name, idxs):
        col = sub[:, idxs].ravel()
        print(f"    {name:<22}  mean={col.mean():>+.3f}  std={col.std():.3f}  "
              f"min={col.min():>+.2f}  max={col.max():>+.2f}")
    _stat("static numeric (16,17)", [16, 17])
    _stat("vol/atr  (0,1)",         [0, 1])
    _stat("log_return (20-25)",     list(range(20, 26)))
    _stat("taker_net_60_z (26-31)", list(range(26, 32)))
    _stat("ofi_perp_10 (32-37)",    list(range(32, 38)))
    _stat("vwap_dev_240 (38-43)",   list(range(38, 44)))
    _stat("log_volume_z (44-49)",   list(range(44, 50)))

    # signal flag activity per-strategy
    print(f"\n  DQN-train signal activity:")
    for i, key in enumerate(STRAT_KEYS):
        s = sig_arr[a:b, i]
        n_active = (s != 0).sum()
        print(f"    {key:<14}  active bars = {n_active:>6,}  ({n_active/(b-a)*100:>5.2f}%)")

    print(f"\n  Action mask coverage (≥1 strategy active per bar):")
    for nm, a2, b2 in splits:
        v = valid[a2:b2, 1:].any(axis=1)
        print(f"    {nm:<6}  any-strategy-active = {v.mean()*100:>5.2f}%")

    print(f"\n  Total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
