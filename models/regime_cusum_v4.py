"""
CUSUM + Hurst regime classifier v4 — refit on bars 1,440 → 101,440.

DQN-v5 variant: ignores `clean_mask`. Operates on the continuous index.

Outputs:
  cache/btc_regime_cusum_v4.parquet  — per-bar state + state_name
  cache/btc_regime_cusum_v4_thresholds.json  — fitted thresholds

Run: python3 -m models.regime_cusum_v4 [ticker]
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_meta

CACHE        = ROOT / "cache"
WARMUP       = 1440
VOL_TRAIN_E  = 101_440
WINDOW       = 60
FWD          = 30
STATES       = ["calm", "trend_up", "trend_down", "ranging", "chop"]
S2I          = {s: i for i, s in enumerate(STATES)}


def _rolling_hurst(returns: np.ndarray, window: int = WINDOW) -> np.ndarray:
    s   = pd.Series(returns)
    lag = s.shift(1)
    cov = (s * lag).rolling(window, min_periods=window // 2).mean() - \
          s.rolling(window, min_periods=window // 2).mean() * \
          lag.rolling(window, min_periods=window // 2).mean()
    var = s.rolling(window, min_periods=window // 2).var()
    acf = (cov / var.replace(0, np.nan)).fillna(0).clip(-1, 1)
    return (0.5 + acf / 2).values


def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  REGIME CUSUM v4 — {ticker.upper()}\n{'='*70}")

    pq        = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    ts_arr    = pq["timestamp"].values
    bb        = pq["bb_width"].values

    meta      = load_meta(ticker)
    assert (meta["timestamp"].values == ts_arr).all()
    price     = meta["perp_ask_price"].values

    # ── work on bars WARMUP → end (full RL period) ───────────────────────────
    n_full = len(price) - WARMUP
    p_use  = price[WARMUP:]
    bb_use = bb[WARMUP:]
    ts_use = ts_arr[WARMUP:]
    print(f"  bars used: {n_full:,} (warmup={WARMUP} dropped)")

    # ── features ──────────────────────────────────────────────────────────────
    ret1   = np.diff(np.log(np.maximum(p_use, 1e-8)), prepend=0.0)
    hurst  = _rolling_hurst(ret1, WINDOW)

    ret_smooth = pd.Series(ret1).rolling(5, min_periods=1).mean()
    cusum_p    = ret_smooth.clip(lower=0).rolling(WINDOW, min_periods=1).sum().values
    cusum_m    = ret_smooth.clip(upper=0).rolling(WINDOW, min_periods=1).sum().values
    cusum_std  = pd.Series(ret1).rolling(WINDOW, min_periods=WINDOW//2).std().values + 1e-8
    cusum_p_n  = cusum_p / cusum_std
    cusum_m_n  = cusum_m / cusum_std

    # ── train slice for thresholds: bars 0 → (VOL_TRAIN_E - WARMUP) ─────────
    n_vt = VOL_TRAIN_E - WARMUP
    cusp_th  = float(np.nanpercentile(cusum_p_n[:n_vt], 75))
    cusm_th  = float(np.nanpercentile(cusum_m_n[:n_vt], 25))
    hurst_hi = float(np.nanpercentile(hurst    [:n_vt], 65))
    hurst_lo = float(np.nanpercentile(hurst    [:n_vt], 35))
    bb_lo    = float(np.nanpercentile(bb_use   [:n_vt], 30))

    # NaN-safe label inputs: leading rolling-window NaNs become 0 so they fall
    # into "chop" rather than producing string-NaN labels downstream.
    cusum_p_n = np.nan_to_num(cusum_p_n, nan=0.0)
    cusum_m_n = np.nan_to_num(cusum_m_n, nan=0.0)
    hurst     = np.nan_to_num(hurst,     nan=0.5)

    print(f"\n  Thresholds (vol-train fit):")
    print(f"    CUSUM+   p75 = {cusp_th:>+8.3f}")
    print(f"    CUSUM-   p25 = {cusm_th:>+8.3f}")
    print(f"    Hurst    p65 = {hurst_hi:>+8.3f}    p35 = {hurst_lo:>+8.3f}")
    print(f"    bb_width p30 = {bb_lo:.5f}")

    # ── label all bars ────────────────────────────────────────────────────────
    is_calm       = bb_use < bb_lo
    is_trending   = (~is_calm) & (hurst >= hurst_hi)
    is_trend_up   = is_trending & (cusum_p_n >= cusp_th)
    is_trend_down = is_trending & (~is_trend_up) & (cusum_m_n <= cusm_th)
    is_chop_hi    = is_trending & (~is_trend_up) & (~is_trend_down)
    is_ranging    = (~is_calm) & (~is_trending) & (hurst <= hurst_lo)

    states = np.select(
        [is_calm, is_trend_up, is_trend_down, is_ranging, is_chop_hi],
        ["calm",  "trend_up",  "trend_down",  "ranging",  "chop"],
        default="chop",
    )

    # ── distribution per split (vol-train / DQN-train / DQN-val / DQN-test) ─
    DQN_TRAIN_E = 281_440 - WARMUP
    DQN_VAL_E   = 332_307 - WARMUP
    splits = [
        ("vol-train", 0,           n_vt),
        ("DQN-train", n_vt,        DQN_TRAIN_E),
        ("DQN-val",   DQN_TRAIN_E, DQN_VAL_E),
        ("DQN-test",  DQN_VAL_E,   n_full),
    ]
    print(f"\n  State distribution per split:")
    for nm, a, b in splits:
        line = f"    {nm:<10}:"
        for st in STATES:
            pct = (states[a:b] == st).mean() * 100
            line += f"  {st}={pct:>4.1f}%"
        print(line)

    # ── KW gate on DQN-train fwd returns ─────────────────────────────────────
    fwd = np.full(n_full, np.nan)
    fwd[:n_full - FWD] = p_use[FWD:n_full] / p_use[:n_full - FWD] - 1
    dq_a, dq_b = n_vt, DQN_TRAIN_E
    fwd_dq = fwd[dq_a:dq_b]
    st_dq  = states[dq_a:dq_b]
    valid  = ~np.isnan(fwd_dq)
    groups = [fwd_dq[(st_dq == s) & valid] for s in STATES]
    groups = [g for g in groups if len(g) >= 20]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        gate = "PASS ✓" if p < 0.01 else "FAIL ✗"
        print(f"\n  KW gate (DQN-train fwd-{FWD} distinctness): {gate}  (p={p:.2e})")

    # ── per-state fwd returns on DQN-test ────────────────────────────────────
    print(f"\n  DQN-test split — per-state fwd returns (fwd={FWD} bars):")
    print(f"    {'State':<11} {'N':>6}  {'Mean%':>8}  {'Med%':>8}  {'Pos%':>6}")
    print("    " + "─" * 46)
    fwd_te = fwd[DQN_VAL_E:n_full]
    st_te  = states[DQN_VAL_E:n_full]
    valid  = ~np.isnan(fwd_te)
    for s in STATES:
        m = (st_te == s) & valid
        if m.sum() < 10:
            continue
        r = fwd_te[m]
        print(f"    {s:<11} {len(r):>6,}  {r.mean()*100:>+7.3f}%  "
              f"{np.median(r)*100:>+7.3f}%  {(r>0).mean()*100:>5.1f}%")

    # ── save ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame({
        "timestamp":  ts_use,
        "state":      np.array([S2I[s] for s in states], dtype=np.int8),
        "state_name": states,
    })
    out = CACHE / f"{ticker}_regime_cusum_v4.parquet"
    df_out.to_parquet(out, index=False)

    thresh = {
        "cusp_p75": cusp_th, "cusm_p25": cusm_th,
        "hurst_p65": hurst_hi, "hurst_p35": hurst_lo,
        "bb_p30": bb_lo, "kw_p_value": float(p) if len(groups) >= 2 else None,
    }
    (CACHE / f"{ticker}_regime_cusum_v4_thresholds.json").write_text(json.dumps(thresh, indent=2))

    print(f"\n  → {out.name}  ({len(df_out):,} rows)  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
