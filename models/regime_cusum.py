"""
CUSUM + Hurst regime classifier.

Detects 5 states from price-only features:
  trend_up   — persistent positive returns (CUSUM+ ≥ p75 train, Hurst ≥ p65)
  trend_down — persistent negative returns (CUSUM- ≤ p25, Hurst ≥ p65)
  ranging    — mean-reverting (Hurst ≤ p35)
  chop       — high volatility but no clear direction
  calm       — low volatility (bb_width ≤ p30)

Thresholds fitted on train split only. KW gate validates per-state forward
return distinctness.

Run: python3 -m models.regime_cusum [ticker]
Saves: cache/{ticker}_regime_cusum.parquet
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader   import load_meta
from data.gaps     import clean_mask
from models.splits import sequential

CACHE  = ROOT / "cache"
WINDOW = 60          # look-back for rolling CUSUM + Hurst
FWD    = 30          # forward-return horizon for KW validation
STATES = ["calm", "trend_up", "trend_down", "ranging", "chop"]


def _rolling_hurst(returns: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Hurst proxy via lag-1 autocorrelation: H = 0.5 + ACF(1)/2.

    ACF > 0 → persistent (trending)        → H > 0.5
    ACF < 0 → anti-persistent (reverting) → H < 0.5
    Vectorised; O(n) via pandas rolling.
    """
    s   = pd.Series(returns)
    lag = s.shift(1)
    cov = (s * lag).rolling(window, min_periods=window // 2).mean() - \
          s.rolling(window, min_periods=window // 2).mean() * \
          lag.rolling(window, min_periods=window // 2).mean()
    var = s.rolling(window, min_periods=window // 2).var()
    acf = (cov / var.replace(0, np.nan)).fillna(0).clip(-1, 1)
    return (0.5 + acf / 2).values


def _load(ticker: str) -> dict:
    pq        = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    feat_cols = [c for c in pq.columns if c != "timestamp"]
    ci        = {c: i for i, c in enumerate(feat_cols)}
    meta      = load_meta(ticker)
    ts_meta   = meta["timestamp"].values
    price_meta= meta["perp_ask_price"].values
    gap_ok    = clean_mask(pd.Series(ts_meta), max_lookback=1440)

    X_raw  = pq[feat_cols].values
    ts_all = pq["timestamp"].values
    row_ok = gap_ok & ~np.isnan(X_raw).any(axis=1)
    X      = X_raw[row_ok]
    ts     = ts_all[row_ok]
    prices = np.array([dict(zip(ts_meta, price_meta))[t] for t in ts])
    n      = len(X)
    sp     = sequential(n, 0.50, 0.25)
    return dict(X=X, ts=ts, prices=prices, ci=ci, sp=sp, n=n)


def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  CUSUM + Hurst regime classifier — {ticker.upper()}")
    print(f"{'='*70}\n")

    d = _load(ticker)
    print(f"  Data loaded ({d['n']:,} bars)  {time.perf_counter()-t0:.1f}s")

    prices = d["prices"]
    ret1   = np.diff(np.log(np.maximum(prices, 1e-8)), prepend=0.0)

    # ── Hurst + CUSUM features ────────────────────────────────────────────────
    hurst = _rolling_hurst(ret1, WINDOW)

    ret_smooth = pd.Series(ret1).rolling(5, min_periods=1).mean()
    cusum_p    = ret_smooth.clip(lower=0).rolling(WINDOW, min_periods=1).sum().values
    cusum_m    = ret_smooth.clip(upper=0).rolling(WINDOW, min_periods=1).sum().values
    cusum_std  = pd.Series(ret1).rolling(WINDOW, min_periods=WINDOW//2).std().values + 1e-8
    cusum_p_n  = cusum_p / cusum_std
    cusum_m_n  = cusum_m / cusum_std

    bb = d["X"][:, d["ci"]["bb_width"]]

    # ── fit thresholds on TRAIN ───────────────────────────────────────────────
    sp = d["sp"]
    cusp_th  = float(np.percentile(cusum_p_n[sp.train], 75))
    cusm_th  = float(np.percentile(cusum_m_n[sp.train], 25))
    hurst_hi = float(np.percentile(hurst[sp.train],     65))
    hurst_lo = float(np.percentile(hurst[sp.train],     35))
    bb_lo    = float(np.percentile(bb[sp.train],        30))

    print(f"\n  Thresholds (train fit):")
    print(f"    CUSUM+   p75 = {cusp_th:>+8.3f}")
    print(f"    CUSUM-   p25 = {cusm_th:>+8.3f}")
    print(f"    Hurst    p65 = {hurst_hi:>+8.3f}    p35 = {hurst_lo:>+8.3f}")
    print(f"    bb_width p30 = {bb_lo:.5f}")

    # ── label all bars (vectorised) ───────────────────────────────────────────
    is_calm       = bb < bb_lo
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

    # ── per-split distribution ────────────────────────────────────────────────
    print(f"\n  State distribution:")
    for split_name, idx in [("train", sp.train), ("val", sp.val), ("test", sp.test)]:
        line = f"    {split_name:<5}:"
        for st in STATES:
            pct = (states[idx] == st).mean() * 100
            line += f"  {st}={pct:>4.1f}%"
        print(line)

    # ── KW test ───────────────────────────────────────────────────────────────
    n       = d["n"]
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - FWD] = prices[FWD:n] / prices[:n - FWD] - 1

    groups = [fwd_ret[(states == st) & ~np.isnan(fwd_ret)] for st in STATES]
    groups = [g for g in groups if len(g) >= 20]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        gate = "PASS ✓" if p < 0.01 else "FAIL ✗"
        print(f"\n  KW gate (per-state fwd-return distinctness, fwd={FWD}): {gate}  (p={p:.2e})")

    # ── per-state forward returns on TEST (out-of-sample) ─────────────────────
    print(f"\n  TEST split — per-state forward returns (fwd={FWD} bars):")
    print(f"    {'State':<11} {'N':>6}  {'Mean%':>8}  {'Med%':>8}  {'Pos%':>6}")
    print("    " + "─" * 46)
    te = sp.test
    fr = fwd_ret[te]
    st = states[te]
    valid = ~np.isnan(fr)
    for stn in STATES:
        m = (st == stn) & valid
        if m.sum() < 10:
            continue
        r = fr[m]
        print(f"    {stn:<11} {len(r):>6,}  {r.mean()*100:>+7.3f}%  "
              f"{np.median(r)*100:>+7.3f}%  {(r>0).mean()*100:>5.1f}%")

    # ── save ──────────────────────────────────────────────────────────────────
    s2i = {s: i for i, s in enumerate(STATES)}
    df_out = pd.DataFrame({
        "timestamp":  d["ts"],
        "state":      np.array([s2i[s] for s in states], dtype=np.int8),
        "state_name": states,
    })
    out = CACHE / f"{ticker}_regime_cusum.parquet"
    df_out.to_parquet(out, index=False)
    print(f"\n  → {out.name}  ({len(df_out):,} rows)  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
