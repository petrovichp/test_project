"""
HMM regime classifier with ADX + Hurst features.

7 features (in this order):
  bb_width, ret_60, fund_mean_1440, oi_z_1440, vwap_dev_240, adx, hurst

GaussianHMM(5 states, diag covariance), fit on train split, predict on full
sequence. States labeled heuristically from scaled feature means.

Run: python3 -m models.regime_hmm [ticker]
Saves: cache/{ticker}_regime_hmm.parquet
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import kruskal

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader   import load_meta
from data.gaps     import clean_mask
from models.splits import sequential

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("pip install hmmlearn")

CACHE        = ROOT / "cache"
HURST_WINDOW = 100
ADX_PERIOD   = 14
N_STATES     = 5
FWD          = 30


def _adx(prices: np.ndarray, period: int = ADX_PERIOD) -> np.ndarray:
    """Simplified ADX from close prices (no high/low). Returns ADX array."""
    c   = pd.Series(prices)
    d   = c.diff()
    k   = 2 / (period + 1)
    dmp = d.clip(lower=0).ewm(alpha=k, min_periods=period).mean()
    dmm = (-d).clip(lower=0).ewm(alpha=k, min_periods=period).mean()
    trr = d.abs().ewm(alpha=k, min_periods=period).mean()
    di_p = (dmp / trr.replace(0, np.nan)).fillna(0) * 100
    di_m = (dmm / trr.replace(0, np.nan)).fillna(0) * 100
    dx   = ((di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)).fillna(0) * 100
    return dx.ewm(alpha=k, min_periods=period).mean().values


def _rolling_hurst(returns: np.ndarray, window: int = HURST_WINDOW) -> np.ndarray:
    """Hurst proxy via lag-1 autocorrelation: H = 0.5 + ACF(1)/2."""
    s   = pd.Series(returns)
    lag = s.shift(1)
    cov = (s * lag).rolling(window, min_periods=window // 2).mean() - \
          s.rolling(window, min_periods=window // 2).mean() * \
          lag.rolling(window, min_periods=window // 2).mean()
    var = s.rolling(window, min_periods=window // 2).var()
    acf = (cov / var.replace(0, np.nan)).fillna(0).clip(-1, 1)
    return (0.5 + acf / 2).values


def _name_state(means_scaled: np.ndarray) -> str:
    """Label HMM state from scaled feature means.
    Order: bb_width, ret_60, fund_mean_1440, oi_z_1440, vwap_dev_240, adx, hurst
    """
    bb, r60, fund, oi, vwap, adx, hurst = means_scaled
    if fund >  0.7: return "fund_long"
    if fund < -0.7: return "fund_short"
    if hurst > 0.3 and adx > 0.3:
        comp = 0.6 * r60 + 0.4 * vwap
        if comp >  0.15: return "trend_bull"
        if comp < -0.15: return "trend_bear"
        return "high_vol_chop"
    if hurst < -0.3: return "ranging"
    if bb >  0.4:    return "high_vol_chop"
    return "ranging"


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
    print(f"  HMM + ADX + Hurst regime classifier — {ticker.upper()}")
    print(f"{'='*70}\n")

    d = _load(ticker)
    print(f"  Data loaded ({d['n']:,} bars)  {time.perf_counter()-t0:.1f}s")

    prices = d["prices"]
    ret1   = np.diff(np.log(np.maximum(prices, 1e-8)),
                      prepend=np.log(max(prices[0], 1e-8)))

    # ── compute ADX + Hurst ───────────────────────────────────────────────────
    adx_arr = _adx(prices)
    hurst   = _rolling_hurst(ret1)

    # ── feature matrix ────────────────────────────────────────────────────────
    base_feats = ["bb_width", "ret_60", "fund_mean_1440", "oi_z_1440", "vwap_dev_240"]
    X_base = np.column_stack([d["X"][:, d["ci"][f]] for f in base_feats])
    X_ext  = np.column_stack([X_base, adx_arr, hurst])
    X_ext  = pd.DataFrame(X_ext).ffill().fillna(0).values   # NaN warmup → ffill/0

    # ── scale: fit on train only ─────────────────────────────────────────────
    sp      = d["sp"]
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_ext[sp.train])
    X_sc    = scaler.transform(X_ext)

    # ── fit HMM ───────────────────────────────────────────────────────────────
    print(f"  Fitting GaussianHMM (5 states, diag, 50 iter) ...")
    t1 = time.perf_counter()
    model = GaussianHMM(n_components=N_STATES, covariance_type="diag",
                         n_iter=50, random_state=42, tol=1e-3)
    model.fit(X_tr_sc)
    print(f"    fit done in {time.perf_counter()-t1:.1f}s")

    states_int = model.predict(X_sc)

    # ── label states (do NOT deduplicate — multiple states may carry same label) ──
    state_labels = [_name_state(m) for m in model.means_]
    states       = np.array([state_labels[s] for s in states_int])

    print(f"\n  HMM state means → labels:")
    for i, (m, lbl) in enumerate(zip(model.means_, state_labels)):
        print(f"    state {i}  → {lbl:<14}  bb={m[0]:>+5.2f}  ret60={m[1]:>+5.2f}  "
              f"fund={m[2]:>+5.2f}  oi={m[3]:>+5.2f}  vwap={m[4]:>+5.2f}  "
              f"adx={m[5]:>+5.2f}  hurst={m[6]:>+5.2f}")

    unique_lbls = sorted(set(state_labels))
    print(f"\n  Unique labels: {unique_lbls}")
    print(f"\n  State distribution:")
    for split_name, idx in [("train", sp.train), ("val", sp.val), ("test", sp.test)]:
        line = f"    {split_name:<5}:"
        for st in unique_lbls:
            pct = (states[idx] == st).mean() * 100
            line += f"  {st}={pct:>4.1f}%"
        print(line)

    # ── KW test ───────────────────────────────────────────────────────────────
    n       = d["n"]
    fwd_ret = np.full(n, np.nan)
    fwd_ret[:n - FWD] = prices[FWD:n] / prices[:n - FWD] - 1

    groups = [fwd_ret[(states == st) & ~np.isnan(fwd_ret)] for st in unique_lbls]
    groups = [g for g in groups if len(g) >= 20]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        gate = "PASS ✓" if p < 0.01 else "FAIL ✗"
        print(f"\n  KW gate (fwd-return distinctness, fwd={FWD}): {gate}  (p={p:.2e})")

    # ── per-state forward returns on TEST ─────────────────────────────────────
    print(f"\n  TEST split — per-state forward returns (fwd={FWD} bars):")
    print(f"    {'Label':<16} {'N':>6}  {'Mean%':>8}  {'Med%':>8}  {'Pos%':>6}")
    print("    " + "─" * 50)
    te = sp.test
    fr = fwd_ret[te]
    st_te = states[te]
    valid = ~np.isnan(fr)
    for stn in unique_lbls:
        m = (st_te == stn) & valid
        if m.sum() < 10:
            continue
        r = fr[m]
        print(f"    {stn:<16} {len(r):>6,}  {r.mean()*100:>+7.3f}%  "
              f"{np.median(r)*100:>+7.3f}%  {(r>0).mean()*100:>5.1f}%")

    # ── save ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame({
        "timestamp":  d["ts"],
        "state":      states_int.astype(np.int8),   # raw HMM state ID (may carry dup labels)
        "state_name": states,                        # human label (may be non-unique)
    })
    out = CACHE / f"{ticker}_regime_hmm.parquet"
    df_out.to_parquet(out, index=False)
    print(f"\n  → {out.name}  ({len(df_out):,} rows)  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
