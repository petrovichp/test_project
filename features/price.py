"""
Price and momentum features.

Features: log returns, SMA/EMA ratios, RSI, MACD, ROC, Stochastic,
          Bollinger bands, VWAP, perp-spot basis.

Output: cache/{ticker}_features_price.parquet
Run   : python3 -m features.price [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_meta

CACHE_DIR = Path(__file__).parent.parent / "cache"


def compute(ticker: str, force: bool = False) -> pd.DataFrame:
    out = CACHE_DIR / "features" / f"{ticker}_features_price.parquet"
    if out.exists() and not force:
        print(f"Loading price features from cache: {out.name}")
        return pd.read_parquet(out)

    print(f"Computing price features for {ticker} ...")
    meta = load_meta(ticker)

    price  = meta["perp_ask_price"]
    spot   = meta["spot_ask_price"]
    vol_p  = meta["perp_minute_volume"]

    feats = pd.DataFrame(index=meta.index)
    feats["timestamp"] = meta["timestamp"].values

    log_ret = np.log(price).diff()

    # ── log returns at multiple horizons ─────────────────────────────────────
    for h in [1, 5, 15, 30, 60, 120, 240]:
        feats[f"ret_{h}"] = log_ret.rolling(h, min_periods=h).sum()

    # ── SMA and close/SMA ratio ───────────────────────────────────────────────
    for w in [5, 10, 20, 50, 200]:
        sma = price.rolling(w, min_periods=w).mean()
        feats[f"sma_{w}"]       = sma
        feats[f"ret_sma_{w}"]   = price / sma - 1

    # ── EMA and close/EMA ratio ───────────────────────────────────────────────
    for w in [5, 12, 26, 50, 200]:
        ema = price.ewm(span=w, adjust=False).mean()
        feats[f"ema_{w}"]       = ema
        feats[f"ret_ema_{w}"]   = price / ema - 1

    # ── RSI ───────────────────────────────────────────────────────────────────
    for period in [6, 14]:
        gain = log_ret.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-log_ret.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        feats[f"rsi_{period}"] = 100 - 100 / (1 + gain / (loss + 1e-12))

    # ── MACD (12/26/9) ────────────────────────────────────────────────────────
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd"]          = macd / (price + 1e-12)
    feats["macd_signal"]   = signal / (price + 1e-12)
    feats["macd_hist"]     = (macd - signal) / (price + 1e-12)

    # ── ROC ───────────────────────────────────────────────────────────────────
    for w in [5, 20]:
        feats[f"roc_{w}"] = price.pct_change(w)

    # ── Stochastic ────────────────────────────────────────────────────────────
    for w in [5, 14]:
        lo = price.rolling(w, min_periods=w).min()
        hi = price.rolling(w, min_periods=w).max()
        k  = (price - lo) / (hi - lo + 1e-12)
        feats[f"stoch_k_{w}"] = k
        feats[f"stoch_d_{w}"] = k.rolling(3, min_periods=3).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    ma20  = price.rolling(20, min_periods=20).mean()
    std20 = price.rolling(20, min_periods=20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    feats["bb_pct_b"]  = (price - lower) / (upper - lower + 1e-12)
    feats["bb_width"]  = (upper - lower) / (ma20 + 1e-12)

    # ── Rolling VWAP ─────────────────────────────────────────────────────────
    pv = price * vol_p
    for w in [60, 240, 1440]:
        vwap = pv.rolling(w, min_periods=w).sum() / vol_p.rolling(w, min_periods=w).sum()
        feats[f"vwap_{w}"]     = vwap
        feats[f"vwap_dev_{w}"] = price / vwap - 1

    # ── Perp-spot basis ───────────────────────────────────────────────────────
    basis = (price - spot) / spot * 10000
    feats["basis_bps"] = basis
    for w in [60, 240]:
        mu  = basis.rolling(w, min_periods=w).mean()
        sig = basis.rolling(w, min_periods=w).std()
        feats[f"basis_z_{w}"]  = (basis - mu) / (sig + 1e-12)
    for w in [5, 15]:
        feats[f"basis_mom_{w}"] = basis.diff(w)

    CACHE_DIR.mkdir(exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  Saved {out.name}  ({out.stat().st_size // 1024:,} KB)  shape={feats.shape}")
    return feats


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    df = compute(ticker, force=True)
    print(df.shape)
    print(df.head(3).to_string())
