"""
Market and derivatives features: OI, funding rate, calendar, session flags.

Output: cache/{ticker}_features_market.parquet
Run   : python3 -m features.market [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_meta

CACHE_DIR = Path(__file__).parent.parent / "cache"


def compute(ticker: str, force: bool = False) -> pd.DataFrame:
    out = CACHE_DIR / f"{ticker}_features_market.parquet"
    if out.exists() and not force:
        print(f"Loading market features from cache: {out.name}")
        return pd.read_parquet(out)

    print(f"Computing market features for {ticker} ...")
    meta = load_meta(ticker)

    oi    = meta["oi_usd"]
    fr    = meta["fund_rate"]
    price = meta["perp_ask_price"]

    feats = pd.DataFrame(index=meta.index)
    feats["timestamp"] = meta["timestamp"].values

    # ── OI velocity and z-score ───────────────────────────────────────────────
    for w in [1, 5, 15, 60]:
        feats[f"oi_vel_{w}"] = oi.pct_change(w)
    for w in [60, 240, 1440]:
        mu  = oi.rolling(w, min_periods=w).mean()
        sig = oi.rolling(w, min_periods=w).std()
        feats[f"oi_z_{w}"] = (oi - mu) / (sig + 1e-12)

    # ── OI-price divergence: OI accelerating while price falls = squeeze risk ─
    feats["oi_price_div_15"] = oi.pct_change(15) - np.log(price).diff(15)
    feats["oi_price_div_60"] = oi.pct_change(60) - np.log(price).diff(60)

    # ── funding rate ──────────────────────────────────────────────────────────
    feats["fund_rate"] = fr
    for w in [480, 1440]:
        feats[f"fund_mean_{w}"] = fr.rolling(w, min_periods=1).mean()
        feats[f"fund_std_{w}"]  = fr.rolling(w, min_periods=w).std()
    feats["fund_mom_480"]  = fr - fr.shift(480)
    feats["fund_mom_1440"] = fr - fr.shift(1440)

    # ── spread and imbalance (raw microstructure state) ───────────────────────
    feats["perp_spread_bps"] = meta["perp_spread_bps"]
    feats["perp_imbalance"]  = meta["perp_imbalance"]
    feats["spot_imbalance"]  = meta["spot_imbalance"]
    for w in [20, 60]:
        mu  = meta["perp_spread_bps"].rolling(w, min_periods=w).mean()
        sig = meta["perp_spread_bps"].rolling(w, min_periods=w).std()
        feats[f"spread_z_{w}"] = (meta["perp_spread_bps"] - mu) / (sig + 1e-12)

    # ── calendar (sin/cos encoding — never raw integers) ─────────────────────
    dt = pd.to_datetime(meta["timestamp"], unit="s")
    feats["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    feats["dow_sin"]  = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    feats["dow_cos"]  = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    feats["min_sin"]  = np.sin(2 * np.pi * dt.dt.minute / 60)
    feats["min_cos"]  = np.cos(2 * np.pi * dt.dt.minute / 60)

    # ── session flags (UTC) ───────────────────────────────────────────────────
    h = dt.dt.hour
    feats["session_asia"]   = ((h >= 0)  & (h < 8)).astype(np.float32)
    feats["session_london"] = ((h >= 7)  & (h < 16)).astype(np.float32)
    feats["session_ny"]     = ((h >= 13) & (h < 21)).astype(np.float32)

    CACHE_DIR.mkdir(exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  Saved {out.name}  ({out.stat().st_size // 1024:,} KB)  shape={feats.shape}")
    return feats


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    df = compute(ticker, force=True)
    print(df.shape)
    print(df.head(3).to_string())
