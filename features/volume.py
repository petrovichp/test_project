"""
Volume and taker flow features.

Features: taker imbalance, cumulative taker net, volume z-score, OBV, OFI proxy.

Output: cache/{ticker}_features_volume.parquet
Run   : python3 -m features.volume [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_meta

CACHE_DIR = Path(__file__).parent.parent / "cache"


def compute(ticker: str, force: bool = False) -> pd.DataFrame:
    out = CACHE_DIR / f"{ticker}_features_volume.parquet"
    if out.exists() and not force:
        print(f"Loading volume features from cache: {out.name}")
        return pd.read_parquet(out)

    print(f"Computing volume features for {ticker} ...")
    meta = load_meta(ticker)

    buy  = meta["taker_buy"]
    sell = meta["taker_sell"]
    vol_spot = meta["spot_minute_volume"]
    vol_perp = meta["perp_minute_volume"]

    feats = pd.DataFrame(index=meta.index)
    feats["timestamp"] = meta["timestamp"].values

    # ── taker imbalance at multiple horizons ─────────────────────────────────
    raw_imb = (buy - sell) / (buy + sell + 1e-12)
    feats["taker_imb_1"] = raw_imb
    for w in [5, 15, 30]:
        feats[f"taker_imb_{w}"] = raw_imb.rolling(w, min_periods=w).mean()

    # ── cumulative taker net (rolling sum buy - sell) ─────────────────────────
    net = buy - sell
    for w in [5, 15, 30, 60]:
        feats[f"taker_net_{w}"] = net.rolling(w, min_periods=w).sum()

    # ── volume z-score ────────────────────────────────────────────────────────
    for col, name in [(vol_spot, "spot"), (vol_perp, "perp")]:
        for w in [20, 60]:
            mu  = col.rolling(w, min_periods=w).mean()
            sig = col.rolling(w, min_periods=w).std()
            feats[f"vol_z_{name}_{w}"] = (col - mu) / (sig + 1e-12)

    # ── OBV (on-balance volume, rolling reset every 1440 bars) ───────────────
    price   = meta["perp_ask_price"]
    ret_sign = np.sign(price.diff())
    signed_vol = ret_sign * vol_perp
    feats["obv_1440"] = signed_vol.rolling(1440, min_periods=1).sum()

    # ── OFI proxy: 1-bar change in perp orderbook imbalance ──────────────────
    feats["ofi_proxy"] = meta["perp_imbalance"].diff(1)

    # ── spot/perp volume ratio ────────────────────────────────────────────────
    feats["spot_perp_vol_ratio"] = vol_spot / (vol_perp + 1e-12)
    for w in [20, 60]:
        mu  = feats["spot_perp_vol_ratio"].rolling(w, min_periods=w).mean()
        sig = feats["spot_perp_vol_ratio"].rolling(w, min_periods=w).std()
        feats[f"spot_perp_vol_ratio_z_{w}"] = (feats["spot_perp_vol_ratio"] - mu) / (sig + 1e-12)

    CACHE_DIR.mkdir(exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  Saved {out.name}  ({out.stat().st_size // 1024:,} KB)  shape={feats.shape}")
    return feats


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    df = compute(ticker, force=True)
    print(df.shape)
    print(df.head(3).to_string())
