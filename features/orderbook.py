"""
Orderbook features from the 800-column OB snapshot.

Features per row:
  - 12 bucket amounts  : spot/perp × bids/asks × [0-50, 50-100, 100-200]
  - 6  bucket imbalances: (bid-ask)/(bid+ask) per bucket per instrument
  - 12 bucket velocities: 1-bar diff of bucket amounts
  - 2  span scalars    : span_spot_price, span_perp_price

Output: cache/{ticker}_features_ob.parquet
Run   : python3 -m features.orderbook [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load, load_meta

CACHE_DIR = Path(__file__).parent.parent / "cache"
BUCKETS = [(0, 50), (50, 100), (100, 200)]
SIDES = ["bids", "asks"]
INSTRUMENTS = ["spot", "perp"]


def _bucket_cols(inst: str, side: str, lo: int, hi: int) -> list[str]:
    return [f"{inst}_{side}_amount_{i}" for i in range(lo, hi)]


def compute(ticker: str, force: bool = False) -> pd.DataFrame:
    out = CACHE_DIR / f"{ticker}_features_ob.parquet"
    if out.exists() and not force:
        print(f"Loading OB features from cache: {out.name}")
        return pd.read_parquet(out)

    print(f"Computing OB features for {ticker} ...")
    meta, ob = load(ticker, include_ob=True)

    feats = pd.DataFrame(index=meta.index)
    feats["timestamp"] = meta["timestamp"].values

    # ── bucket amounts ────────────────────────────────────────────────────────
    buckets: dict[str, pd.Series] = {}
    for inst in INSTRUMENTS:
        for side in SIDES:
            for lo, hi in BUCKETS:
                key = f"{inst}_{side}_{lo}_{hi}"
                cols = _bucket_cols(inst, side, lo, hi)
                buckets[key] = ob[cols].sum(axis=1)
                feats[f"ob_amt_{key}"] = buckets[key]

    # ── per-bucket imbalance: (bid - ask) / (bid + ask) ──────────────────────
    for inst in INSTRUMENTS:
        for lo, hi in BUCKETS:
            bid = buckets[f"{inst}_bids_{lo}_{hi}"]
            ask = buckets[f"{inst}_asks_{lo}_{hi}"]
            denom = bid + ask
            feats[f"ob_imb_{inst}_{lo}_{hi}"] = (bid - ask) / denom.replace(0, np.nan)

    # ── OB velocity: 1-bar diff of bucket amounts ─────────────────────────────
    for inst in INSTRUMENTS:
        for side in SIDES:
            for lo, hi in BUCKETS:
                key = f"ob_amt_{inst}_{side}_{lo}_{hi}"
                feats[f"ob_vel_{inst}_{side}_{lo}_{hi}"] = feats[key].diff(1)

    # ── span scalars ──────────────────────────────────────────────────────────
    feats["span_spot"] = meta["span_spot_price"].values
    feats["span_perp"] = meta["span_perp_price"].values

    CACHE_DIR.mkdir(exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  Saved {out.name}  ({out.stat().st_size // 1024:,} KB)  shape={feats.shape}")
    return feats


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    df = compute(ticker, force=True)
    print(df.shape)
    print(df.head(3).to_string())
