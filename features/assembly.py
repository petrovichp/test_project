"""
Feature assembly — combines all feature modules into a single aligned DataFrame.

Applies clean_mask (gap-aware), drops NaN rows, and returns train/val/test splits.
Scaler is fit on train only and applied to val/test.

Usage:
    from features.assembly import assemble
    X_train, X_val, X_test, feature_cols = assemble("btc")

Run: python3 -m features.assembly [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load_meta
from data.gaps import clean_mask
from models.splits import sequential
from features.orderbook import compute as compute_ob
from features.price     import compute as compute_price
from features.volume    import compute as compute_volume
from features.market    import compute as compute_market

CACHE_DIR  = Path(__file__).parent.parent / "cache"
MAX_LOOKBACK = 1440   # longest rolling window used across all feature modules


def assemble(
    ticker:      str,
    force:       bool = False,
    train_frac:  float = 0.50,
    val_frac:    float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Build and return (X_train, X_val, X_test, feature_cols,
                      ts_train, ts_val, ts_test).

    All NaN rows and gap-contaminated rows are removed before splitting.
    Scaler is fit on train only.
    """
    cache_file = CACHE_DIR / f"{ticker}_features_assembled.parquet"

    if cache_file.exists() and not force:
        print(f"Loading assembled features from cache: {cache_file.name}")
        df = pd.read_parquet(cache_file)
    else:
        print(f"Assembling features for {ticker} ...")
        meta = load_meta(ticker)

        ob     = compute_ob(ticker)
        price  = compute_price(ticker)
        vol    = compute_volume(ticker)
        mkt    = compute_market(ticker)

        # Align all on timestamp (inner join via index — all share the same source index)
        df = pd.concat([
            ob.set_index("timestamp"),
            price.set_index("timestamp"),
            vol.set_index("timestamp"),
            mkt.set_index("timestamp"),
        ], axis=1)
        df.index.name = "timestamp"
        df = df.reset_index()

        df.to_parquet(cache_file, index=False)
        print(f"  Assembled → {cache_file.name}  shape={df.shape}")

    # ── apply gap mask and drop NaN rows ─────────────────────────────────────
    meta      = load_meta(ticker)
    ts        = meta["timestamp"].values
    gap_ok    = clean_mask(pd.Series(ts), max_lookback=MAX_LOOKBACK)

    feature_cols = [c for c in df.columns if c != "timestamp"]
    X_full    = df[feature_cols].values
    ts_full   = df["timestamp"].values

    row_ok    = gap_ok & ~np.isnan(X_full).any(axis=1)
    X_clean   = X_full[row_ok]
    ts_clean  = ts_full[row_ok]

    n = len(X_clean)
    sp = sequential(n, train_frac, val_frac)

    X_train = X_clean[sp.train]
    X_val   = X_clean[sp.val]
    X_test  = X_clean[sp.test]

    ts_train = ts_clean[sp.train]
    ts_val   = ts_clean[sp.val]
    ts_test  = ts_clean[sp.test]

    # ── scale: fit on train only ──────────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, feature_cols, ts_train, ts_val, ts_test


if __name__ == "__main__":
    from datetime import datetime
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"

    X_train, X_val, X_test, cols, ts_tr, ts_val, ts_te = assemble(ticker, force=True)

    fmt = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    print(f"\nFeature columns : {len(cols)}")
    print(f"Train  : {X_train.shape}  {fmt(ts_tr[0])} → {fmt(ts_tr[-1])}")
    print(f"Val    : {X_val.shape}    {fmt(ts_val[0])} → {fmt(ts_val[-1])}")
    print(f"Test   : {X_test.shape}   {fmt(ts_te[0])} → {fmt(ts_te[-1])}")
    print(f"\nSample features: {cols[:5]} ...")
    print(f"NaN in train   : {np.isnan(X_train).sum()}")
    print(f"NaN in val     : {np.isnan(X_val).sum()}")
    print(f"NaN in test    : {np.isnan(X_test).sum()}")
