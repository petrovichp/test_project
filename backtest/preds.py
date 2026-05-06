"""
Vol, direction, and regime prediction caching.

`_vol_preds` runs the LightGBM ATR model on all 3 splits, computes percentile rank,
and caches the result to `cache/{ticker}_pred_vol.npz`.

`_dir_preds` runs the 4 CNN-LSTM direction models and caches each to
`cache/{ticker}_pred_dir_{direction}_{H}.npz`.

`_regime_preds` aligns regime labels to val/test timestamps and caches the result
to `cache/{ticker}_{regime_file}_preds.npz`.

Cache invalidation: based on file mtime of the source model artefact.
"""

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.direction_dl import SEQ_FEATURES, SEQ_LEN, HORIZONS

CACHE_DIR = Path(__file__).parent.parent / "cache"
VOL_MODEL = "atr_30"


def _vol_preds(ticker, X_sc_tr, X_sc_v, X_sc_te):
    model_path = CACHE_DIR / f"{ticker}_lgbm_{VOL_MODEL}.txt"
    cache_path = CACHE_DIR / f"{ticker}_pred_vol.npz"

    if cache_path.exists() and cache_path.stat().st_mtime >= model_path.stat().st_mtime:
        d = np.load(cache_path)
        print(f"  [cache] vol predictions loaded from {cache_path.name}")
        return d["atr_tr"], d["atr_v"], d["atr_te"], d["rank_tr"], d["rank_v"], d["rank_te"]

    vol    = lgb.Booster(model_file=str(model_path))
    atr_tr = vol.predict(X_sc_tr)
    atr_v  = vol.predict(X_sc_v)
    atr_te = vol.predict(X_sc_te)
    sorted_tr = np.sort(atr_tr)
    rank_tr   = np.argsort(np.argsort(atr_tr)) / len(atr_tr)
    rank_v    = np.clip(np.searchsorted(sorted_tr, atr_v)  / len(sorted_tr), 0, 1)
    rank_te   = np.clip(np.searchsorted(sorted_tr, atr_te) / len(sorted_tr), 0, 1)
    np.savez(cache_path, atr_tr=atr_tr, atr_v=atr_v, atr_te=atr_te,
             rank_tr=rank_tr, rank_v=rank_v, rank_te=rank_te)
    print(f"  [cache] vol predictions saved to {cache_path.name}")
    return atr_tr, atr_v, atr_te, rank_tr, rank_v, rank_te


def _dir_preds(ticker, X_sc_tr, X_sc_v, X_sc_te, feat_cols, rank_tr, rank_v, rank_te):
    sel = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    result = {}
    for H in HORIZONS:
        for direction in ["up", "down"]:
            col        = f"{direction}_{H}"
            model_path = CACHE_DIR / f"{ticker}_cnn2s_dir_{direction}_{H}.keras"
            cache_path = CACHE_DIR / f"{ticker}_pred_dir_{col}.npz"

            if cache_path.exists() and model_path.exists() and \
               cache_path.stat().st_mtime >= model_path.stat().st_mtime:
                d = np.load(cache_path)
                result[col] = (d["tr"], d["val"], d["te"])
                print(f"  [cache] dir {col} loaded from {cache_path.name}")
                continue

            if not model_path.exists():
                result[col] = (np.full(len(X_sc_tr), 0.5),
                               np.full(len(X_sc_v),  0.5),
                               np.full(len(X_sc_te), 0.5))
                continue

            model = tf.keras.models.load_model(str(model_path))
            def _pred(X, rank):
                n = len(X); probs = np.full(n, 0.5)
                if n > SEQ_LEN:
                    idx = np.arange(SEQ_LEN, n)
                    Xs  = np.stack([X[i - SEQ_LEN:i] for i in idx])
                    Xs  = np.concatenate([Xs, np.tile(rank[idx, None, None], (1, SEQ_LEN, 1))], axis=2)
                    probs[idx] = model.predict(Xs, verbose=0).flatten()
                return probs
            tr  = _pred(X_sc_tr[:, sel], rank_tr)
            val = _pred(X_sc_v[:,  sel], rank_v)
            te  = _pred(X_sc_te[:, sel], rank_te)
            np.savez(cache_path, tr=tr, val=val, te=te)
            print(f"  [cache] dir {col} saved to {cache_path.name}")
            result[col] = (tr, val, te)
    return result


def _regime_preds(ticker: str, ts_v: np.ndarray, ts_te: np.ndarray,
                   regime_file: str = "regime_cusum"):
    """Align regime labels to val/test timestamps. Cache to .npz.

    Returns (regime_v, regime_te) — string arrays of state names per bar.
    Returns (None, None) if the source parquet doesn't exist.
    Cache is invalidated when source parquet mtime changes OR the timestamp
    arrays don't match the cached ones (data refresh case).
    """
    src   = CACHE_DIR / f"{ticker}_{regime_file}.parquet"
    cache = CACHE_DIR / f"{ticker}_{regime_file}_preds.npz"

    if not src.exists():
        return None, None

    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        d = np.load(cache, allow_pickle=False)
        if (d["ts_v"].shape == ts_v.shape and np.array_equal(d["ts_v"], ts_v) and
            d["ts_te"].shape == ts_te.shape and np.array_equal(d["ts_te"], ts_te)):
            print(f"  [cache] regime preds loaded from {cache.name}")
            return d["regime_v"].astype(str), d["regime_te"].astype(str)

    rdf         = pd.read_parquet(src)
    ts_to_state = dict(zip(rdf["timestamp"].values, rdf["state_name"].values))
    regime_v    = np.array([ts_to_state.get(t, "unknown") for t in ts_v])
    regime_te   = np.array([ts_to_state.get(t, "unknown") for t in ts_te])
    np.savez(cache, regime_v=regime_v, regime_te=regime_te, ts_v=ts_v, ts_te=ts_te)
    print(f"  [cache] regime preds saved to {cache.name}")
    return regime_v, regime_te
