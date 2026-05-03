"""
Volatility prediction research module.

Explores all combinations of:
  Target types : realized_vol | price_range | atr
  Horizons     : 15, 30, 60, 100, 240 bars

For each combination trains a LightGBM regressor, evaluates on a
time-ordered val set, and prints a ranked comparison table.

Usage:
    python -m models.volatility
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from data.gaps import clean_mask
from models.splits import sequential

CACHE_DIR = Path(__file__).parent.parent / "cache"

HORIZONS    = [15, 30, 60, 100, 240]
TARGET_TYPES = ["realized_vol", "price_range", "atr"]


# ──────────────────────────────────────────────────────────────────────────────
# Target computation  (forward-looking — labels only, never used as features)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_targets(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all forward-looking volatility targets.
    Returns DataFrame aligned to meta index.
    All values are NaN for the last max(HORIZONS) rows (no future data).
    """
    price = meta["perp_ask_price"].values
    log_ret = np.diff(np.log(price), prepend=np.nan)

    targets = {}

    for H in HORIZONS:
        rv   = np.full(len(price), np.nan)
        rng  = np.full(len(price), np.nan)
        atr_ = np.full(len(price), np.nan)

        for i in range(len(price) - H):
            window_ret   = log_ret[i + 1 : i + H + 1]
            window_price = price[i + 1 : i + H + 1]

            rv[i]   = np.std(window_ret)
            rng[i]  = (window_price.max() - window_price.min()) / price[i]

            # ATR: mean of abs(close[t] - close[t-1]) over window — simple proxy
            atr_[i] = np.mean(np.abs(np.diff(window_price)))

        targets[f"realized_vol_{H}"]  = rv
        targets[f"price_range_{H}"]   = rng
        targets[f"atr_{H}"]           = atr_

    return pd.DataFrame(targets, index=meta.index)


# ──────────────────────────────────────────────────────────────────────────────
# Feature computation  (backward-looking only)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_features(meta: pd.DataFrame) -> pd.DataFrame:
    price    = meta["perp_ask_price"]
    log_ret  = np.log(price).diff()

    feats = pd.DataFrame(index=meta.index)

    # Current realized vol at multiple lookbacks
    for w in [5, 15, 30, 60, 240, 1440]:
        feats[f"rvol_{w}"]    = log_ret.rolling(w, min_periods=w).std()
        feats[f"ret_{w}"]     = log_ret.rolling(w, min_periods=w).sum()

    # Vol of vol
    feats["vol_of_vol_60"]  = feats["rvol_15"].rolling(60,  min_periods=60).std()
    feats["vol_of_vol_240"] = feats["rvol_30"].rolling(240, min_periods=240).std()

    # Vol ratio (term structure)
    feats["vol_ratio_15_60"]   = feats["rvol_15"]  / (feats["rvol_60"]   + 1e-12)
    feats["vol_ratio_30_240"]  = feats["rvol_30"]  / (feats["rvol_240"]  + 1e-12)

    # RSI (14)
    delta = log_ret
    gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    feats["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-12))

    # Bollinger band width
    ma20  = price.rolling(20, min_periods=20).mean()
    std20 = price.rolling(20, min_periods=20).std()
    feats["bb_width_20"] = (2 * std20) / (ma20 + 1e-12)

    # ATR proxy (using close-to-close)
    for w in [5, 14]:
        feats[f"atr_proxy_{w}"] = log_ret.abs().rolling(w, min_periods=w).mean()

    # OI and funding
    feats["oi_norm"]       = meta["oi_usd"] / meta["oi_usd"].rolling(1440, min_periods=60).mean()
    feats["oi_vel_15"]     = meta["oi_usd"].pct_change(15)
    feats["oi_vel_60"]     = meta["oi_usd"].pct_change(60)
    feats["fund_rate"]     = meta["fund_rate"]
    feats["fund_roll_8h"]  = meta["fund_rate"].rolling(480,  min_periods=1).mean()
    feats["fund_roll_24h"] = meta["fund_rate"].rolling(1440, min_periods=1).mean()

    # Taker imbalance
    taker_sum = meta["taker_buy"] + meta["taker_sell"] + 1e-12
    feats["taker_imb_1"]  = (meta["taker_buy"] - meta["taker_sell"]) / taker_sum
    for w in [5, 15, 30]:
        feats[f"taker_imb_{w}"] = (
            (meta["taker_buy"] - meta["taker_sell"]) / taker_sum
        ).rolling(w, min_periods=w).mean()

    # Volume z-score
    for w in [20, 60]:
        vm = meta["perp_minute_volume"].rolling(w, min_periods=w)
        feats[f"vol_z_{w}"] = (meta["perp_minute_volume"] - vm.mean()) / (vm.std() + 1e-12)

    # Spread and imbalance
    feats["perp_spread_bps"]  = meta["perp_spread_bps"]
    feats["perp_imbalance"]   = meta["perp_imbalance"]
    feats["spot_imbalance"]   = meta["spot_imbalance"]

    # Basis
    feats["basis_bps"] = (meta["perp_ask_price"] - meta["spot_ask_price"]) / meta["spot_ask_price"] * 10000
    feats["basis_z_60"]   = (feats["basis_bps"] - feats["basis_bps"].rolling(60,   min_periods=60).mean()) / \
                             (feats["basis_bps"].rolling(60,   min_periods=60).std() + 1e-12)
    feats["basis_z_240"]  = (feats["basis_bps"] - feats["basis_bps"].rolling(240,  min_periods=240).mean()) / \
                             (feats["basis_bps"].rolling(240,  min_periods=240).std() + 1e-12)

    # Calendar (sin/cos encoding)
    dt = pd.to_datetime(meta["timestamp"], unit="s")
    feats["hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour / 24)
    feats["hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour / 24)
    feats["dow_sin"]   = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    feats["dow_cos"]   = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Span (OB price range covered)
    feats["span_spot"] = meta["span_spot_price"]
    feats["span_perp"] = meta["span_perp_price"]

    return feats


# ──────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────────────────────────────────────

_LGB_PARAMS = {
    "objective":       "regression",
    "metric":          "rmse",
    "boosting_type":   "gbdt",
    "num_leaves":      64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "learning_rate":   0.05,
    "verbosity":       -1,
}


def _train_eval(X_train, y_train, X_val, y_val) -> dict:
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    model = lgb.train(
        _LGB_PARAMS,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    preds = model.predict(X_val, num_iteration=model.best_iteration)

    rmse  = np.sqrt(np.mean((preds - y_val) ** 2))
    mae   = np.mean(np.abs(preds - y_val))
    corr  = spearmanr(preds, y_val).statistic
    # Directional accuracy: does predicted high vol land in actual top tercile?
    tercile = np.percentile(y_val, 66.7)
    dir_acc = np.mean((preds > tercile) == (y_val > tercile))

    return {"model": model, "rmse": rmse, "mae": mae, "spearman": corr, "dir_acc": dir_acc}


# ──────────────────────────────────────────────────────────────────────────────
# Main research run
# ──────────────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc") -> pd.DataFrame:
    print(f"Loading {ticker} meta ...")
    meta = load_meta(ticker).reset_index(drop=True)

    print("Computing targets ...")
    targets = _compute_targets(meta)

    print("Computing features ...")
    feats = _compute_features(meta)

    # Drop rows with NaN features (rolling warmup) or NaN targets
    max_lookback = 1440
    mask = clean_mask(meta["timestamp"], max_lookback=max_lookback)
    mask &= feats.notna().all(axis=1).values

    split = sequential(mask.sum())
    clean_idx = np.where(mask)[0]
    train_idx = clean_idx[split.train]
    val_idx   = clean_idx[split.val]

    X_all   = feats.values
    feat_cols = feats.columns.tolist()

    results = []

    for ttype in TARGET_TYPES:
        for H in HORIZONS:
            col = f"{ttype}_{H}"
            y_all = targets[col].values

            # Additional mask: target must not be NaN
            valid = mask & ~np.isnan(y_all)
            clean_valid = np.where(valid)[0]

            # Re-split on valid-only indices
            sp = sequential(len(clean_valid))
            tr_idx  = clean_valid[sp.train]
            val_idx_ = clean_valid[sp.val]

            X_train, y_train = X_all[tr_idx],  y_all[tr_idx]
            X_val,   y_val   = X_all[val_idx_], y_all[val_idx_]

            res = _train_eval(X_train, y_train, X_val, y_val)

            print(f"  {ttype:<14}  H={H:<4}  "
                  f"RMSE={res['rmse']:.6f}  "
                  f"Spearman={res['spearman']:.3f}  "
                  f"DirAcc={res['dir_acc']:.3f}")

            results.append({
                "target_type": ttype,
                "horizon":     H,
                "rmse":        res["rmse"],
                "mae":         res["mae"],
                "spearman":    res["spearman"],
                "dir_acc":     res["dir_acc"],
            })

    df = pd.DataFrame(results).sort_values("spearman", ascending=False)

    print("\n── Ranked by Spearman correlation ──────────────────────────────")
    print(df.to_string(index=False))

    out = CACHE_DIR / f"{ticker}_volatility_research.parquet"
    df.to_parquet(out, index=False)
    print(f"\nResults saved → {out.name}")

    return df


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
