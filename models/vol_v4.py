"""
Vol model v4 — LightGBM ATR-30 retrained on vol-train (bars 1,440 → 101,440).

DQN-v5 variant: ignores `clean_mask`. Treats the 384k feature parquet as a
continuous index sequence; only the first 1,440 warmup rows are dropped.

Outputs:
  cache/btc_lgbm_atr_30_v4.txt   — model
  cache/btc_pred_vol_v4.npz      — atr predictions + rank (relative to vol-train)
                                    arrays cover bars 1,440 → 384,614

Run: python3 -m models.vol_v4 [ticker]
"""

import sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import load_meta

CACHE       = ROOT / "cache"
WARMUP      = 1440
VOL_TRAIN_E = 101_440           # exclusive
HORIZON     = 30                # atr_30
LGB_PARAMS  = {
    "objective":        "regression",
    "metric":           "rmse",
    "boosting_type":    "gbdt",
    "num_leaves":       64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "learning_rate":    0.05,
    "verbosity":        -1,
}


def _atr_target(price: np.ndarray, H: int = HORIZON) -> np.ndarray:
    """ATR-H = mean(|diff(price)|) over next H bars; NaN at the tail."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price)
    out = np.full(n, np.nan)
    if n > H:
        wins = sliding_window_view(price[1:], H)
        out[: n - H] = np.mean(np.abs(np.diff(wins, axis=1)), axis=1)
    return out


def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  VOL v4 (ATR-30, no clean_mask) — {ticker.upper()}\n{'='*70}")

    pq        = pd.read_parquet(CACHE / "features" / f"{ticker}_features_assembled.parquet")
    feat_cols = [c for c in pq.columns if c != "timestamp"]
    X_raw     = pq[feat_cols].values
    ts_arr    = pq["timestamp"].values

    meta      = load_meta(ticker)
    assert (meta["timestamp"].values == ts_arr).all(), "meta/features misaligned"
    price_arr = meta["perp_ask_price"].values

    # NaN beyond warmup must be 0
    nan_post = np.isnan(X_raw[WARMUP:]).any(axis=1).sum()
    print(f"  feat parquet shape={X_raw.shape}  NaN beyond warmup={nan_post}")
    assert nan_post == 0

    # ── target ───────────────────────────────────────────────────────────────
    y = _atr_target(price_arr, H=HORIZON)
    print(f"  ATR-{HORIZON} target NaN tail = {np.isnan(y).sum()} (expected {HORIZON})")

    # ── slice ────────────────────────────────────────────────────────────────
    tr_idx = np.arange(WARMUP, VOL_TRAIN_E)
    rl_idx = np.arange(VOL_TRAIN_E, len(X_raw))

    # drop NaN target rows from train (last HORIZON rows of train at most)
    ok_tr  = ~np.isnan(y[tr_idx])
    X_tr   = X_raw[tr_idx][ok_tr]
    y_tr   = y[tr_idx][ok_tr]
    print(f"  vol-train usable rows: {ok_tr.sum():,} / {len(tr_idx):,}")

    # tiny holdout from tail of vol-train for early stopping (last 5%)
    n_tr   = len(X_tr)
    n_es   = int(n_tr * 0.05)
    X_es   = X_tr[-n_es:];  y_es   = y_tr[-n_es:]
    X_fit  = X_tr[:-n_es];  y_fit  = y_tr[:-n_es]

    # scaler on fit-portion only
    scaler   = StandardScaler()
    X_fit_sc = scaler.fit_transform(X_fit)
    X_es_sc  = scaler.transform(X_es)
    X_rl_sc  = scaler.transform(X_raw[rl_idx])           # no NaN past warmup
    X_tr_sc  = scaler.transform(X_tr)                     # for in-sample rank

    # ── train ────────────────────────────────────────────────────────────────
    print(f"  fitting LightGBM ATR-{HORIZON} ...")
    t1 = time.perf_counter()
    ds_fit = lgb.Dataset(X_fit_sc, label=y_fit)
    ds_es  = lgb.Dataset(X_es_sc,  label=y_es, reference=ds_fit)
    model  = lgb.train(
        LGB_PARAMS, ds_fit, num_boost_round=500, valid_sets=[ds_es],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    model.save_model(str(CACHE / "preds" / f"{ticker}_lgbm_atr_{HORIZON}_v4.txt"))
    print(f"    fit done in {time.perf_counter()-t1:.1f}s "
          f"(best_iter={model.best_iteration})")

    # ── predict + spearman + rank ────────────────────────────────────────────
    pred_tr = model.predict(X_tr_sc)
    pred_rl = model.predict(X_rl_sc)

    sp_tr   = spearmanr(pred_tr, y_tr).statistic
    print(f"  Spearman vol-train (in-sample): {sp_tr:+.3f}")

    # informational test spearman: bars whose ATR target is fully realized
    y_rl    = y[rl_idx]
    ok_rl   = ~np.isnan(y_rl)
    if ok_rl.sum() > 100:
        sp_rl = spearmanr(pred_rl[ok_rl], y_rl[ok_rl]).statistic
        print(f"  Spearman RL period (out-of-sample): {sp_rl:+.3f}")
    else:
        sp_rl = float("nan")

    # rank: vol-train sorted, RL bars searchsorted into it
    sorted_tr = np.sort(pred_tr)
    rank_tr   = np.argsort(np.argsort(pred_tr)) / len(pred_tr)
    rank_rl   = np.clip(np.searchsorted(sorted_tr, pred_rl) / len(sorted_tr), 0, 1)

    # ── stitch full arrays (bars 1,440 → 384,614) ────────────────────────────
    # Entire RL has predictions; vol-train has predictions on its `ok_tr` mask
    n_full        = len(X_raw) - WARMUP                  # 383,174
    full_atr      = np.full(n_full, np.nan, dtype=np.float32)
    full_rank     = np.full(n_full, np.nan, dtype=np.float32)

    # vol-train slice: indices 0 → (VOL_TRAIN_E - WARMUP) = 100,000
    n_vt_full     = VOL_TRAIN_E - WARMUP
    full_atr[:n_vt_full][ok_tr]  = pred_tr.astype(np.float32)
    full_rank[:n_vt_full][ok_tr] = rank_tr.astype(np.float32)

    # RL slice
    full_atr[n_vt_full:]  = pred_rl.astype(np.float32)
    full_rank[n_vt_full:] = rank_rl.astype(np.float32)

    np.savez(
        CACHE / "preds" / f"{ticker}_pred_vol_v4.npz",
        atr        = full_atr,
        rank       = full_rank,
        ts         = ts_arr[WARMUP:].astype(np.int64),
        warmup     = WARMUP,
        vol_train_e= VOL_TRAIN_E,
        spearman_in= sp_tr,
        spearman_oos = sp_rl,
        atr_train_median = float(np.median(pred_tr)),
        atr_train_iqr    = float(np.percentile(pred_tr, 75) - np.percentile(pred_tr, 25)),
    )
    print(f"  → cache/{ticker}_pred_vol_v4.npz  total {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
