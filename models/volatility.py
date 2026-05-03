"""
Volatility prediction model — uses assembled feature pipeline.

Targets  : realized_vol | price_range | atr
Horizons : 15, 30, 60, 100, 240 bars
Models   : LightGBM regressor + quantile (75th, 90th pct) on best targets

Evaluation per model:
  - Train on train split
  - Analyse on val: Spearman, RMSE, directional accuracy, top features
  - Analyse on test: same metrics, compare val vs test gap
  - Walk-forward: 6 folds, report per-fold Spearman

Run: python3 -m models.volatility [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from models.splits import sequential, walk_forward
from features.assembly import assemble

CACHE_DIR = Path(__file__).parent.parent / "cache"

HORIZONS     = [15, 30, 60, 100, 240]
TARGET_TYPES = ["realized_vol", "price_range", "atr"]
QUANTILE_TARGETS = [("atr", 15), ("atr", 30), ("realized_vol", 15)]


# ── label computation ─────────────────────────────────────────────────────────

def _compute_targets(price_arr: np.ndarray) -> dict[str, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price_arr)
    targets = {}
    for H in HORIZONS:
        rv  = np.full(n, np.nan)
        rng = np.full(n, np.nan)
        atr = np.full(n, np.nan)

        if n > H:
            wins = sliding_window_view(price_arr[1:], H)   # (n-H, H)
            log_wins = np.log(sliding_window_view(price_arr, H + 1))  # (n-H, H+1)

            rv[:n - H]  = np.std(np.diff(log_wins, axis=1), axis=1)
            rng[:n - H] = (wins.max(axis=1) - wins.min(axis=1)) / price_arr[:n - H]
            atr[:n - H] = np.mean(np.abs(np.diff(wins, axis=1)), axis=1)

        targets[f"realized_vol_{H}"] = rv
        targets[f"price_range_{H}"]  = rng
        targets[f"atr_{H}"]          = atr
    return targets


# ── metrics ───────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse    = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae     = np.mean(np.abs(y_pred - y_true))
    corr    = spearmanr(y_pred, y_true).statistic
    cutoff  = np.percentile(y_true, 66.7)
    dir_acc = np.mean((y_pred > cutoff) == (y_true > cutoff))
    return {"rmse": rmse, "mae": mae, "spearman": corr, "dir_acc": dir_acc}


# ── LightGBM helpers ──────────────────────────────────────────────────────────

def _lgb_params(objective="regression", alpha=None):
    p = {
        "objective":        objective,
        "metric":           "quantile" if objective == "quantile" else "rmse",
        "alpha":            alpha if alpha else 0.5,
        "boosting_type":    "gbdt",
        "num_leaves":       64,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "learning_rate":    0.05,
        "verbosity":        -1,
    }
    return p


def _train(X_tr, y_tr, X_val, y_val, params):
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model  = lgb.train(
        params, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


# ── print analysis ────────────────────────────────────────────────────────────

def _print_eval(tag: str, m: dict):
    print(f"  {tag:<10}  Spearman={m['spearman']:+.3f}  "
          f"RMSE={m['rmse']:.6f}  DirAcc={m['dir_acc']:.3f}")


def _print_top_features(model, feature_cols, n=15):
    imp = pd.Series(model.feature_importance("gain"), index=feature_cols)
    top = imp.nlargest(n)
    print(f"\n  Top {n} features:")
    for feat, score in top.items():
        print(f"    {feat:<45}  {score:>10.1f}")


# ── main research run ─────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*60}")
    print(f"  VOLATILITY MODEL — {ticker.upper()}")
    print(f"{'='*60}\n")

    # features
    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    # labels — from full meta aligned to assembled timestamps
    meta   = load_meta(ticker)
    ts_map = dict(zip(meta["timestamp"].values, range(len(meta))))
    price  = meta["perp_ask_price"].values
    all_targets = _compute_targets(price)

    def _get_labels(ts_arr, target_col):
        idxs = np.array([ts_map[t] for t in ts_arr])
        return all_targets[target_col][idxs]

    rows = []

    # ── 1. regressor sweep: all target × horizon combinations ────────────────
    print("── Regressor sweep ──────────────────────────────────────────────")
    for ttype in TARGET_TYPES:
        for H in HORIZONS:
            col = f"{ttype}_{H}"
            y_tr  = _get_labels(ts_train, col)
            y_val = _get_labels(ts_val,   col)
            y_te  = _get_labels(ts_test,  col)

            # drop NaN labels (last H rows have no forward data)
            ok_tr  = ~np.isnan(y_tr);  ok_val = ~np.isnan(y_val);  ok_te = ~np.isnan(y_te)

            model = _train(X_train[ok_tr], y_tr[ok_tr],
                           X_val[ok_val],  y_val[ok_val],
                           _lgb_params())

            m_val  = _metrics(y_val[ok_val], model.predict(X_val[ok_val]))
            m_test = _metrics(y_te[ok_te],   model.predict(X_test[ok_te]))

            print(f"\n  {col}")
            _print_eval("val",  m_val)
            _print_eval("test", m_test)
            if abs(m_val["spearman"] - m_test["spearman"]) > 0.1:
                print(f"  ⚠  val/test Spearman gap: "
                      f"{m_val['spearman']:.3f} vs {m_test['spearman']:.3f}")

            rows.append({**{"target": col, "split": "val"},  **m_val})
            rows.append({**{"target": col, "split": "test"}, **m_test})

    # ── 2. quantile regression on best targets ────────────────────────────────
    print("\n── Quantile regression (75th / 90th pct) ────────────────────────")
    for ttype, H in QUANTILE_TARGETS:
        col  = f"{ttype}_{H}"
        y_tr  = _get_labels(ts_train, col)
        y_val_ = _get_labels(ts_val,   col)
        y_te  = _get_labels(ts_test,  col)
        ok_tr  = ~np.isnan(y_tr)
        ok_val = ~np.isnan(y_val_)
        ok_te  = ~np.isnan(y_te)

        for alpha in [0.75, 0.90]:
            model = _train(X_train[ok_tr], y_tr[ok_tr],
                           X_val[ok_val],  y_val_[ok_val],
                           _lgb_params("quantile", alpha))
            preds_val  = model.predict(X_val[ok_val])
            preds_test = model.predict(X_test[ok_te])

            # coverage: pct of actuals below predicted quantile
            cov_val  = np.mean(y_val_[ok_val]  <= preds_val)
            cov_test = np.mean(y_te[ok_te] <= preds_test)
            corr_val  = spearmanr(preds_val,  y_val_[ok_val]).statistic
            corr_test = spearmanr(preds_test, y_te[ok_te]).statistic
            print(f"\n  {col}  alpha={alpha}")
            print(f"  val   Spearman={corr_val:+.3f}  Coverage={cov_val:.3f}  (target={alpha})")
            print(f"  test  Spearman={corr_test:+.3f}  Coverage={cov_test:.3f}  (target={alpha})")

    # ── 3. walk-forward on best combination ───────────────────────────────────
    print("\n── Walk-forward: atr_15 ─────────────────────────────────────────")
    all_ts = np.concatenate([ts_train, ts_val, ts_test])
    # rebuild full clean feature matrix for walk-forward
    _, _, _, _, _, _, _ = assemble(ticker)   # already cached
    assembled = pd.read_parquet(CACHE_DIR / f"{ticker}_features_assembled.parquet")
    feat_cols_all = [c for c in assembled.columns if c != "timestamp"]

    from data.gaps import clean_mask as _cmask
    meta_ts = meta["timestamp"].values
    gap_ok  = _cmask(pd.Series(meta_ts), max_lookback=1440)
    X_all   = assembled[feat_cols_all].values
    ts_all  = assembled["timestamp"].values
    row_ok  = gap_ok & ~np.isnan(X_all).any(axis=1)
    X_clean = X_all[row_ok]
    ts_clean = ts_all[row_ok]

    folds   = walk_forward(ts_clean)
    y_full  = _get_labels(ts_clean, "atr_15")

    fmt = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    print(f"  {'Fold':>4}  {'Train':>12}  {'Test':>12}  {'Spearman':>10}  {'DirAcc':>8}")
    print(f"  {'-'*55}")
    for fold in folds:
        ok_tr = ~np.isnan(y_full[fold.train])
        ok_te = ~np.isnan(y_full[fold.test])
        if ok_tr.sum() < 100 or ok_te.sum() < 10:
            continue
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_clean[fold.train][ok_tr])
        Xte = sc.transform(X_clean[fold.test][ok_te])
        ytr = y_full[fold.train][ok_tr]
        yte = y_full[fold.test][ok_te]
        m   = _train(Xtr, ytr, Xte, yte, _lgb_params())
        met = _metrics(yte, m.predict(Xte))
        print(f"  {fold.fold_idx:>4}  "
              f"{fmt(ts_clean[fold.train[0]]):>12} → {fmt(ts_clean[fold.train[-1]]):>12}  "
              f"test {fmt(ts_clean[fold.test[0]]):>12}  "
              f"{met['spearman']:>+.3f}  {met['dir_acc']:>7.3f}")

    # ── 4. top features for best model (atr_15 regressor) ────────────────────
    print("\n── Feature importance: atr_15 (val) ─────────────────────────────")
    col   = "atr_15"
    y_tr  = _get_labels(ts_train, col);  ok_tr  = ~np.isnan(y_tr)
    y_v   = _get_labels(ts_val,   col);  ok_val = ~np.isnan(y_v)
    best_model = _train(X_train[ok_tr], y_tr[ok_tr],
                        X_val[ok_val],  y_v[ok_val], _lgb_params())
    _print_top_features(best_model, feat_cols)

    # save results
    df = pd.DataFrame(rows)
    out = CACHE_DIR / f"{ticker}_volatility_eval.parquet"
    df.to_parquet(out, index=False)
    print(f"\nResults → {out.name}")

    print("\n── Summary table (Spearman, ranked) ─────────────────────────────")
    pivot = df.pivot(index="target", columns="split", values="spearman").sort_values("val", ascending=False)
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
