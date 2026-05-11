"""
Direction prediction — LightGBM baseline.

Labels:
  Y_up   = max(price[t+1:t+H+1]) / price[t] - 1 > threshold
  Y_down = min(price[t+1:t+H+1]) / price[t] - 1 < -threshold

Trains separate binary classifiers for up and down across horizons
H ∈ {60, 100} and threshold = 0.008 (matching original notebook).

Evaluation per model:
  - AUC on val  → decision point before touching test
  - AUC on test → honest number, run once
  - Val/test gap check
  - Top features (SHAP-style LightGBM gain importance)
  - Walk-forward: 6 folds

Run: python3 -m models.direction [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from data.gaps import clean_mask as _cmask
from models.splits import sequential, walk_forward
from features.assembly import assemble

CACHE_DIR  = Path(__file__).parent.parent / "cache"
HORIZONS   = [60, 100]
THRESHOLD  = 0.008


# ── label computation ─────────────────────────────────────────────────────────

def _compute_labels(price_arr: np.ndarray) -> dict[str, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price_arr)
    labels = {}
    for H in HORIZONS:
        y_up   = np.full(n, np.nan)
        y_down = np.full(n, np.nan)
        if n > H:
            wins = sliding_window_view(price_arr[1:], H)   # (n-H, H)
            ret_max = wins.max(axis=1) / price_arr[:n - H] - 1
            ret_min = wins.min(axis=1) / price_arr[:n - H] - 1
            y_up[:n - H]   = (ret_max >  THRESHOLD).astype(float)
            y_down[:n - H] = (ret_min < -THRESHOLD).astype(float)
        labels[f"up_{H}"]   = y_up
        labels[f"down_{H}"] = y_down
    return labels


# ── LightGBM ──────────────────────────────────────────────────────────────────

_LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "boosting_type":    "gbdt",
    "num_leaves":       64,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "learning_rate":    0.05,
    "verbosity":        -1,
}


def _model_path(ticker: str, label: str) -> Path:
    return CACHE_DIR / "preds" / f"{ticker}_direction_lgbm_{label}.txt"


def _train(X_tr, y_tr, X_val, y_val, save_path: Path = None):
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model = lgb.train(
        _LGB_PARAMS, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    if save_path:
        model.save_model(str(save_path))
    return model


def _load_or_train(X_tr, y_tr, X_val, y_val, save_path: Path, force: bool = False):
    if save_path.exists() and not force:
        print(f"  Loading cached model: {save_path.name}")
        return lgb.Booster(model_file=str(save_path))
    return _train(X_tr, y_tr, X_val, y_val, save_path)


def _auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


# ── analysis helpers ──────────────────────────────────────────────────────────

def _print_top_features(model, feature_cols, n=20):
    imp = pd.Series(model.feature_importance("gain"), index=feature_cols)
    top = imp.nlargest(n)
    print(f"\n  Top {n} features by gain:")
    for feat, score in top.items():
        print(f"    {feat:<45}  {score:>12.1f}")


def _calibration_check(y_true, y_prob, n_bins=5):
    bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    print(f"\n  Calibration (predicted → actual win rate):")
    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        pred_mean = y_prob[mask].mean()
        actual    = y_true[mask].mean()
        print(f"    pred {pred_mean:.3f} → actual {actual:.3f}  (n={mask.sum():,})")


# ── main run ──────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*60}")
    print(f"  DIRECTION MODEL — {ticker.upper()}")
    print(f"{'='*60}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta   = load_meta(ticker)
    ts_map = dict(zip(meta["timestamp"].values, range(len(meta))))
    price  = meta["perp_ask_price"].values
    all_labels = _compute_labels(price)

    def _get_y(ts_arr, label_col):
        idxs = np.array([ts_map[t] for t in ts_arr])
        return all_labels[label_col][idxs]

    fmt  = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    rows = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n{'─'*55}")
            print(f"  H={H}  direction={direction}  threshold={THRESHOLD*100:.1f}%")
            print(f"{'─'*55}")

            y_tr  = _get_y(ts_train, col);  ok_tr  = ~np.isnan(y_tr)
            y_v   = _get_y(ts_val,   col);  ok_val = ~np.isnan(y_v)
            y_te  = _get_y(ts_test,  col);  ok_te  = ~np.isnan(y_te)

            pos_rate_tr = y_tr[ok_tr].mean()
            print(f"  Train positives: {pos_rate_tr:.1%}  "
                  f"({ok_tr.sum():,} rows)  "
                  f"{fmt(ts_train[0])} → {fmt(ts_train[-1])}")

            # ── STEP 1: train (or load cached) ───────────────────────────────
            model = _load_or_train(
                X_train[ok_tr], y_tr[ok_tr],
                X_val[ok_val],  y_v[ok_val],
                save_path=_model_path(ticker, col),
            )

            # ── STEP 2: val analysis ──────────────────────────────────────────
            prob_val = model.predict(X_val[ok_val])
            auc_val  = _auc(y_v[ok_val], prob_val)
            print(f"\n  [VAL]   AUC={auc_val:.4f}  "
                  f"({fmt(ts_val[0])} → {fmt(ts_val[-1])})")
            _calibration_check(y_v[ok_val], prob_val)

            gate = auc_val > 0.52
            print(f"\n  Gate (AUC > 0.52): {'PASS ✓' if gate else 'FAIL ✗'}")

            # ── STEP 3: test (run once) ────────────────────────────────────────
            prob_test = model.predict(X_test[ok_te])
            auc_test  = _auc(y_te[ok_te], prob_test)
            print(f"\n  [TEST]  AUC={auc_test:.4f}  "
                  f"({fmt(ts_test[0])} → {fmt(ts_test[-1])})")
            _calibration_check(y_te[ok_te], prob_test)

            gap = abs(auc_val - auc_test)
            if gap > 0.03:
                print(f"\n  ⚠  val/test AUC gap = {gap:.3f} — possible overfit to val period")
            else:
                print(f"\n  ✓  val/test AUC gap = {gap:.3f} — consistent")

            # ── feature importance ────────────────────────────────────────────
            _print_top_features(model, feat_cols)

            rows.append({"label": col, "split": "val",  "auc": auc_val,  "gate": gate})
            rows.append({"label": col, "split": "test", "auc": auc_test, "gate": gate})

    # ── walk-forward on best label ─────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f"  WALK-FORWARD — up_60")
    print(f"{'='*60}")

    assembled = pd.read_parquet(CACHE_DIR / "features" / f"{ticker}_features_assembled.parquet")
    fc_all    = [c for c in assembled.columns if c != "timestamp"]
    meta_ts   = meta["timestamp"].values
    gap_ok    = _cmask(pd.Series(meta_ts), max_lookback=1440)
    X_all     = assembled[fc_all].values
    ts_all    = assembled["timestamp"].values
    row_ok    = gap_ok & ~np.isnan(X_all).any(axis=1)
    X_clean   = X_all[row_ok]
    ts_clean  = ts_all[row_ok]

    folds  = walk_forward(ts_clean)
    y_full = np.array([all_labels["up_60"][ts_map[t]] for t in ts_clean])

    print(f"\n  {'Fold':>4}  {'Train period':>24}  {'Test period':>12}  "
          f"{'AUC':>7}  {'Pos%':>5}")
    print(f"  {'─'*65}")

    wf_aucs = []
    for fold in folds:
        ok_tr = ~np.isnan(y_full[fold.train])
        ok_te = ~np.isnan(y_full[fold.test])
        if ok_tr.sum() < 200 or ok_te.sum() < 50:
            continue
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_clean[fold.train][ok_tr])
        Xte = sc.transform(X_clean[fold.test][ok_te])
        ytr = y_full[fold.train][ok_tr]
        yte = y_full[fold.test][ok_te]
        m   = _train(Xtr, ytr, Xte, yte)
        auc = _auc(yte, m.predict(Xte))
        wf_aucs.append(auc)
        pos = yte.mean()
        print(f"  {fold.fold_idx:>4}  "
              f"{fmt(ts_clean[fold.train[0]])} → {fmt(ts_clean[fold.train[-1]])}  "
              f"{fmt(ts_clean[fold.test[0]])}  "
              f"{auc:>7.4f}  {pos:>5.1%}")

    if wf_aucs:
        passing = sum(a > 0.52 for a in wf_aucs)
        print(f"\n  Walk-forward: {passing}/{len(wf_aucs)} folds > 0.52  "
              f"mean AUC={np.mean(wf_aucs):.4f}")
        print(f"  Gate to DL stage: {'PASS ✓' if passing >= 5 else 'FAIL ✗  → fix features first'}")

    # save
    df  = pd.DataFrame(rows)
    out = CACHE_DIR / "lookup" / f"{ticker}_direction_eval.parquet"
    df.to_parquet(out, index=False)

    print(f"\n\n── Summary ──────────────────────────────────────────────────────")
    pivot = df.pivot(index="label", columns="split", values="auc").round(4)
    print(pivot.to_string())
    print(f"\nResults → {out.name}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
