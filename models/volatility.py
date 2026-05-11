"""
Volatility prediction — LightGBM regressors.

Targets  : atr (avg true range, $) | rvol (realized vol, %)
Horizons : 15, 30, 60, 100, 240 bars

Model coding: {ticker}_lgbm_{target}_{horizon}
  e.g. btc_lgbm_atr_30, btc_lgbm_rvol_60

Each model is saved to cache/ and registered in model_registry.json.
On rerun, cached models are loaded instead of retrained.

Run: python3 -m models.volatility [ticker] [--force]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from models.splits import sequential, walk_forward
from features.assembly import assemble

CACHE_DIR    = Path(__file__).parent.parent / "cache"
REGISTRY     = Path(__file__).parent.parent / "model_registry.json"

HORIZONS     = [15, 30, 60, 100, 240]
TARGETS      = ["atr", "rvol"]   # price_range dropped — weakest signal


# ── model registry ────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {}


def _save_registry(reg: dict):
    REGISTRY.write_text(json.dumps(reg, indent=2))


def _model_code(ticker: str, target: str, horizon: int) -> str:
    return f"{ticker}_lgbm_{target}_{horizon}"


def _model_path(code: str) -> Path:
    return CACHE_DIR / "preds" / f"{code}.txt"


# ── label computation ─────────────────────────────────────────────────────────

def _compute_targets(price_arr: np.ndarray) -> dict[str, np.ndarray]:
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price_arr)
    out = {}
    for H in HORIZONS:
        atr  = np.full(n, np.nan)
        rvol = np.full(n, np.nan)
        if n > H:
            wins     = sliding_window_view(price_arr[1:], H)
            log_wins = np.log(sliding_window_view(price_arr, H + 1))
            atr[:n - H]  = np.mean(np.abs(np.diff(wins, axis=1)), axis=1)
            rvol[:n - H] = np.std(np.diff(log_wins, axis=1), axis=1)
        out[f"atr_{H}"]  = atr
        out[f"rvol_{H}"] = rvol
    return out


# ── LightGBM helpers ──────────────────────────────────────────────────────────

_LGB_PARAMS = {
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


def _train_lgbm(X_tr, y_tr, X_val, y_val, save_path: Path) -> lgb.Booster:
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    model  = lgb.train(
        _LGB_PARAMS, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    model.save_model(str(save_path))
    return model


def _load_or_train(X_tr, y_tr, X_val, y_val, code: str, force: bool) -> lgb.Booster:
    path = _model_path(code)
    if path.exists() and not force:
        print(f"  Loading cached: {code}")
        return lgb.Booster(model_file=str(path))
    print(f"  Training: {code}")
    return _train_lgbm(X_tr, y_tr, X_val, y_val, path)


# ── metrics ───────────────────────────────────────────────────────────────────

def _reg_metrics(y_true, y_pred) -> dict:
    return {
        "spearman": float(spearmanr(y_pred, y_true).statistic),
        "rmse":     float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "dir_acc":  float(np.mean(
            (y_pred > np.percentile(y_true, 66.7)) ==
            (y_true > np.percentile(y_true, 66.7))
        )),
    }


def _confusion_block(y_true_val, y_pred_val, y_true_test, y_pred_test):
    for pct_cut in [33, 10]:
        cutoff = np.percentile(y_true_val, 100 - pct_cut)

        yv_t = (y_true_val  >= cutoff).astype(int)
        yv_p = (y_pred_val  >= cutoff).astype(int)
        yt_t = (y_true_test >= cutoff).astype(int)
        yt_p = (y_pred_test >= cutoff).astype(int)

        print(f"\n  Confusion matrix — top-{pct_cut}% = HIGH vol  "
              f"(cutoff={cutoff:.6f})")
        print(f"  {'':18}  {'VAL':>30}  {'TEST':>30}")
        print(f"  {'':18}  {'Pred LOW':>14}{'Pred HIGH':>16}  "
              f"{'Pred LOW':>14}{'Pred HIGH':>16}")

        for lbl, vt, tt in [("Actual LOW", 0, 0), ("Actual HIGH", 1, 1)]:
            mv = yv_t == vt;  mt = yt_t == tt
            print(f"  {lbl:<18}  "
                  f"{((yv_p==0)&mv).sum():>14,}{((yv_p==1)&mv).sum():>16,}  "
                  f"{((yt_p==0)&mt).sum():>14,}{((yt_p==1)&mt).sum():>16,}")

        def _s(yt, yp, tag):
            print(f"  {tag:<18}  "
                  f"Prec={precision_score(yt,yp,zero_division=0):.3f}  "
                  f"Rec={recall_score(yt,yp,zero_division=0):.3f}  "
                  f"F1={f1_score(yt,yp,zero_division=0):.3f}  "
                  f"Pred-high={yp.sum():,}")
        _s(yv_t, yv_p, "VAL")
        _s(yt_t, yt_p, "TEST")


# ── walk-forward ──────────────────────────────────────────────────────────────

def _walk_forward(X_all, y_full, ts_clean, code: str):
    from sklearn.preprocessing import StandardScaler
    folds = walk_forward(ts_clean)
    fmt   = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    print(f"\n  {'Fold':>4}  {'Train':>23}  {'Test start':>12}  "
          f"{'Spearman':>10}  {'DirAcc':>8}")
    print(f"  {'─'*65}")
    wf_corrs = []
    for fold in folds:
        ok_tr = ~np.isnan(y_full[fold.train])
        ok_te = ~np.isnan(y_full[fold.test])
        if ok_tr.sum() < 100 or ok_te.sum() < 10:
            continue
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_all[fold.train][ok_tr])
        Xte = sc.transform(X_all[fold.test][ok_te])
        ytr = y_full[fold.train][ok_tr]
        yte = y_full[fold.test][ok_te]
        m   = _train_lgbm(Xtr, ytr, Xte, yte,
                          CACHE_DIR / "preds" / f"_wf_tmp_{fold.fold_idx}.txt")
        (CACHE_DIR / "preds" / f"_wf_tmp_{fold.fold_idx}.txt").unlink(missing_ok=True)
        corr    = spearmanr(m.predict(Xte), yte).statistic
        dir_acc = np.mean((m.predict(Xte) > np.percentile(yte, 66.7)) ==
                          (yte > np.percentile(yte, 66.7)))
        wf_corrs.append(corr)
        print(f"  {fold.fold_idx:>4}  "
              f"{fmt(ts_clean[fold.train[0]])} → "
              f"{fmt(ts_clean[fold.train[-1]])}  "
              f"{fmt(ts_clean[fold.test[0]]):>12}  "
              f"{corr:>+.3f}  {dir_acc:>7.3f}")
    if wf_corrs:
        print(f"\n  Mean Spearman={np.mean(wf_corrs):.3f}  "
              f"Min={min(wf_corrs):.3f}  Max={max(wf_corrs):.3f}  "
              f"All positive: {'YES' if all(c>0 for c in wf_corrs) else 'NO'}")


# ── main run ──────────────────────────────────────────────────────────────────

def run(ticker: str = "btc", horizons: list[int] = None,
        targets: list[str] = None, force: bool = False):

    horizons = horizons or [30, 60, 100]
    targets  = targets  or ["atr", "rvol"]

    print(f"\n{'='*65}")
    print(f"  VOLATILITY MODELS — {ticker.upper()}")
    print(f"  Targets: {targets}  Horizons: {horizons}")
    print(f"{'='*65}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_tgt = _compute_targets(price)

    def _get_y(ts_arr, col):
        return all_tgt[col][np.array([ts_map[t] for t in ts_arr])]

    # full clean matrix for walk-forward
    assembled = pd.read_parquet(CACHE_DIR / "features" / f"{ticker}_features_assembled.parquet")
    fc_all    = [c for c in assembled.columns if c != "timestamp"]
    from data.gaps import clean_mask as _cmask
    gap_ok    = _cmask(pd.Series(meta["timestamp"].values), max_lookback=1440)
    X_all     = assembled[fc_all].values
    row_ok    = gap_ok & ~np.isnan(X_all).any(axis=1)
    X_clean   = X_all[row_ok]
    ts_clean  = assembled["timestamp"].values[row_ok]

    reg   = _load_registry()
    fmt   = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    for target in targets:
        for H in horizons:
            col  = f"{target}_{H}"
            code = _model_code(ticker, target, H)

            print(f"\n{'─'*65}")
            print(f"  Model: {code}")
            print(f"{'─'*65}")

            y_tr  = _get_y(ts_train, col);  ok_tr  = ~np.isnan(y_tr)
            y_v   = _get_y(ts_val,   col);  ok_val = ~np.isnan(y_v)
            y_te  = _get_y(ts_test,  col);  ok_te  = ~np.isnan(y_te)

            # ── train or load ────────────────────────────────────────────────
            model  = _load_or_train(X_train[ok_tr], y_tr[ok_tr],
                                    X_val[ok_val],  y_v[ok_val],
                                    code, force)

            p_val  = model.predict(X_val[ok_val])
            p_test = model.predict(X_test[ok_te])

            # ── regression metrics ───────────────────────────────────────────
            mv = _reg_metrics(y_v[ok_val],  p_val)
            mt = _reg_metrics(y_te[ok_te],  p_test)

            print(f"\n  Regression metrics:")
            print(f"  {'':6}  Spearman   RMSE          DirAcc")
            print(f"  {'Val':<6}  {mv['spearman']:>+.3f}     {mv['rmse']:.6f}   {mv['dir_acc']:.3f}")
            print(f"  {'Test':<6}  {mt['spearman']:>+.3f}     {mt['rmse']:.6f}   {mt['dir_acc']:.3f}")
            gap = abs(mv["spearman"] - mt["spearman"])
            flag = "⚠" if gap > 0.1 else "✓"
            print(f"  {flag}  val/test Spearman gap: {gap:.3f}")

            # ── confusion matrices ───────────────────────────────────────────
            _confusion_block(y_v[ok_val], p_val, y_te[ok_te], p_test)

            # ── walk-forward (only on first run to save time) ────────────────
            if not _model_path(code).exists() or force:
                print(f"\n  Walk-forward:")
                y_full = all_tgt[col][np.array([ts_map[t] for t in ts_clean])]
                _walk_forward(X_clean, y_full, ts_clean, code)

            # ── register ─────────────────────────────────────────────────────
            reg[code] = {
                "ticker": ticker, "model_type": "lgbm",
                "target": target, "horizon": H,
                "val_spearman":  round(mv["spearman"], 4),
                "test_spearman": round(mt["spearman"], 4),
                "val_dir_acc":   round(mv["dir_acc"],  4),
                "test_dir_acc":  round(mt["dir_acc"],  4),
                "trained_at":    datetime.utcnow().isoformat(),
                "model_file":    f"{code}.txt",
            }
            _save_registry(reg)
            print(f"\n  Registered: {code} → model_registry.json")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY — {ticker.upper()}")
    print(f"{'='*65}")
    print(f"  {'Code':<25}  {'Val Spearman':>13}  {'Test Spearman':>14}  {'Gap':>6}")
    print(f"  {'─'*65}")
    for code, info in reg.items():
        if info["ticker"] != ticker:
            continue
        gap = abs(info["test_spearman"] - info["val_spearman"])
        print(f"  {code:<25}  {info['val_spearman']:>+13.3f}  "
              f"{info['test_spearman']:>+14.3f}  {gap:>6.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker",    nargs="?", default="btc")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 60, 100])
    parser.add_argument("--targets",  nargs="+", default=["atr", "rvol"])
    args = parser.parse_args()
    run(args.ticker, args.horizons, args.targets, args.force)
