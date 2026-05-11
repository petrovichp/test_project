"""
Probability calibration — fixes compressed score distributions.

Problem: direction models output probabilities in 0.01–0.15 range even for
positive cases. Precision ≥ 0.60 is unachievable because scores are too
compressed to set a meaningful threshold.

Fix: fit isotonic regression on val predictions to remap raw → calibrated probs.
Calibrator trained on val only, applied to test.

Run: python3 -m models.calibration [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from features.assembly import assemble
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD,
    _compute_labels, _build_sequences, _auc,
)

CACHE_DIR = Path(__file__).parent.parent / "cache"

_LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 64, "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5, "learning_rate": 0.05,
    "verbosity": -1,
}


def _train_lgbm(X_tr, y_tr, X_val, y_val):
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    return lgb.train(
        _LGB_PARAMS, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def _best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        p = (y_prob >= t).astype(int)
        if p.sum() == 0:
            continue
        f = f1_score(y_true, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def _metrics_at_threshold(y_true, y_prob, t):
    p = (y_prob >= t).astype(int)
    return {
        "auc":   _auc(y_true, y_prob),
        "prec":  precision_score(y_true, p, zero_division=0),
        "rec":   recall_score(y_true, p, zero_division=0),
        "f1":    f1_score(y_true, p, zero_division=0),
        "n_pos": int(p.sum()),
    }


def _precision_at_recall(y_true, y_prob, min_recall=0.10):
    """Best precision achievable while maintaining at least min_recall."""
    best_prec, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.99, 0.01):
        p = (y_prob >= t).astype(int)
        if p.sum() == 0:
            continue
        rec = recall_score(y_true, p, zero_division=0)
        if rec >= min_recall:
            prec = precision_score(y_true, p, zero_division=0)
            if prec > best_prec:
                best_prec, best_t = prec, t
    return best_prec, best_t


def _print_comparison(label, split, raw: dict, cal: dict, t_raw, t_cal):
    print(f"\n  {label}  |  {split.upper()}")
    print(f"  {'':22}  {'Raw':>10}  {'Calibrated':>12}  {'Delta':>8}")
    print(f"  {'─'*56}")
    print(f"  {'Threshold':22}  {t_raw:>10.2f}  {t_cal:>12.2f}")
    for key, label_ in [("auc","AUC"), ("prec","Precision"),
                         ("rec","Recall"), ("f1","F1"), ("n_pos","Pred pos")]:
        fmt = ",.0f" if key == "n_pos" else ".3f"
        rv = raw[key];  cv = cal[key]
        delta = cv - rv if key != "n_pos" else cv - rv
        delta_fmt = f"{delta:+.3f}" if key != "n_pos" else f"{delta:+,.0f}"
        print(f"  {label_:22}  {format(rv, fmt):>10}  {format(cv, fmt):>12}  {delta_fmt:>8}")


def run(ticker: str = "btc"):
    print(f"\n{'='*65}")
    print(f"  PROBABILITY CALIBRATION — {ticker.upper()}")
    print(f"  Fitting isotonic regression on val predictions")
    print(f"{'='*65}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)

    sel_idx  = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_tr_seq = X_train[:, sel_idx]
    X_v_seq  = X_val[:,   sel_idx]
    X_te_seq = X_test[:,  sel_idx]

    results = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n{'─'*65}")
            print(f"  Label: {col}")
            print(f"{'─'*65}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            # ── get ensemble predictions ──────────────────────────────────────
            lgbm = _train_lgbm(X_train[ok_tr], y_tr[ok_tr],
                               X_val[ok_val],  y_v[ok_val])
            p_lgbm_v  = lgbm.predict(X_val[ok_val])
            p_lgbm_te = lgbm.predict(X_test[ok_te])

            cnn_path = CACHE_DIR / "preds" / f"{ticker}_cnn_dir_{direction}_{H}.keras"
            if not cnn_path.exists():
                print(f"  CNN-LSTM not found, skipping.")
                continue
            cnn = tf.keras.models.load_model(str(cnn_path))

            Xs_v,  ys_v  = _build_sequences(X_v_seq,  y_v,  SEQ_LEN)
            Xs_te, ys_te = _build_sequences(X_te_seq, y_te, SEQ_LEN)
            p_cnn_v  = cnn.predict(Xs_v,  verbose=0).flatten()
            p_cnn_te = cnn.predict(Xs_te, verbose=0).flatten()

            # align to shared rows
            cnn_vi  = np.array([i for i in range(SEQ_LEN, len(X_val))  if not np.isnan(y_v[i])])
            cnn_ti  = np.array([i for i in range(SEQ_LEN, len(X_test)) if not np.isnan(y_te[i])])
            lgbm_vi = {o: p for p, o in enumerate(np.where(ok_val)[0])}
            lgbm_ti = {o: p for p, o in enumerate(np.where(ok_te)[0])}
            sv      = [i for i in cnn_vi if i in lgbm_vi]
            st      = [i for i in cnn_ti if i in lgbm_ti]
            cvmap   = {o: p for p, o in enumerate(cnn_vi)}
            ctmap   = {o: p for p, o in enumerate(cnn_ti)}

            lp_sv = np.array([p_lgbm_v[lgbm_vi[i]]  for i in sv])
            cp_sv = np.array([p_cnn_v[cvmap[i]]       for i in sv])
            lp_st = np.array([p_lgbm_te[lgbm_ti[i]] for i in st])
            cp_st = np.array([p_cnn_te[ctmap[i]]      for i in st])
            y_sv  = y_v[np.array(sv)]
            y_st  = y_te[np.array(st)]

            auc_l = _auc(y_sv, lp_sv);  auc_c = _auc(y_sv, cp_sv)
            w_l   = auc_l / (auc_l + auc_c);  w_c = 1 - w_l
            raw_v = w_l * lp_sv + w_c * cp_sv
            raw_t = w_l * lp_st + w_c * cp_st

            # ── calibration: fit isotonic on val, apply to test ───────────────
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw_v, y_sv)
            cal_v = cal.predict(raw_v)
            cal_t = cal.predict(raw_t)

            # ── score distribution before vs after ────────────────────────────
            print(f"\n  Score distribution (ensemble raw → calibrated):")
            print(f"  {'':12}  {'p5':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'p95':>6}  {'max':>6}")
            for tag, probs in [("raw val", raw_v), ("cal val", cal_v),
                                ("raw test", raw_t), ("cal test", cal_t)]:
                ps = np.percentile(probs, [5, 25, 50, 75, 95, 100])
                print(f"  {tag:12}  " + "  ".join(f"{p:6.3f}" for p in ps))

            # ── comparison at optimal F1 threshold ────────────────────────────
            t_raw_v = _best_threshold(y_sv, raw_v)
            t_cal_v = _best_threshold(y_sv, cal_v)

            m_raw_v = _metrics_at_threshold(y_sv, raw_v, t_raw_v)
            m_cal_v = _metrics_at_threshold(y_sv, cal_v, t_cal_v)
            m_raw_t = _metrics_at_threshold(y_st, raw_t, t_raw_v)
            m_cal_t = _metrics_at_threshold(y_st, cal_t, t_cal_v)

            _print_comparison(col, "val",  m_raw_v, m_cal_v, t_raw_v, t_cal_v)
            _print_comparison(col, "test", m_raw_t, m_cal_t, t_raw_v, t_cal_v)

            # ── best precision at recall ≥ 10% ────────────────────────────────
            prec_raw_t, t_pr = _precision_at_recall(y_st, raw_t, 0.10)
            prec_cal_t, t_pc = _precision_at_recall(y_st, cal_t, 0.10)
            print(f"\n  Best precision at recall≥10% (test):")
            print(f"    Raw:        {prec_raw_t:.3f}  (t={t_pr:.2f})")
            print(f"    Calibrated: {prec_cal_t:.3f}  (t={t_pc:.2f})  "
                  f"delta={prec_cal_t-prec_raw_t:+.3f}")

            results.append({
                "label": col,
                "raw_val_auc":  m_raw_v["auc"],  "cal_val_auc":  m_cal_v["auc"],
                "raw_test_auc": m_raw_t["auc"],  "cal_test_auc": m_cal_t["auc"],
                "raw_test_prec": m_raw_t["prec"], "cal_test_prec": m_cal_t["prec"],
                "raw_test_f1":   m_raw_t["f1"],   "cal_test_f1":   m_cal_t["f1"],
                "best_prec_raw": prec_raw_t,       "best_prec_cal": prec_cal_t,
            })

            # save calibrator
            cal_path = CACHE_DIR / "lookup" / f"{ticker}_cal_{direction}_{H}.npy"
            np.save(str(cal_path), {"thresholds": cal.X_thresholds_,
                                    "y_thresholds": cal.y_thresholds_})

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n\n{'='*65}")
        print(f"  CALIBRATION SUMMARY (test set)")
        print(f"{'='*65}")
        print(f"  {'Label':<12}  {'AUC raw':>8}  {'AUC cal':>8}  "
              f"{'Prec raw':>9}  {'Prec cal':>9}  {'F1 raw':>7}  {'F1 cal':>7}")
        print(f"  {'─'*65}")
        for _, r in df.iterrows():
            print(f"  {r['label']:<12}  {r['raw_test_auc']:>8.4f}  {r['cal_test_auc']:>8.4f}  "
                  f"{r['raw_test_prec']:>9.3f}  {r['cal_test_prec']:>9.3f}  "
                  f"{r['raw_test_f1']:>7.3f}  {r['cal_test_f1']:>7.3f}")

        out = CACHE_DIR / "lookup" / f"{ticker}_calibration_eval.parquet"
        df.to_parquet(out, index=False)
        print(f"\nResults → {out.name}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
