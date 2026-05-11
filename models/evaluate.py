"""
Model evaluation with confusion matrices.

Retrains LightGBM, CNN-LSTM, and Ensemble, saves all predictions,
then prints side-by-side confusion matrices for val and test at:
  - optimal threshold (maximises F1 on val)
  - high-precision threshold (precision >= 0.70 on val)

Run: python3 -m models.evaluate [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from models.splits import sequential
from features.assembly import assemble
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD,
    _compute_labels, _build_sequences, build_cnn_lstm, _fit, _auc,
)

CACHE_DIR = Path(__file__).parent.parent / "cache"

_LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 64, "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5, "learning_rate": 0.05,
    "verbosity": -1,
}


# ── threshold selection ───────────────────────────────────────────────────────

def _optimal_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_score = 0.5, -1
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score, best_t = score, t
    return best_t


def _precision_threshold(y_true, y_prob, min_precision=0.70):
    """Lowest threshold that still achieves min_precision on val."""
    for t in np.arange(0.95, 0.04, -0.01):
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        if precision_score(y_true, preds, zero_division=0) >= min_precision:
            return t
    return 0.90   # fallback if never achieved


# ── confusion matrix display ──────────────────────────────────────────────────

def _cm_block(y_true, y_prob, threshold):
    preds = (y_prob >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], 0, 0, 0)
    prec  = precision_score(y_true, preds, zero_division=0)
    rec   = recall_score(y_true, preds, zero_division=0)
    f1    = f1_score(y_true, preds, zero_division=0)
    auc   = _auc(y_true, y_prob)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                prec=prec, rec=rec, f1=f1, auc=auc, threshold=threshold,
                n_pred_pos=int(preds.sum()), n_total=len(y_true))


def _print_comparison(label, split, models_data: dict, threshold_type: str):
    """
    models_data: {model_name: cm_block_dict}
    """
    names  = list(models_data.keys())
    col_w  = 22

    header = f"\n{'─'*70}"
    print(header)
    print(f"  {label}  |  {split.upper()}  |  threshold: {threshold_type}")
    print(f"{'─'*70}")

    # column headers
    print(f"  {'':18}", end="")
    for name in names:
        print(f"  {name:>{col_w}}", end="")
    print()
    print(f"  {'':18}", end="")
    for name in names:
        t = models_data[name]["threshold"]
        print(f"  {'t='+f'{t:.2f}':>{col_w}}", end="")
    print()
    print(f"  {'─'*18}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    # confusion matrix rows
    for row_label, neg_key, pos_key in [("Actual NEG", "tn", "fp"),
                                         ("Actual POS", "fn", "tp")]:
        print(f"  {row_label:<18}", end="")
        for name in names:
            d = models_data[name]
            neg_v = d[neg_key]
            pos_v = d[pos_key]
            cell  = f"TN={neg_v:,}  FP={pos_v:,}" if neg_key == "tn" else \
                    f"FN={neg_v:,}  TP={pos_v:,}"
            print(f"  {cell:>{col_w}}", end="")
        print()

    print(f"  {'─'*18}", end="")
    for _ in names:
        print(f"  {'─'*col_w}", end="")
    print()

    # metrics
    for metric_name, key, fmt in [
        ("AUC",       "auc",   ".4f"),
        ("Precision", "prec",  ".3f"),
        ("Recall",    "rec",   ".3f"),
        ("F1",        "f1",    ".3f"),
        ("Pred pos",  "n_pred_pos", ",d"),
    ]:
        print(f"  {metric_name:<18}", end="")
        for name in names:
            v = models_data[name][key]
            val_str = format(v, fmt)
            print(f"  {val_str:>{col_w}}", end="")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*70}")
    print(f"  CONFUSION MATRIX COMPARISON — {ticker.upper()}")
    print(f"{'='*70}")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)

    sel_idx  = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_tr_seq = X_train[:, sel_idx]
    X_v_seq  = X_val[:,   sel_idx]
    X_te_seq = X_test[:,  sel_idx]
    n_feat   = X_tr_seq.shape[1]

    all_rows = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n\n{'='*70}")
            print(f"  LABEL: {col}  (threshold={THRESHOLD*100:.1f}%, horizon={H} bars)")
            print(f"{'='*70}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            print(f"\n  Class balance — train pos={y_tr[ok_tr].mean():.1%}  "
                  f"val pos={y_v[ok_val].mean():.1%}  "
                  f"test pos={y_te[ok_te].mean():.1%}")

            # ── LightGBM ──────────────────────────────────────────────────────
            ds_tr  = lgb.Dataset(X_train[ok_tr], label=y_tr[ok_tr])
            ds_val = lgb.Dataset(X_val[ok_val],  label=y_v[ok_val], reference=ds_tr)
            lgbm   = lgb.train(
                _LGB_PARAMS, ds_tr, num_boost_round=500,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
            p_lgbm_v  = lgbm.predict(X_val[ok_val])
            p_lgbm_te = lgbm.predict(X_test[ok_te])

            # ── CNN-LSTM ──────────────────────────────────────────────────────
            Xs_tr, ys_tr = _build_sequences(X_tr_seq, y_tr, SEQ_LEN)
            Xs_v,  ys_v  = _build_sequences(X_v_seq,  y_v,  SEQ_LEN)
            Xs_te, ys_te = _build_sequences(X_te_seq, y_te, SEQ_LEN)
            cnn = build_cnn_lstm(SEQ_LEN, n_feat)
            _fit(cnn, Xs_tr, ys_tr, Xs_v, ys_v)
            p_cnn_v  = cnn.predict(Xs_v,  verbose=0).flatten()
            p_cnn_te = cnn.predict(Xs_te, verbose=0).flatten()

            # ── align for ensemble (CNN-LSTM drops first SEQ_LEN rows) ────────
            cnn_vi   = np.array([i for i in range(SEQ_LEN, len(X_val))  if not np.isnan(y_v[i])])
            cnn_ti   = np.array([i for i in range(SEQ_LEN, len(X_test)) if not np.isnan(y_te[i])])
            lgbm_vi  = {orig: pos for pos, orig in enumerate(np.where(ok_val)[0])}
            lgbm_ti  = {orig: pos for pos, orig in enumerate(np.where(ok_te)[0])}
            shared_v = [i for i in cnn_vi if i in lgbm_vi]
            shared_t = [i for i in cnn_ti if i in lgbm_ti]
            cnn_vmap = {orig: pos for pos, orig in enumerate(cnn_vi)}
            cnn_tmap = {orig: pos for pos, orig in enumerate(cnn_ti)}

            def _shared(shared, lgbm_p, lgbm_map, cnn_p, cnn_map, y_arr):
                lp = np.array([lgbm_p[lgbm_map[i]] for i in shared])
                cp = np.array([cnn_p[cnn_map[i]]   for i in shared])
                yt = y_arr[np.array(shared)]
                return lp, cp, yt

            lp_sv, cp_sv, y_sv = _shared(shared_v, p_lgbm_v, lgbm_vi, p_cnn_v, cnn_vmap, y_v)
            lp_st, cp_st, y_st = _shared(shared_t, p_lgbm_te, lgbm_ti, p_cnn_te, cnn_tmap, y_te)

            w_lgbm = _auc(y_sv, lp_sv) / (_auc(y_sv, lp_sv) + _auc(y_sv, cp_sv))
            w_cnn  = 1 - w_lgbm
            ep_sv  = w_lgbm * lp_sv + w_cnn * cp_sv
            ep_st  = w_lgbm * lp_st + w_cnn * cp_st

            # ── thresholds from val ────────────────────────────────────────────
            t_opt  = _optimal_threshold(y_sv, ep_sv)        # ensemble optimal
            t_prec = _precision_threshold(y_sv, ep_sv, 0.70)

            for split, preds_dict in [
                ("val",  {"lgbm_yv": (y_v[ok_val], p_lgbm_v),
                          "cnn_yv":  (ys_v,         p_cnn_v),
                          "ens_yv":  (y_sv,          ep_sv)}),
                ("test", {"lgbm_yt": (y_te[ok_te],  p_lgbm_te),
                          "cnn_yt":  (ys_te,          p_cnn_te),
                          "ens_yt":  (y_st,            ep_st)}),
            ]:
                y_lgbm, p_lgbm = list(preds_dict.values())[0]
                y_cnn,  p_cnn  = list(preds_dict.values())[1]
                y_ens,  p_ens  = list(preds_dict.values())[2]

                for t_type, t_val in [("optimal F1", t_opt), (f"precision≥0.70", t_prec)]:
                    models_data = {
                        "LightGBM": _cm_block(y_lgbm, p_lgbm, t_val),
                        "CNN-LSTM":  _cm_block(y_cnn,  p_cnn,  t_val),
                        "Ensemble":  _cm_block(y_ens,  p_ens,  t_val),
                    }
                    _print_comparison(col, split, models_data, t_type)

                    for mname, cm in models_data.items():
                        all_rows.append({
                            "label": col, "split": split, "model": mname,
                            "threshold_type": t_type, **cm,
                        })

    df  = pd.DataFrame(all_rows)
    out = CACHE_DIR / "lookup" / f"{ticker}_confusion_eval.parquet"
    df.to_parquet(out, index=False)
    print(f"\n\nAll results → {out.name}")
    return df


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
