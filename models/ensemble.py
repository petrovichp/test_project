"""
Ensemble — weighted average of LightGBM + CNN-LSTM.

Loads cached CNN-LSTM models (.keras files).
Trains LightGBM fresh (fast, ~10s per label) with the full 191-feature set.
Weights derived from val AUC per label.

Prints confusion matrices for val and test at:
  - optimal F1 threshold (set on val)
  - high-precision threshold (precision >= 0.60 on val)

Run: python3 -m models.ensemble [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from models.splits import sequential
from features.assembly import assemble
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD,
    _compute_labels, _build_sequences, _auc,
)

CACHE_DIR = Path(__file__).parent.parent / "cache"
REGISTRY  = Path(__file__).parent.parent / "model_registry.json"

_LGB_PARAMS = {
    "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
    "num_leaves": 64, "min_data_in_leaf": 100, "feature_fraction": 0.7,
    "bagging_fraction": 0.8, "bagging_freq": 5, "learning_rate": 0.05,
    "verbosity": -1,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _train_lgbm(X_tr, y_tr, X_val, y_val):
    ds_tr  = lgb.Dataset(X_tr,  label=y_tr)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_tr)
    return lgb.train(
        _LGB_PARAMS, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )


def _best_threshold(y_true, y_prob, metric="f1"):
    best_t, best_s = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        p = (y_prob >= t).astype(int)
        if p.sum() == 0:
            continue
        s = f1_score(y_true, p, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return best_t


def _prec_threshold(y_true, y_prob, min_prec=0.60):
    for t in np.arange(0.95, 0.04, -0.01):
        p = (y_prob >= t).astype(int)
        if p.sum() == 0:
            continue
        if precision_score(y_true, p, zero_division=0) >= min_prec:
            return t
    return 0.90


def _cm_block(y_true, y_prob, t):
    p    = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, p, labels=[0, 1]).ravel()
    prec = precision_score(y_true, p, zero_division=0)
    rec  = recall_score(y_true, p, zero_division=0)
    f1   = f1_score(y_true, p, zero_division=0)
    auc  = _auc(y_true, y_prob)
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                prec=prec, rec=rec, f1=f1, auc=auc,
                n_pos=int(p.sum()), threshold=t)


def _print_cm(label, split, t_label, models_data: dict):
    names  = list(models_data.keys())
    col_w  = 24
    print(f"\n  {'─'*72}")
    print(f"  {label}  |  {split.upper()}  |  {t_label}")
    print(f"  {'─'*72}")
    print(f"  {'':20}", end="")
    for n in names:
        print(f"  {n:>{col_w}}", end="")
    print()
    print(f"  {'':20}", end="")
    for n in names:
        t = models_data[n]["threshold"]
        print(f"  {'t='+f'{t:.2f}':>{col_w}}", end="")
    print()

    for row_label, neg_k, pos_k in [("Actual NEG", "tn", "fp"),
                                     ("Actual POS", "fn", "tp")]:
        print(f"  {row_label:<20}", end="")
        for n in names:
            d = models_data[n]
            cell = (f"TN={d['tn']:,}  FP={d['fp']:,}"
                    if neg_k == "tn" else
                    f"FN={d['fn']:,}  TP={d['tp']:,}")
            print(f"  {cell:>{col_w}}", end="")
        print()

    for m_label, key, fmt in [
        ("AUC",       "auc",  ".4f"),
        ("Precision", "prec", ".3f"),
        ("Recall",    "rec",  ".3f"),
        ("F1",        "f1",   ".3f"),
        ("Pred pos",  "n_pos", ",d"),
    ]:
        print(f"  {m_label:<20}", end="")
        for n in names:
            v = models_data[n][key]
            print(f"  {format(v, fmt):>{col_w}}", end="")
        print()


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*72}")
    print(f"  ENSEMBLE — LightGBM + CNN-LSTM  ({ticker.upper()})")
    print(f"{'='*72}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)
    fmt     = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # CNN-LSTM feature subset
    sel_idx  = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_tr_seq = X_train[:, sel_idx]
    X_v_seq  = X_val[:,   sel_idx]
    X_te_seq = X_test[:,  sel_idx]

    reg  = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    rows = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n{'='*72}")
            print(f"  Label: {col}  (threshold={THRESHOLD*100:.1f}%,  H={H} bars)")
            print(f"{'='*72}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            pos_tr = y_tr[ok_tr].mean()
            print(f"\n  Class balance — train {pos_tr:.1%} | "
                  f"val {y_v[ok_val].mean():.1%} | "
                  f"test {y_te[ok_te].mean():.1%}")

            # ── LightGBM ──────────────────────────────────────────────────────
            print(f"\n  Training LightGBM ...")
            lgbm = _train_lgbm(X_train[ok_tr], y_tr[ok_tr],
                               X_val[ok_val],  y_v[ok_val])
            p_lgbm_v  = lgbm.predict(X_val[ok_val])
            p_lgbm_te = lgbm.predict(X_test[ok_te])
            auc_lgbm_v  = _auc(y_v[ok_val],  p_lgbm_v)
            auc_lgbm_te = _auc(y_te[ok_te],  p_lgbm_te)
            print(f"  LightGBM  val={auc_lgbm_v:.4f}  test={auc_lgbm_te:.4f}")

            # ── CNN-LSTM (load cached) ─────────────────────────────────────────
            cnn_code = f"{ticker}_cnn_dir_{direction}_{H}"
            cnn_path = CACHE_DIR / f"{cnn_code}.keras"
            if not cnn_path.exists():
                print(f"  CNN-LSTM model not found: {cnn_path.name} — skipping")
                continue
            print(f"  Loading CNN-LSTM: {cnn_path.name}")
            cnn = tf.keras.models.load_model(str(cnn_path))

            Xs_v,  ys_v  = _build_sequences(X_v_seq,  y_v,  SEQ_LEN)
            Xs_te, ys_te = _build_sequences(X_te_seq, y_te, SEQ_LEN)
            p_cnn_v  = cnn.predict(Xs_v,  verbose=0).flatten()
            p_cnn_te = cnn.predict(Xs_te, verbose=0).flatten()
            auc_cnn_v  = _auc(ys_v,  p_cnn_v)
            auc_cnn_te = _auc(ys_te, p_cnn_te)
            print(f"  CNN-LSTM  val={auc_cnn_v:.4f}  test={auc_cnn_te:.4f}")

            # ── align LightGBM to CNN-LSTM rows (CNN drops first SEQ_LEN rows) ─
            cnn_vi   = np.array([i for i in range(SEQ_LEN, len(X_val))  if not np.isnan(y_v[i])])
            cnn_ti   = np.array([i for i in range(SEQ_LEN, len(X_test)) if not np.isnan(y_te[i])])
            lgbm_vi  = {orig: pos for pos, orig in enumerate(np.where(ok_val)[0])}
            lgbm_ti  = {orig: pos for pos, orig in enumerate(np.where(ok_te)[0])}
            shared_v = [i for i in cnn_vi if i in lgbm_vi]
            shared_t = [i for i in cnn_ti if i in lgbm_ti]
            cnn_vmap = {orig: pos for pos, orig in enumerate(cnn_vi)}
            cnn_tmap = {orig: pos for pos, orig in enumerate(cnn_ti)}

            def _shared(shared, lp, lm, cp, cm, y_arr):
                lv = np.array([lp[lm[i]] for i in shared])
                cv = np.array([cp[cm[i]] for i in shared])
                yv = y_arr[np.array(shared)]
                return lv, cv, yv

            lp_sv, cp_sv, y_sv = _shared(shared_v, p_lgbm_v,  lgbm_vi, p_cnn_v,  cnn_vmap, y_v)
            lp_st, cp_st, y_st = _shared(shared_t, p_lgbm_te, lgbm_ti, p_cnn_te, cnn_tmap, y_te)

            # ── ensemble weights from val AUC ──────────────────────────────────
            auc_l = _auc(y_sv, lp_sv)
            auc_c = _auc(y_sv, cp_sv)
            w_l   = auc_l / (auc_l + auc_c)
            w_c   = auc_c / (auc_l + auc_c)
            ep_sv = w_l * lp_sv + w_c * cp_sv
            ep_st = w_l * lp_st + w_c * cp_st

            auc_ens_v  = _auc(y_sv, ep_sv)
            auc_ens_te = _auc(y_st, ep_st)
            print(f"  Ensemble  val={auc_ens_v:.4f}  test={auc_ens_te:.4f}  "
                  f"(w_lgbm={w_l:.2f}  w_cnn={w_c:.2f})")

            # ── confusion matrices ─────────────────────────────────────────────
            t_f1   = _best_threshold(y_sv,  ep_sv)
            t_prec = _prec_threshold(y_sv,  ep_sv, min_prec=0.60)

            for split, y_true, lgbm_p, cnn_p, ens_p in [
                ("val",  y_sv, lp_sv, cp_sv, ep_sv),
                ("test", y_st, lp_st, cp_st, ep_st),
            ]:
                for t_label, t_val in [
                    ("optimal F1",     t_f1),
                    ("precision≥0.60", t_prec),
                ]:
                    models_data = {
                        "LightGBM": _cm_block(y_true, lgbm_p, t_val),
                        "CNN-LSTM":  _cm_block(y_true, cnn_p,  t_val),
                        "Ensemble":  _cm_block(y_true, ens_p,  t_val),
                    }
                    _print_cm(col, split, t_label, models_data)

            # ── register ensemble ──────────────────────────────────────────────
            ens_code = f"{ticker}_ens_dir_{direction}_{H}"
            reg[ens_code] = {
                "ticker": ticker, "model_type": "ens",
                "target": f"dir_{direction}", "horizon": H,
                "val_auc":  round(auc_ens_v,  4),
                "test_auc": round(auc_ens_te, 4),
                "w_lgbm": round(w_l, 3), "w_cnn": round(w_c, 3),
                "trained_at": datetime.utcnow().isoformat(),
            }
            REGISTRY.write_text(json.dumps(reg, indent=2))

            for m, av, at in [("lightgbm", auc_lgbm_v, auc_lgbm_te),
                               ("cnn_lstm", auc_cnn_v,  auc_cnn_te),
                               ("ensemble", auc_ens_v,  auc_ens_te)]:
                rows.append({"label": col, "model": m,
                             "val_auc": av, "test_auc": at})

    # ── final summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n\n{'='*72}")
        print(f"  FINAL AUC SUMMARY")
        print(f"{'='*72}")
        pivot = df.pivot_table(index="label", columns="model",
                               values=["val_auc", "test_auc"]).round(4)
        print(pivot.to_string())

        out = CACHE_DIR / f"{ticker}_ensemble_eval.parquet"
        df.to_parquet(out, index=False)
        print(f"\nResults → {out.name}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
