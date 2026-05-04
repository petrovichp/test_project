"""
Ensemble — weighted average of LightGBM + CNN-LSTM predictions.

Weights are derived from val AUC per label so the better model on val
contributes more. Evaluates on val and test, compares to each individual model.

Run: python3 -m models.ensemble [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from models.splits import sequential
from features.assembly import assemble
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD,
    _compute_labels, _build_sequences, build_cnn_lstm, _fit, _auc
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


def run(ticker: str = "btc"):
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE (LightGBM + CNN-LSTM) — {ticker.upper()}")
    print(f"{'='*60}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)
    fmt     = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # CNN-LSTM feature subset
    sel_idx = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_tr_seq = X_train[:, sel_idx]
    X_v_seq  = X_val[:,   sel_idx]
    X_te_seq = X_test[:,  sel_idx]
    n_feat   = X_tr_seq.shape[1]

    rows = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n── {col} ────────────────────────────────────────────────")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            # ── LightGBM ──────────────────────────────────────────────────────
            lgbm = _train_lgbm(X_train[ok_tr], y_tr[ok_tr],
                               X_val[ok_val],  y_v[ok_val])
            prob_lgbm_val  = lgbm.predict(X_val[ok_val])
            prob_lgbm_test = lgbm.predict(X_test[ok_te])

            auc_lgbm_val  = _auc(y_v[ok_val],  prob_lgbm_val)
            auc_lgbm_test = _auc(y_te[ok_te],  prob_lgbm_test)
            print(f"  LightGBM   val={auc_lgbm_val:.4f}  test={auc_lgbm_test:.4f}")

            # ── CNN-LSTM ──────────────────────────────────────────────────────
            X_seq_tr, y_seq_tr = _build_sequences(X_tr_seq, y_tr, SEQ_LEN)
            X_seq_v,  y_seq_v  = _build_sequences(X_v_seq,  y_v,  SEQ_LEN)
            X_seq_te, y_seq_te = _build_sequences(X_te_seq, y_te, SEQ_LEN)

            cnn = build_cnn_lstm(SEQ_LEN, n_feat)
            _fit(cnn, X_seq_tr, y_seq_tr, X_seq_v, y_seq_v)

            prob_cnn_val  = cnn.predict(X_seq_v,  verbose=0).flatten()
            prob_cnn_test = cnn.predict(X_seq_te, verbose=0).flatten()

            auc_cnn_val  = _auc(y_seq_v,  prob_cnn_val)
            auc_cnn_test = _auc(y_seq_te, prob_cnn_test)
            print(f"  CNN-LSTM   val={auc_cnn_val:.4f}  test={auc_cnn_test:.4f}")

            # ── Align val predictions (CNN-LSTM drops first SEQ_LEN rows) ─────
            # LightGBM val uses ok_val mask; CNN-LSTM val uses seq-valid subset
            # Intersect to same rows for fair ensemble
            n_v = len(X_val)
            cnn_valid_v = np.array([i for i in range(SEQ_LEN, n_v) if not np.isnan(y_v[i])])
            lgbm_map_v  = {orig: pos for pos, orig in enumerate(np.where(ok_val)[0])}
            shared_v    = [i for i in cnn_valid_v if i in lgbm_map_v]

            n_te = len(X_test)
            cnn_valid_te = np.array([i for i in range(SEQ_LEN, n_te) if not np.isnan(y_te[i])])
            lgbm_map_te  = {orig: pos for pos, orig in enumerate(np.where(ok_te)[0])}
            shared_te    = [i for i in cnn_valid_te if i in lgbm_map_te]

            def _shared_probs(shared, lgbm_probs, lgbm_map, cnn_probs, cnn_valid, y_arr):
                cnn_pos_map = {orig: pos for pos, orig in enumerate(cnn_valid)}
                lgb_p = np.array([lgbm_probs[lgbm_map[i]] for i in shared])
                cnn_p = np.array([cnn_probs[cnn_pos_map[i]] for i in shared])
                y_s   = y_arr[np.array(shared)]
                return lgb_p, cnn_p, y_s

            lgb_pv, cnn_pv, y_sv = _shared_probs(
                shared_v, prob_lgbm_val, lgbm_map_v,
                prob_cnn_val, cnn_valid_v, y_v)
            lgb_pt, cnn_pt, y_st = _shared_probs(
                shared_te, prob_lgbm_test, lgbm_map_te,
                prob_cnn_test, cnn_valid_te, y_te)

            # ── val-AUC weighted ensemble ──────────────────────────────────────
            w_lgbm = auc_lgbm_val / (auc_lgbm_val + auc_cnn_val)
            w_cnn  = auc_cnn_val  / (auc_lgbm_val + auc_cnn_val)

            ens_pv = w_lgbm * lgb_pv + w_cnn * cnn_pv
            ens_pt = w_lgbm * lgb_pt + w_cnn * cnn_pt

            auc_ens_val  = _auc(y_sv, ens_pv)
            auc_ens_test = _auc(y_st, ens_pt)
            gap = abs(auc_ens_val - auc_ens_test)
            flag = "✓" if gap <= 0.03 else "⚠"

            print(f"  Ensemble   val={auc_ens_val:.4f}  test={auc_ens_test:.4f}  "
                  f"(w_lgbm={w_lgbm:.2f} w_cnn={w_cnn:.2f})  {flag} gap={gap:.3f}")

            best_test = max(auc_lgbm_test, auc_cnn_test)
            lift = auc_ens_test - best_test
            print(f"  Lift over best single model: {lift:+.4f}")

            for model_name, av, at in [
                ("lightgbm", auc_lgbm_val, auc_lgbm_test),
                ("cnn_lstm",  auc_cnn_val,  auc_cnn_test),
                ("ensemble",  auc_ens_val,  auc_ens_test),
            ]:
                rows.append({"model": model_name, "label": col,
                             "split": "val",  "auc": av})
                rows.append({"model": model_name, "label": col,
                             "split": "test", "auc": at})

    df  = pd.DataFrame(rows)
    out = CACHE_DIR / f"{ticker}_ensemble_eval.parquet"
    df.to_parquet(out, index=False)

    print(f"\n\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    pivot = df.pivot_table(
        index="label", columns=["model", "split"], values="auc"
    ).round(4)
    print(pivot.to_string())
    print(f"\nResults → {out.name}")
    return df


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
