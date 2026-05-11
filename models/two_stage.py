"""
Two-stage pipeline — feeds predicted ATR as a feature into direction models.

Stage 1: btc_lgbm_atr_30 predicts volatility for each bar
Stage 2: direction model (LightGBM + CNN-LSTM ensemble) uses ATR prediction
         as an additional input feature

Hypothesis: high predicted ATR → larger expected moves → direction model
should fire with higher confidence. Low ATR → avoid trading (noisy market).

Run: python3 -m models.two_stage [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from features.assembly import assemble
from models.volatility import _compute_targets, _LGB_PARAMS as _VOL_LGB_PARAMS
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD,
    _compute_labels, _build_sequences, _auc,
    build_cnn_lstm, _fit,
)

CACHE_DIR = Path(__file__).parent.parent / "cache"
VOL_MODEL = "btc_lgbm_atr_30"

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


def _cm_row(y_true, y_prob, t):
    p    = (y_prob >= t).astype(int)
    prec = precision_score(y_true, p, zero_division=0)
    rec  = recall_score(y_true, p, zero_division=0)
    f1   = f1_score(y_true, p, zero_division=0)
    return _auc(y_true, y_prob), prec, rec, f1, int(p.sum())


def _print_comparison(col, split, baseline, two_stage, t_b, t_ts):
    b_auc, b_prec, b_rec, b_f1, b_n = baseline
    t_auc, t_prec, t_rec, t_f1, t_n = two_stage
    print(f"\n  {col}  |  {split.upper()}")
    print(f"  {'':18}  {'Baseline':>10}  {'Two-stage':>11}  {'Delta':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Threshold':18}  {t_b:>10.2f}  {t_ts:>11.2f}")
    for name, bv, tv in [("AUC", b_auc, t_auc), ("Precision", b_prec, t_prec),
                          ("Recall", b_rec, t_rec), ("F1", b_f1, t_f1)]:
        print(f"  {name:18}  {bv:>10.3f}  {tv:>11.3f}  {tv-bv:>+8.3f}")
    print(f"  {'Pred pos':18}  {b_n:>10,}  {t_n:>11,}  {t_n-b_n:>+8,}")


def run(ticker: str = "btc"):
    print(f"\n{'='*65}")
    print(f"  TWO-STAGE PIPELINE — {ticker.upper()}")
    print(f"  Stage 1: ATR prediction  →  Stage 2: direction model")
    print(f"{'='*65}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)
    all_vol = _compute_targets(price)

    # ── Stage 1: get ATR predictions for all splits ───────────────────────────
    vol_path = CACHE_DIR / "preds" / f"{VOL_MODEL}.txt"
    if vol_path.exists():
        print(f"  Loading vol model: {VOL_MODEL}")
        vol_model = lgb.Booster(model_file=str(vol_path))
    else:
        print(f"  Training vol model: {VOL_MODEL} ...")
        y_atr_tr  = all_vol["atr_30"][np.array([ts_map[t] for t in ts_train])]
        y_atr_v   = all_vol["atr_30"][np.array([ts_map[t] for t in ts_val])]
        ok_tr     = ~np.isnan(y_atr_tr)
        ok_v      = ~np.isnan(y_atr_v)
        ds_tr     = lgb.Dataset(X_train[ok_tr], label=y_atr_tr[ok_tr])
        ds_v      = lgb.Dataset(X_val[ok_v],    label=y_atr_v[ok_v], reference=ds_tr)
        vp        = dict(_VOL_LGB_PARAMS)
        vol_model = lgb.train(vp, ds_tr, num_boost_round=500,
                              valid_sets=[ds_v],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(-1)])
        vol_model.save_model(str(vol_path))

    atr_train = vol_model.predict(X_train)
    atr_val   = vol_model.predict(X_val)
    atr_test  = vol_model.predict(X_test)

    # normalise ATR predictions to [0,1] using train percentile rank
    from scipy.stats import rankdata
    atr_all  = np.concatenate([atr_train, atr_val, atr_test])
    atr_rank_train = rankdata(atr_train) / len(atr_train)
    # for val/test: rank relative to train distribution
    atr_rank_val  = np.searchsorted(np.sort(atr_train), atr_val) / len(atr_train)
    atr_rank_test = np.searchsorted(np.sort(atr_train), atr_test) / len(atr_train)
    atr_rank_val  = np.clip(atr_rank_val, 0, 1)
    atr_rank_test = np.clip(atr_rank_test, 0, 1)

    print(f"\n  ATR prediction stats (percentile rank):")
    print(f"    Train  p25={np.percentile(atr_rank_train,25):.2f}  "
          f"p50={np.percentile(atr_rank_train,50):.2f}  "
          f"p75={np.percentile(atr_rank_train,75):.2f}")
    print(f"    Val    p25={np.percentile(atr_rank_val,25):.2f}  "
          f"p50={np.percentile(atr_rank_val,50):.2f}  "
          f"p75={np.percentile(atr_rank_val,75):.2f}")
    print(f"    Test   p25={np.percentile(atr_rank_test,25):.2f}  "
          f"p50={np.percentile(atr_rank_test,50):.2f}  "
          f"p75={np.percentile(atr_rank_test,75):.2f}")

    # ── Append ATR rank as extra feature ─────────────────────────────────────
    X_train_2s = np.column_stack([X_train, atr_rank_train])
    X_val_2s   = np.column_stack([X_val,   atr_rank_val])
    X_test_2s  = np.column_stack([X_test,  atr_rank_test])

    # CNN-LSTM: append ATR rank to sequential features
    sel_idx    = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    n_feat_seq = len(sel_idx)
    # add ATR rank as the last sequential feature (same value repeated per step)
    X_tr_seq   = X_train[:, sel_idx]
    X_v_seq    = X_val[:,   sel_idx]
    X_te_seq   = X_test[:,  sel_idx]

    # for sequences: broadcast ATR rank to (seq_len, 1) appended to each timestep
    def _add_atr_to_seq(X_seq_base, atr_rank_flat, seq_len):
        n = len(X_seq_base)
        atr_col = atr_rank_flat[seq_len:].reshape(-1, 1)[:len(X_seq_base)]
        # tile to (n, seq_len, 1)
        atr_tiled = np.tile(atr_col[:, np.newaxis, :], (1, seq_len, 1))
        return np.concatenate([X_seq_base, atr_tiled], axis=2)

    results = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n{'─'*65}")
            print(f"  {col}")
            print(f"{'─'*65}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            # ── BASELINE: LightGBM without ATR ────────────────────────────────
            lgbm_b = _train_lgbm(X_train[ok_tr], y_tr[ok_tr],
                                  X_val[ok_val],  y_v[ok_val])
            pb_lgbm_v  = lgbm_b.predict(X_val[ok_val])
            pb_lgbm_te = lgbm_b.predict(X_test[ok_te])

            # ── TWO-STAGE: LightGBM with ATR ──────────────────────────────────
            lgbm_ts = _train_lgbm(X_train_2s[ok_tr], y_tr[ok_tr],
                                   X_val_2s[ok_val],  y_v[ok_val])
            pts_lgbm_v  = lgbm_ts.predict(X_val_2s[ok_val])
            pts_lgbm_te = lgbm_ts.predict(X_test_2s[ok_te])

            print(f"  LightGBM: baseline val={_auc(y_v[ok_val],pb_lgbm_v):.4f} "
                  f"test={_auc(y_te[ok_te],pb_lgbm_te):.4f} | "
                  f"2-stage val={_auc(y_v[ok_val],pts_lgbm_v):.4f} "
                  f"test={_auc(y_te[ok_te],pts_lgbm_te):.4f}")

            # ── CNN-LSTM baseline (load cached) ───────────────────────────────
            cnn_path = CACHE_DIR / "preds" / f"{ticker}_cnn_dir_{direction}_{H}.keras"
            cnn_ts_path = CACHE_DIR / "preds" / f"{ticker}_cnn2s_dir_{direction}_{H}.keras"

            Xs_tr_b, ys_tr = _build_sequences(X_tr_seq, y_tr, SEQ_LEN)
            Xs_v_b,  ys_v  = _build_sequences(X_v_seq,  y_v,  SEQ_LEN)
            Xs_te_b, ys_te = _build_sequences(X_te_seq, y_te, SEQ_LEN)

            if cnn_path.exists():
                cnn_b = tf.keras.models.load_model(str(cnn_path))
            else:
                cnn_b = build_cnn_lstm(SEQ_LEN, n_feat_seq)
                _fit(cnn_b, Xs_tr_b, ys_tr, Xs_v_b, ys_v)

            pb_cnn_v  = cnn_b.predict(Xs_v_b,  verbose=0).flatten()
            pb_cnn_te = cnn_b.predict(Xs_te_b, verbose=0).flatten()

            # ── CNN-LSTM two-stage (with ATR rank appended) ───────────────────
            Xs_tr_ts = _add_atr_to_seq(Xs_tr_b, atr_rank_train, SEQ_LEN)
            Xs_v_ts  = _add_atr_to_seq(Xs_v_b,  atr_rank_val,   SEQ_LEN)
            Xs_te_ts = _add_atr_to_seq(Xs_te_b, atr_rank_test,  SEQ_LEN)

            if cnn_ts_path.exists():
                print(f"  Loading cached 2-stage CNN: {cnn_ts_path.name}")
                cnn_ts = tf.keras.models.load_model(str(cnn_ts_path))
            else:
                print(f"  Training 2-stage CNN ...")
                cnn_ts = build_cnn_lstm(SEQ_LEN, n_feat_seq + 1)
                _fit(cnn_ts, Xs_tr_ts, ys_tr, Xs_v_ts, ys_v)
                cnn_ts.save(str(cnn_ts_path))
                print(f"  Saved → {cnn_ts_path.name}")

            pts_cnn_v  = cnn_ts.predict(Xs_v_ts,  verbose=0).flatten()
            pts_cnn_te = cnn_ts.predict(Xs_te_ts, verbose=0).flatten()

            print(f"  CNN-LSTM: baseline val={_auc(ys_v,pb_cnn_v):.4f} "
                  f"test={_auc(ys_te,pb_cnn_te):.4f} | "
                  f"2-stage val={_auc(ys_v,pts_cnn_v):.4f} "
                  f"test={_auc(ys_te,pts_cnn_te):.4f}")

            # ── align and build ensembles ──────────────────────────────────────
            cnn_vi  = np.array([i for i in range(SEQ_LEN, len(X_val))  if not np.isnan(y_v[i])])
            cnn_ti  = np.array([i for i in range(SEQ_LEN, len(X_test)) if not np.isnan(y_te[i])])
            lvi     = {o: p for p, o in enumerate(np.where(ok_val)[0])}
            lti     = {o: p for p, o in enumerate(np.where(ok_te)[0])}
            sv      = [i for i in cnn_vi if i in lvi]
            st      = [i for i in cnn_ti if i in lti]
            cvm     = {o: p for p, o in enumerate(cnn_vi)}
            ctm     = {o: p for p, o in enumerate(cnn_ti)}

            def _get(sv_, lp, lm, cp, cm, y_arr):
                return (np.array([lp[lm[i]] for i in sv_]),
                        np.array([cp[cm[i]] for i in sv_]),
                        y_arr[np.array(sv_)])

            lb_sv, cb_sv, y_sv = _get(sv, pb_lgbm_v,  lvi, pb_cnn_v,  cvm, y_v)
            lb_st, cb_st, y_st = _get(st, pb_lgbm_te, lti, pb_cnn_te, ctm, y_te)
            lt_sv, ct_sv, _    = _get(sv, pts_lgbm_v,  lvi, pts_cnn_v,  cvm, y_v)
            lt_st, ct_st, _    = _get(st, pts_lgbm_te, lti, pts_cnn_te, ctm, y_te)

            # weights from baseline val AUC
            wl = _auc(y_sv, lb_sv) / (_auc(y_sv, lb_sv) + _auc(y_sv, cb_sv))
            wc = 1 - wl

            ens_b_v  = wl * lb_sv + wc * cb_sv
            ens_b_t  = wl * lb_st + wc * cb_st
            ens_ts_v = wl * lt_sv + wc * ct_sv
            ens_ts_t = wl * lt_st + wc * ct_st

            t_b  = _best_threshold(y_sv, ens_b_v)
            t_ts = _best_threshold(y_sv, ens_ts_v)

            baseline_v  = _cm_row(y_sv, ens_b_v,  t_b)
            two_stage_v = _cm_row(y_sv, ens_ts_v, t_ts)
            baseline_t  = _cm_row(y_st, ens_b_t,  t_b)
            two_stage_t = _cm_row(y_st, ens_ts_t, t_ts)

            _print_comparison(col, "val",  baseline_v,  two_stage_v,  t_b, t_ts)
            _print_comparison(col, "test", baseline_t,  two_stage_t,  t_b, t_ts)

            results.append({
                "label": col,
                "base_val_auc":  baseline_v[0],  "ts_val_auc":  two_stage_v[0],
                "base_test_auc": baseline_t[0],  "ts_test_auc": two_stage_t[0],
                "base_test_prec": baseline_t[1], "ts_test_prec": two_stage_t[1],
                "base_test_f1":  baseline_t[3],  "ts_test_f1":  two_stage_t[3],
            })

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n\n{'='*65}")
        print(f"  TWO-STAGE SUMMARY (test set)")
        print(f"{'='*65}")
        print(f"  {'Label':<12}  {'AUC base':>9}  {'AUC 2s':>8}  "
              f"{'Prec base':>10}  {'Prec 2s':>9}  {'F1 base':>8}  {'F1 2s':>7}")
        print(f"  {'─'*72}")
        for _, r in df.iterrows():
            print(f"  {r['label']:<12}  {r['base_test_auc']:>9.4f}  {r['ts_test_auc']:>8.4f}  "
                  f"{r['base_test_prec']:>10.3f}  {r['ts_test_prec']:>9.3f}  "
                  f"{r['base_test_f1']:>8.3f}  {r['ts_test_f1']:>7.3f}")

        out = CACHE_DIR / "lookup" / f"{ticker}_two_stage_eval.parquet"
        df.to_parquet(out, index=False)
        print(f"\nResults → {out.name}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
