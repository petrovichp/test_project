"""
Ensemble — weighted average of LightGBM + CNN-LSTM (two-stage pipeline).

ATR rank prediction from btc_lgbm_atr_30 is permanently included as an
extra feature in both models (two-stage pipeline, improves AUC by +0.008–0.050).

Confusion matrices shown for val and test at:
  - optimal F1 threshold (set on val)
  - precision >= 0.60 threshold (set on val)

Run: python3 -m models.ensemble [ticker]
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
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
from models.volatility import _compute_targets as _vol_targets, _LGB_PARAMS as _VOL_PARAMS

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
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                prec=prec, rec=rec, f1=f1, auc=_auc(y_true, y_prob),
                n_pos=int(p.sum()), threshold=t)


def _print_cm(label, split, t_label, models_data):
    names = list(models_data.keys())
    cw    = 24
    print(f"\n  {'─'*76}")
    print(f"  {label}  |  {split.upper()}  |  {t_label}")
    print(f"  {'─'*76}")
    print(f"  {'':20}", end="")
    for n in names:
        print(f"  {n:>{cw}}", end="")
    print()
    print(f"  {'':20}", end="")
    for n in names:
        t_str = "t=" + f"{models_data[n]['threshold']:.2f}"
        print(f"  {t_str:>{cw}}", end="")
    print()
    for row, nk, pk in [("Actual NEG", "tn", "fp"), ("Actual POS", "fn", "tp")]:
        print(f"  {row:<20}", end="")
        for n in names:
            d = models_data[n]
            cell = (f"TN={d['tn']:,}  FP={d['fp']:,}" if nk == "tn"
                    else f"FN={d['fn']:,}  TP={d['tp']:,}")
            print(f"  {cell:>{cw}}", end="")
        print()
    for m_lbl, key, fmt in [("AUC","auc",".4f"),("Precision","prec",".3f"),
                              ("Recall","rec",".3f"),("F1","f1",".3f"),
                              ("Pred pos","n_pos",",d")]:
        print(f"  {m_lbl:<20}", end="")
        for n in names:
            print(f"  {format(models_data[n][key], fmt):>{cw}}", end="")
        print()


# ── ATR rank feature ──────────────────────────────────────────────────────────

def _get_atr_rank(X_train, X_val, X_test, ticker):
    vol_path = CACHE_DIR / f"{ticker}_lgbm_atr_30.txt"
    if vol_path.exists():
        vol_model = lgb.Booster(model_file=str(vol_path))
    else:
        print("  ATR model not found — run models.volatility first")
        return (np.zeros(len(X_train)), np.zeros(len(X_val)),
                np.zeros(len(X_test)))

    atr_tr = vol_model.predict(X_train)
    atr_v  = vol_model.predict(X_val)
    atr_te = vol_model.predict(X_test)

    atr_rank_tr = np.argsort(np.argsort(atr_tr)) / len(atr_tr)
    atr_rank_v  = np.clip(
        np.searchsorted(np.sort(atr_tr), atr_v) / len(atr_tr), 0, 1)
    atr_rank_te = np.clip(
        np.searchsorted(np.sort(atr_tr), atr_te) / len(atr_tr), 0, 1)

    return atr_rank_tr, atr_rank_v, atr_rank_te


# ── sequence builder with ATR appended ────────────────────────────────────────

def _build_seqs_with_atr(X_seq_base, y, atr_rank_flat, seq_len):
    n = len(X_seq_base)
    idx = [i for i in range(seq_len, n) if not np.isnan(y[i])]
    if not idx:
        return np.empty((0, seq_len, X_seq_base.shape[1] + 1)), np.empty(0)
    atr_col = atr_rank_flat[np.array(idx), np.newaxis, np.newaxis]
    atr_tiled = np.tile(atr_col, (1, seq_len, 1))
    Xs = np.stack([X_seq_base[i - seq_len:i] for i in idx])
    return np.concatenate([Xs, atr_tiled], axis=2), y[np.array(idx)]


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*76}")
    print(f"  ENSEMBLE — LightGBM + CNN-LSTM  ({ticker.upper()})")
    print(f"  Two-stage: ATR rank included as permanent feature")
    print(f"{'='*76}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)
    fmt     = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    # ── ATR rank feature ──────────────────────────────────────────────────────
    print("  Getting ATR rank predictions ...")
    atr_tr, atr_v, atr_te = _get_atr_rank(X_train, X_val, X_test, ticker)
    print(f"  ATR rank — val p50={np.median(atr_v):.2f}  test p50={np.median(atr_te):.2f}")

    X_train_aug = np.column_stack([X_train, atr_tr])
    X_val_aug   = np.column_stack([X_val,   atr_v])
    X_test_aug  = np.column_stack([X_test,  atr_te])

    # CNN-LSTM sequential features + ATR
    sel_idx  = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_tr_seq = X_train[:, sel_idx]
    X_v_seq  = X_val[:,   sel_idx]
    X_te_seq = X_test[:,  sel_idx]

    reg  = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    rows = []

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n{'='*76}")
            print(f"  Label: {col}  (H={H} bars, threshold={THRESHOLD*100:.1f}%)")
            print(f"{'='*76}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            print(f"\n  Class balance — train {y_tr[ok_tr].mean():.1%} | "
                  f"val {y_v[ok_val].mean():.1%} | "
                  f"test {y_te[ok_te].mean():.1%}")

            # ── LightGBM (with ATR) ───────────────────────────────────────────
            print(f"\n  Training LightGBM (with ATR rank) ...")
            lgbm = _train_lgbm(X_train_aug[ok_tr], y_tr[ok_tr],
                               X_val_aug[ok_val],  y_v[ok_val])
            p_lgbm_v  = lgbm.predict(X_val_aug[ok_val])
            p_lgbm_te = lgbm.predict(X_test_aug[ok_te])
            auc_lgbm_v  = _auc(y_v[ok_val],  p_lgbm_v)
            auc_lgbm_te = _auc(y_te[ok_te],  p_lgbm_te)
            print(f"  LightGBM  val={auc_lgbm_v:.4f}  test={auc_lgbm_te:.4f}")

            # ── CNN-LSTM (two-stage, load cached) ─────────────────────────────
            cnn_path = CACHE_DIR / f"{ticker}_cnn2s_dir_{direction}_{H}.keras"
            if not cnn_path.exists():
                print(f"  CNN-LSTM two-stage model not found: {cnn_path.name}")
                print(f"  Run: python3 -m models.two_stage {ticker}")
                continue
            print(f"  Loading CNN-LSTM: {cnn_path.name}")
            cnn = tf.keras.models.load_model(str(cnn_path))

            Xs_v,  ys_v  = _build_seqs_with_atr(X_v_seq,  y_v,  atr_v,  SEQ_LEN)
            Xs_te, ys_te = _build_seqs_with_atr(X_te_seq, y_te, atr_te, SEQ_LEN)
            p_cnn_v  = cnn.predict(Xs_v,  verbose=0).flatten()
            p_cnn_te = cnn.predict(Xs_te, verbose=0).flatten()
            auc_cnn_v  = _auc(ys_v,  p_cnn_v)
            auc_cnn_te = _auc(ys_te, p_cnn_te)
            print(f"  CNN-LSTM  val={auc_cnn_v:.4f}  test={auc_cnn_te:.4f}")

            # ── align and ensemble ────────────────────────────────────────────
            cnn_vi  = np.array([i for i in range(SEQ_LEN, len(X_val))  if not np.isnan(y_v[i])])
            cnn_ti  = np.array([i for i in range(SEQ_LEN, len(X_test)) if not np.isnan(y_te[i])])
            lvi     = {o: p for p, o in enumerate(np.where(ok_val)[0])}
            lti     = {o: p for p, o in enumerate(np.where(ok_te)[0])}
            sv      = [i for i in cnn_vi if i in lvi]
            st      = [i for i in cnn_ti if i in lti]
            cvm     = {o: p for p, o in enumerate(cnn_vi)}
            ctm     = {o: p for p, o in enumerate(cnn_ti)}

            lp_sv = np.array([p_lgbm_v[lvi[i]]  for i in sv])
            cp_sv = np.array([p_cnn_v[cvm[i]]    for i in sv])
            lp_st = np.array([p_lgbm_te[lti[i]] for i in st])
            cp_st = np.array([p_cnn_te[ctm[i]]   for i in st])
            y_sv  = y_v[np.array(sv)]
            y_st  = y_te[np.array(st)]

            w_l   = _auc(y_sv, lp_sv) / (_auc(y_sv, lp_sv) + _auc(y_sv, cp_sv))
            w_c   = 1 - w_l
            ep_sv = w_l * lp_sv + w_c * cp_sv
            ep_st = w_l * lp_st + w_c * cp_st

            auc_ens_v  = _auc(y_sv, ep_sv)
            auc_ens_te = _auc(y_st, ep_st)
            print(f"  Ensemble  val={auc_ens_v:.4f}  test={auc_ens_te:.4f}  "
                  f"(w_lgbm={w_l:.2f}  w_cnn={w_c:.2f})")

            # ── confusion matrices ─────────────────────────────────────────────
            t_f1   = _best_threshold(y_sv, ep_sv)
            t_prec = _prec_threshold(y_sv, ep_sv, 0.60)

            for split, yt, lp, cp, ep in [
                ("val",  y_sv, lp_sv, cp_sv, ep_sv),
                ("test", y_st, lp_st, cp_st, ep_st),
            ]:
                for t_lbl, tv in [("optimal F1", t_f1),
                                   ("precision≥0.60", t_prec)]:
                    _print_cm(col, split, t_lbl, {
                        "LightGBM": _cm_block(yt, lp, tv),
                        "CNN-LSTM":  _cm_block(yt, cp, tv),
                        "Ensemble":  _cm_block(yt, ep, tv),
                    })

            # ── register ──────────────────────────────────────────────────────
            code = f"{ticker}_ens2s_dir_{direction}_{H}"
            reg[code] = {
                "ticker": ticker, "model_type": "ens2s",
                "target": f"dir_{direction}", "horizon": H,
                "val_auc":  round(auc_ens_v,  4),
                "test_auc": round(auc_ens_te, 4),
                "w_lgbm": round(w_l, 3), "w_cnn": round(w_c, 3),
                "atr_feature": True,
                "trained_at": datetime.utcnow().isoformat(),
            }
            REGISTRY.write_text(json.dumps(reg, indent=2))

            rows.append({"label": col,
                         "lgbm_val": auc_lgbm_v, "lgbm_test": auc_lgbm_te,
                         "cnn_val":  auc_cnn_v,  "cnn_test":  auc_cnn_te,
                         "ens_val":  auc_ens_v,  "ens_test":  auc_ens_te})

    # ── final summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n\n{'='*76}")
        print(f"  FINAL AUC SUMMARY (two-stage ensemble)")
        print(f"{'='*76}")
        print(f"  {'Label':<12}  {'LGB val':>8}  {'LGB test':>9}  "
              f"{'CNN val':>8}  {'CNN test':>9}  {'Ens val':>8}  {'Ens test':>9}")
        print(f"  {'─'*76}")
        for _, r in df.iterrows():
            print(f"  {r['label']:<12}  {r['lgbm_val']:>8.4f}  {r['lgbm_test']:>9.4f}  "
                  f"{r['cnn_val']:>8.4f}  {r['cnn_test']:>9.4f}  "
                  f"{r['ens_val']:>8.4f}  {r['ens_test']:>9.4f}")

        out = CACHE_DIR / f"{ticker}_ensemble_eval.parquet"
        df.to_parquet(out, index=False)
        print(f"\nResults → {out.name}")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
