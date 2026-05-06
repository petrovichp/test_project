"""
Direction CNN-LSTM v4 — retrained on the same vol-train chunk as vol_v4.

DQN-v5 variant: ignores `clean_mask`. Treats the 384k feature parquet as a
continuous index sequence; only the first 1,440 warmup rows are dropped.

Bar layout (matches vol_v4 / regime_cusum_v4):
  [0,       1,440)         warmup       — dropped
  [1,440,   91,440)        dir-train    — 90,000 bars  CNN-LSTM fits here
  [91,440,  101,440)       dir-holdout  — 10,000 bars  early-stopping val
  [101,440, 384,614)       RL period    — 283,174 bars inference
                                          (DQN-train + DQN-val + DQN-test)

Two-stage architecture (matches old `cnn2s_*` models):
  inputs = SEQ_LEN × (SEQ_FEATURES + vol-rank channel)

Vol-rank source: cache/btc_pred_vol_v4.npz (`rank` field, indexed from bar 1,440).
NOTE: rank values for the dir-holdout (bars 91,440–101,440) are in-sample-vol
because vol_v4 was trained on the full vol-train chunk. This minorly inflates
the holdout AUC used for early stopping, but the locked DQN-test evaluation
(bars 332,307+) uses honest OOS vol rank. Documented intentionally.

Outputs:
  cache/btc_cnn2s_dir_{up,down}_{60,100}_v4.keras
  cache/btc_pred_dir_{up,down}_{60,100}_v4.npz   — preds aligned to bar 1,440+
                                                    (full 383k array;
                                                     bars 1,440-101,440 are in-sample,
                                                     bars 101,440+ are honest OOS)

Run: python3 -m models.direction_dl_v4 [ticker]
"""

import sys, time, json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from tensorflow.keras import layers, Input, Model, callbacks
from sklearn.metrics    import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader        import load_meta
from models.direction_dl import SEQ_FEATURES, SEQ_LEN, HORIZONS, THRESHOLD, build_cnn_lstm

CACHE        = ROOT / "cache"
WARMUP       = 1440
DIR_TRAIN_E  = 91_440          # bars 1,440  → 91,440  : direction CNN-LSTM training
VOL_TRAIN_E  = 101_440         # bars 91,440 → 101,440 : holdout (early stopping)


# ── label computation ─────────────────────────────────────────────────────────

def _compute_labels(price: np.ndarray) -> dict:
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price)
    out = {}
    for H in HORIZONS:
        y_up   = np.full(n, np.nan)
        y_down = np.full(n, np.nan)
        if n > H:
            wins    = sliding_window_view(price[1:], H)
            ret_max = wins.max(axis=1) / price[:n - H] - 1
            ret_min = wins.min(axis=1) / price[:n - H] - 1
            y_up  [:n - H] = (ret_max >  THRESHOLD).astype(float)
            y_down[:n - H] = (ret_min < -THRESHOLD).astype(float)
        out[f"up_{H}"]   = y_up
        out[f"down_{H}"] = y_down
    return out


# ── sequence assembly ────────────────────────────────────────────────────────

def _build_sequences(X_with_rank: np.ndarray, y: np.ndarray, seq_len: int):
    """Returns (Xs, ys, idx_keep). idx_keep is the absolute index in X_with_rank."""
    n = len(X_with_rank)
    keep = [i for i in range(seq_len, n) if not np.isnan(y[i])]
    if not keep:
        return (np.empty((0, seq_len, X_with_rank.shape[1])),
                np.empty(0), np.array([], dtype=np.int64))
    keep = np.asarray(keep, dtype=np.int64)
    Xs   = np.stack([X_with_rank[i - seq_len:i] for i in keep])
    return Xs, y[keep], keep


def _predict_sequences(model, X_with_rank: np.ndarray, seq_len: int,
                        batch: int = 4096) -> np.ndarray:
    """Predict every bar t ≥ seq_len. Returns array of length n with NaN at
    positions [0, seq_len)."""
    n = len(X_with_rank)
    out = np.full(n, np.nan, dtype=np.float32)
    for s in range(seq_len, n, batch):
        idxs = np.arange(s, min(s + batch, n), dtype=np.int64)
        Xs   = np.stack([X_with_rank[i - seq_len:i] for i in idxs])
        out[idxs] = model.predict(Xs, verbose=0).flatten()
    return out


# ── progress callback ────────────────────────────────────────────────────────

class _Prog(callbacks.Callback):
    def __init__(self, total: int, label: str):
        super().__init__(); self.total = total; self.label = label
        self.t0 = datetime.utcnow()
    def on_epoch_end(self, epoch, logs=None):
        m, s = divmod((datetime.utcnow() - self.t0).seconds, 60)
        auc_tr = logs.get("auc",     logs.get("AUC", 0))
        auc_v  = logs.get("val_auc", logs.get("val_AUC", 0))
        print(f"  [{self.label}] ep {epoch+1:>2}/{self.total}  "
              f"loss={logs.get('loss',0):.4f}  auc={auc_tr:.4f}  "
              f"val_auc={auc_v:.4f}  [{m:02d}:{s:02d}]", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  DIRECTION CNN-LSTM v4 — {ticker.upper()}\n{'='*70}")

    # ── load aligned source data ─────────────────────────────────────────────
    pq        = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    feat_cols = [c for c in pq.columns if c != "timestamp"]
    sel_idx   = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    miss      = [f for f in SEQ_FEATURES if f not in feat_cols]
    print(f"  feat parquet: {pq.shape}  SEQ_FEATURES used: {len(sel_idx)}  missing: {miss}")

    meta  = load_meta(ticker)
    assert (meta["timestamp"].values == pq["timestamp"].values).all()
    price = meta["perp_ask_price"].values

    # ── labels (computed on raw price, no NaN past warmup until tail H) ──────
    labels = _compute_labels(price)

    # ── feature subset (bars 1,440 → end) + vol-rank channel ─────────────────
    X_full = pq[feat_cols].values[:, sel_idx]            # (384,614, 30)
    n_full = len(X_full)
    nan_post = np.isnan(X_full[WARMUP:]).any(axis=1).sum()
    assert nan_post == 0, f"NaN past warmup: {nan_post}"

    # vol-rank from npz: indexed from bar WARMUP
    vol  = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    rank = vol["rank"]                                    # length 383,174 (bars 1,440 → end)
    assert len(rank) == n_full - WARMUP, "vol-rank length mismatch"
    rank_full = np.full(n_full, 0.5, dtype=np.float32)    # bars [0, 1440) → 0.5 placeholder
    rank_full[WARMUP:] = rank

    # ── scaler fit on dir-train slice ─────────────────────────────────────────
    sc = StandardScaler()
    sc.fit(X_full[WARMUP:DIR_TRAIN_E])
    Xs = sc.transform(X_full).astype(np.float32)          # (n_full, 30)

    # concat vol-rank as 31st channel
    X_2s = np.concatenate([Xs, rank_full[:, None]], axis=1).astype(np.float32)
    n_feat_2s = X_2s.shape[1]
    print(f"  X_2s shape: {X_2s.shape}  (SEQ_FEATURES + vol-rank)")

    # ── train each (direction × horizon) model ────────────────────────────────
    reg_summary = {}
    for H in HORIZONS:
        for d in ["up", "down"]:
            col   = f"{d}_{H}"
            label = f"{col}_v4"
            mp    = CACHE / f"{ticker}_cnn2s_dir_{col}_v4.keras"
            pp    = CACHE / f"{ticker}_pred_dir_{col}_v4.npz"

            print(f"\n{'─'*70}\n  MODEL  {label}\n{'─'*70}")

            y = labels[col]
            # slices of indices (absolute)
            idx_tr  = np.arange(WARMUP,      DIR_TRAIN_E)
            idx_ho  = np.arange(DIR_TRAIN_E, VOL_TRAIN_E)

            X_tr    = X_2s[idx_tr]
            y_tr    = y[idx_tr]
            X_ho    = X_2s[idx_ho]
            y_ho    = y[idx_ho]

            Xs_tr, ys_tr, _ = _build_sequences(X_tr, y_tr, SEQ_LEN)
            Xs_ho, ys_ho, _ = _build_sequences(X_ho, y_ho, SEQ_LEN)
            print(f"  dir-train seqs: {len(ys_tr):,}  pos={ys_tr.mean():.1%}")
            print(f"  dir-holdout seqs: {len(ys_ho):,}  pos={ys_ho.mean():.1%}")

            if mp.exists():
                print(f"  Loading cached model: {mp.name}")
                model = tf.keras.models.load_model(str(mp))
            else:
                t1 = time.perf_counter()
                model = build_cnn_lstm(SEQ_LEN, n_feat_2s)
                pos = ys_tr.mean()
                cw  = {0: 1.0, 1: float((1 - pos) / (pos + 1e-9))}
                model.fit(
                    Xs_tr, ys_tr,
                    validation_data = (Xs_ho, ys_ho),
                    epochs    = 20,
                    batch_size= 256,
                    class_weight = cw,
                    verbose   = 0,
                    callbacks = [
                        _Prog(20, label),
                        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                                  restore_best_weights=True,
                                                  mode="max", verbose=0),
                    ],
                )
                model.save(str(mp))
                print(f"  trained in {time.perf_counter()-t1:.1f}s  → {mp.name}")

            prob_tr = model.predict(Xs_tr, verbose=0, batch_size=512).flatten()
            prob_ho = model.predict(Xs_ho, verbose=0, batch_size=512).flatten()
            auc_tr  = (roc_auc_score(ys_tr, prob_tr) if len(np.unique(ys_tr)) > 1
                       else float("nan"))
            auc_ho  = (roc_auc_score(ys_ho, prob_ho) if len(np.unique(ys_ho)) > 1
                       else float("nan"))
            print(f"  AUC  dir-train (in-sample) = {auc_tr:.4f}  "
                  f"dir-holdout = {auc_ho:.4f}  gap={abs(auc_tr-auc_ho):.3f}")

            # ── predict full sequence (bars [WARMUP, n_full]) ───────────────
            #   - bars [WARMUP, WARMUP+SEQ_LEN) get NaN (lookback underflow)
            #   - bars [WARMUP+SEQ_LEN, n_full) get a proba
            t1 = time.perf_counter()
            X_eval = X_2s[WARMUP:]                          # (383,174, 31)
            preds  = _predict_sequences(model, X_eval, SEQ_LEN)
            preds  = pd.Series(preds).bfill().fillna(0.5).values.astype(np.float32)
            print(f"  inference {len(preds):,} bars in {time.perf_counter()-t1:.1f}s")

            # informational AUC over RL period (bars 101,440 → end), where labels exist
            rl_a = VOL_TRAIN_E - WARMUP
            y_rl = y[WARMUP:][rl_a:]
            p_rl = preds[rl_a:]
            ok   = ~np.isnan(y_rl)
            auc_rl = (roc_auc_score(y_rl[ok], p_rl[ok])
                      if ok.sum() > 100 and len(np.unique(y_rl[ok])) > 1
                      else float("nan"))
            print(f"  AUC  RL period (out-of-sample) = {auc_rl:.4f}")

            np.savez(
                pp,
                preds  = preds,
                ts     = pq["timestamp"].values[WARMUP:].astype(np.int64),
                warmup = WARMUP,
                dir_train_e = DIR_TRAIN_E,
                vol_train_e = VOL_TRAIN_E,
                auc_train   = auc_tr,
                auc_holdout = auc_ho,
                auc_rl      = auc_rl,
            )
            print(f"  → {pp.name}")
            reg_summary[col] = {"auc_train": float(auc_tr),
                                 "auc_holdout": float(auc_ho),
                                 "auc_rl": float(auc_rl)}

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  SUMMARY\n{'='*70}")
    print(f"  {'Label':<10} {'AUC train':>10}  {'AUC holdout':>12}  {'AUC RL':>8}")
    for col, m in reg_summary.items():
        print(f"  {col:<10} {m['auc_train']:>10.4f}  {m['auc_holdout']:>12.4f}  "
              f"{m['auc_rl']:>8.4f}")
    print(f"\n  Total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
