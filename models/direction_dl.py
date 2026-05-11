"""
Direction prediction — DL models (CNN-LSTM and DeepLOB).

Model codes (saved to cache/, registered in model_registry.json):
  {ticker}_cnn_dir_up_{H}   e.g. btc_cnn_dir_up_60
  {ticker}_dlob_dir_dn_{H}  e.g. btc_dlob_dir_dn_60

Training progress printed per epoch.
Confusion matrices printed after each label finishes.

Run: python3 -m models.direction_dl [ticker] [cnn_lstm|deeplob|both]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, callbacks
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta, load
from data.gaps import clean_mask as _cmask
from models.splits import sequential
from features.assembly import assemble

CACHE_DIR  = Path(__file__).parent.parent / "cache"
REGISTRY   = Path(__file__).parent.parent / "model_registry.json"

SEQ_LEN    = 30
THRESHOLD  = 0.008
HORIZONS   = [60, 100]
OB_LEVELS  = 10

SEQ_FEATURES = [
    # OB imbalance (original buckets)
    "ob_imb_perp_0_50", "ob_imb_perp_50_100",
    "ob_imb_spot_0_50",
    # True OFI — new
    "ofi_perp_10", "ofi_perp_10_r5", "ofi_perp_10_r15",
    "ofi_spot_10", "ofi_spot_10_r5",
    # Depth band imbalance near mid — new
    "band_imb_perp_0_5", "band_imb_perp_5_20",
    "band_imb_spot_0_5",
    # Spot/perp divergence — new
    "ob_spot_perp_imb_div",
    # Book shape — new
    "near_conc_bid_perp", "near_conc_ask_perp",
    # Span
    "span_perp",
    # Price
    "ret_1", "ret_5", "ret_15", "ret_30",
    "rsi_6", "macd_hist", "bb_width",
    "vwap_dev_60", "vwap_dev_240",
    # Volume / taker
    "taker_imb_1", "taker_imb_5", "taker_net_15",
    # Market
    "fund_rate", "fund_mom_480",
    "oi_vel_5",
]


# ── registry ──────────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}

def _save_registry(reg: dict):
    REGISTRY.write_text(json.dumps(reg, indent=2))

def _model_code(ticker, model_type, direction, H):
    return f"{ticker}_{model_type}_dir_{direction}_{H}"

def _model_path(code):
    return CACHE_DIR / "preds" / f"{code}.keras"


# ── label computation ─────────────────────────────────────────────────────────

def _compute_labels(price_arr):
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(price_arr)
    labels = {}
    for H in HORIZONS:
        y_up = y_down = np.full(n, np.nan)
        if n > H:
            wins    = sliding_window_view(price_arr[1:], H)
            ret_max = wins.max(axis=1) / price_arr[:n - H] - 1
            ret_min = wins.min(axis=1) / price_arr[:n - H] - 1
            y_up    = np.full(n, np.nan); y_up[:n - H]   = (ret_max >  THRESHOLD).astype(float)
            y_down  = np.full(n, np.nan); y_down[:n - H] = (ret_min < -THRESHOLD).astype(float)
        labels[f"up_{H}"]   = y_up
        labels[f"down_{H}"] = y_down
    return labels


# ── sequence builder ──────────────────────────────────────────────────────────

def _build_sequences(X, y, seq_len):
    n = len(X)
    idx = [i for i in range(seq_len, n) if not np.isnan(y[i])]
    if not idx:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    return np.stack([X[i - seq_len:i] for i in idx]), y[np.array(idx)]


# ── architectures ─────────────────────────────────────────────────────────────

def build_cnn_lstm(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.GRU(64, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


def _inception_block(x, filters=16):
    b1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    b2 = layers.Conv2D(filters, (3, 1), padding="same", activation="relu")(x)
    b3 = layers.MaxPooling2D((3, 1), strides=(1, 1), padding="same")(x)
    b3 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(b3)
    return layers.Concatenate()([b1, b2, b3])


def build_deeplob(seq_len, ob_levels):
    inp = Input(shape=(seq_len, ob_levels * 4, 1))
    x = layers.Conv2D(16, (1, 2), strides=(1, 2), activation="relu")(inp)
    x = layers.Conv2D(16, (3, 1), padding="same", activation="relu")(x)
    x = _inception_block(x, 16)
    x = layers.Reshape((x.shape[1], -1))(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


# ── training ──────────────────────────────────────────────────────────────────

class EpochProgress(callbacks.Callback):
    """Prints one clear line per epoch so background output is readable."""
    def __init__(self, total):
        super().__init__()
        self.total = total
        self.t0    = datetime.utcnow()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = (datetime.utcnow() - self.t0).seconds
        mins, secs = divmod(elapsed, 60)
        auc_tr = logs.get("auc", logs.get("AUC", 0))
        auc_v  = logs.get("val_auc", logs.get("val_AUC", 0))
        loss   = logs.get("loss", 0)
        print(f"  ep {epoch+1:>2}/{self.total}  "
              f"loss={loss:.4f}  auc={auc_tr:.4f}  val_auc={auc_v:.4f}  "
              f"[{mins:02d}:{secs:02d}]", flush=True)


def _fit(model, X_tr, y_tr, X_val, y_val, epochs=20, batch=256):
    pos   = y_tr.mean()
    cw    = {0: 1.0, 1: float((1 - pos) / (pos + 1e-9))}
    cb    = [
        EpochProgress(epochs),
        callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                restore_best_weights=True, mode="max",
                                verbose=0),
    ]
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch,
              callbacks=cb, verbose=0,
              class_weight=cw)
    return model


# ── confusion matrix ──────────────────────────────────────────────────────────

def _best_f1_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        if preds.sum() == 0:
            continue
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def _print_cm(label, split, y_true, y_prob, threshold):
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    print(f"\n  [{split.upper()}]  label={label}  t={threshold:.2f}")
    print(f"  {'':12}  Pred NEG    Pred POS")
    print(f"  Actual NEG    {tn:>8,}    {fp:>8,}")
    print(f"  Actual POS    {fn:>8,}    {tp:>8,}")
    print(f"  AUC={auc:.4f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  "
          f"Pred-pos={preds.sum():,}")


def _auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")


# ── CNN-LSTM run ──────────────────────────────────────────────────────────────

def run_cnn_lstm(ticker):
    print(f"\n{'='*60}")
    print(f"  CNN-LSTM — {ticker.upper()}")
    print(f"  SEQ_LEN={SEQ_LEN}  features={len(SEQ_FEATURES)}")
    print(f"{'='*60}")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    sel_idx  = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    missing  = [f for f in SEQ_FEATURES if f not in feat_cols]
    if missing:
        print(f"  Missing features: {missing}")
    X_tr = X_train[:, sel_idx]
    X_v  = X_val[:,   sel_idx]
    X_te = X_test[:,  sel_idx]
    n_feat = X_tr.shape[1]

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)
    fmt     = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    reg     = _load_registry()

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col  = f"{direction}_{H}"
            code = _model_code(ticker, "cnn", direction, H)
            path = _model_path(code)

            print(f"\n{'─'*60}")
            print(f"  {code}  ({fmt(ts_train[0])} → {fmt(ts_test[-1])})")
            print(f"{'─'*60}")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train);  ok_tr  = ~np.isnan(y_tr)
            y_v  = _lbl(ts_val);    ok_val = ~np.isnan(y_v)
            y_te = _lbl(ts_test);   ok_te  = ~np.isnan(y_te)

            Xs_tr, ys_tr = _build_sequences(X_tr, y_tr, SEQ_LEN)
            Xs_v,  ys_v  = _build_sequences(X_v,  y_v,  SEQ_LEN)
            Xs_te, ys_te = _build_sequences(X_te, y_te, SEQ_LEN)

            print(f"  Train seqs: {len(ys_tr):,}  pos={ys_tr.mean():.1%}")
            print(f"  Val seqs  : {len(ys_v):,}   Test seqs: {len(ys_te):,}")
            print()

            if path.exists():
                print(f"  Loading cached model: {path.name}")
                model = tf.keras.models.load_model(str(path))
            else:
                model = build_cnn_lstm(SEQ_LEN, n_feat)
                model = _fit(model, Xs_tr, ys_tr, Xs_v, ys_v)
                model.save(str(path))
                print(f"  Saved → {path.name}")

            prob_v  = model.predict(Xs_v,  verbose=0).flatten()
            prob_te = model.predict(Xs_te, verbose=0).flatten()

            auc_v  = _auc(ys_v,  prob_v)
            auc_te = _auc(ys_te, prob_te)
            print(f"\n  AUC — val={auc_v:.4f}  test={auc_te:.4f}  "
                  f"gap={abs(auc_v-auc_te):.3f}")

            t_opt = _best_f1_threshold(ys_v, prob_v)
            _print_cm(col, "val",  ys_v,  prob_v,  t_opt)
            _print_cm(col, "test", ys_te, prob_te, t_opt)

            reg[code] = {
                "ticker": ticker, "model_type": "cnn",
                "target": f"dir_{direction}", "horizon": H,
                "val_auc":  round(auc_v,  4),
                "test_auc": round(auc_te, 4),
                "seq_len":  SEQ_LEN,
                "n_features": n_feat,
                "trained_at": datetime.utcnow().isoformat(),
                "model_file": path.name,
            }
            _save_registry(reg)
            print(f"\n  Registered: {code}")


# ── DeepLOB run ───────────────────────────────────────────────────────────────

def run_deeplob(ticker):
    print(f"\n{'='*60}")
    print(f"  DeepLOB — {ticker.upper()}")
    print(f"  SEQ_LEN={SEQ_LEN}  OB_LEVELS={OB_LEVELS}")
    print(f"{'='*60}")

    ob_cache = CACHE_DIR / "preds" / f"{ticker}_deeplob_ob{OB_LEVELS}_scaled.npz"

    meta    = load_meta(ticker)
    ts_map  = dict(zip(meta["timestamp"].values, range(len(meta))))
    price   = meta["perp_ask_price"].values
    all_lbl = _compute_labels(price)

    assembled = pd.read_parquet(CACHE_DIR / "features" / f"{ticker}_features_assembled.parquet")
    fc        = [c for c in assembled.columns if c != "timestamp"]
    gap_ok    = _cmask(pd.Series(meta["timestamp"].values), max_lookback=1440)
    X_all     = assembled[fc].values
    row_ok    = gap_ok & ~np.isnan(X_all).any(axis=1)
    sp        = sequential(row_ok.sum())
    clean_idx = np.where(row_ok)[0]
    tr_idx    = clean_idx[sp.train]
    val_idx   = clean_idx[sp.val]
    te_idx    = clean_idx[sp.test]

    if ob_cache.exists():
        print(f"  Loading cached OB matrix ...")
        arrs = np.load(ob_cache)
        ob_tr, ob_v, ob_te = arrs["tr"], arrs["v"], arrs["te"]
    else:
        print(f"  Building OB matrix (top {OB_LEVELS} levels) ...")
        _, ob_df = load(ticker, include_ob=True)
        channels = []
        for inst in ["spot", "perp"]:
            for side in ["bids", "asks"]:
                cols = [f"{inst}_{side}_amount_{i}" for i in range(OB_LEVELS)]
                channels.append(ob_df[cols].values.astype(np.float32))
        ob_mat = np.stack(channels, axis=-1)
        del ob_df
        sc   = StandardScaler()
        flat = ob_mat[tr_idx].reshape(-1, OB_LEVELS * 4)
        sc.fit(flat)
        def _scale(idx):
            return sc.transform(
                ob_mat[idx].reshape(-1, OB_LEVELS * 4)
            ).reshape(-1, OB_LEVELS, 4).astype(np.float32)
        ob_tr = _scale(tr_idx)
        ob_v  = _scale(val_idx)
        ob_te = _scale(te_idx)
        del ob_mat
        np.savez_compressed(ob_cache, tr=ob_tr, v=ob_v, te=ob_te)
        print(f"  Cached → {ob_cache.name}")

    fmt = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    reg = _load_registry()

    def _make_ds(ob_chunk, y_chunk, shuffle=False, batch=256):
        valid_idx = np.array([i for i in range(SEQ_LEN, len(ob_chunk))
                              if not np.isnan(y_chunk[i])])
        if len(valid_idx) == 0:
            return None, np.empty(0)
        ob4 = ob_chunk.shape[1] * ob_chunk.shape[2]
        def gen():
            for i in valid_idx:
                x = ob_chunk[i - SEQ_LEN:i].reshape(SEQ_LEN, ob4, 1)
                yield x.astype(np.float32), np.float32(y_chunk[i])
        sig = (tf.TensorSpec(shape=(SEQ_LEN, ob4, 1), dtype=tf.float32),
               tf.TensorSpec(shape=(),               dtype=tf.float32))
        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
        if shuffle:
            ds = ds.shuffle(min(5000, len(valid_idx)))
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE), y_chunk[valid_idx]

    def _predict(model, ob_chunk, y_chunk, batch=512):
        valid_idx = np.array([i for i in range(SEQ_LEN, len(ob_chunk))
                              if not np.isnan(y_chunk[i])])
        ob4 = ob_chunk.shape[1] * ob_chunk.shape[2]
        preds = []
        for s in range(0, len(valid_idx), batch):
            idxs = valid_idx[s:s+batch]
            xs = np.stack([ob_chunk[i-SEQ_LEN:i].reshape(SEQ_LEN, ob4, 1)
                           for i in idxs]).astype(np.float32)
            preds.append(model.predict(xs, verbose=0).flatten())
        return np.concatenate(preds), y_chunk[valid_idx]

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col  = f"{direction}_{H}"
            code = _model_code(ticker, "dlob", direction, H)
            path = _model_path(code)

            print(f"\n{'─'*60}")
            print(f"  {code}")
            print(f"{'─'*60}")

            y_tr = all_lbl[col][tr_idx]
            y_v  = all_lbl[col][val_idx]
            y_te = all_lbl[col][te_idx]

            ds_tr, ys_tr = _make_ds(ob_tr, y_tr, shuffle=True)
            ds_v,  ys_v  = _make_ds(ob_v,  y_v)

            if ds_tr is None or len(ys_tr) < 200:
                print("  Too few samples, skipping.")
                continue

            print(f"  Train seqs: {len(ys_tr):,}  pos={ys_tr.mean():.1%}")
            print()

            if path.exists():
                print(f"  Loading cached model: {path.name}")
                model = tf.keras.models.load_model(str(path))
            else:
                model = build_deeplob(SEQ_LEN, OB_LEVELS)
                cw    = {0: 1.0, 1: float((1 - ys_tr.mean()) / (ys_tr.mean() + 1e-9))}
                cb    = [EpochProgress(20),
                         callbacks.EarlyStopping(monitor="val_auc", patience=5,
                                                 restore_best_weights=True, mode="max",
                                                 verbose=0)]
                model.fit(ds_tr, validation_data=ds_v,
                          epochs=20, callbacks=cb, verbose=0, class_weight=cw)
                model.save(str(path))
                print(f"  Saved → {path.name}")

            prob_v,  ys_v_true  = _predict(model, ob_v,  y_v)
            prob_te, ys_te_true = _predict(model, ob_te, y_te)

            auc_v  = _auc(ys_v_true,  prob_v)
            auc_te = _auc(ys_te_true, prob_te)
            print(f"\n  AUC — val={auc_v:.4f}  test={auc_te:.4f}  "
                  f"gap={abs(auc_v-auc_te):.3f}")

            t_opt = _best_f1_threshold(ys_v_true, prob_v)
            _print_cm(col, "val",  ys_v_true,  prob_v,  t_opt)
            _print_cm(col, "test", ys_te_true, prob_te, t_opt)

            reg[code] = {
                "ticker": ticker, "model_type": "dlob",
                "target": f"dir_{direction}", "horizon": H,
                "val_auc":  round(auc_v,  4),
                "test_auc": round(auc_te, 4),
                "seq_len":  SEQ_LEN, "ob_levels": OB_LEVELS,
                "trained_at": datetime.utcnow().isoformat(),
                "model_file": path.name,
            }
            _save_registry(reg)
            print(f"\n  Registered: {code}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    mode   = sys.argv[2] if len(sys.argv) > 2 else "both"

    if mode in ("cnn_lstm", "both"):
        run_cnn_lstm(ticker)
    if mode in ("deeplob", "both"):
        run_deeplob(ticker)
