"""
Direction prediction — DL models (CNN-LSTM and DeepLOB).

Two architectures:
  1. CNN-LSTM  : CausalConv1D → GRU → Dense. Input: (batch, SEQ_LEN, N_FEATURES).
                 Best general-purpose DL baseline for 1-min crypto.
  2. DeepLOB   : Conv blocks → Inception → LSTM. Input: (batch, SEQ_LEN, OB_LEVELS*4).
                 Specialist for raw orderbook sequences.

Labels and splits identical to models/direction.py for direct AUC comparison.

Run: python3 -m models.direction_dl [ticker] [model=cnn_lstm|deeplob|both]
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta, load
from data.gaps import clean_mask as _cmask
from models.splits import sequential, walk_forward
from features.assembly import assemble

CACHE_DIR = Path(__file__).parent.parent / "cache"

SEQ_LEN   = 60       # lookback window in bars
THRESHOLD = 0.008
HORIZONS  = [60, 100]

# most informative features for sequential model (avoid low-signal noise)
SEQ_FEATURES = [
    "ob_imb_perp_0_50", "ob_imb_perp_50_100", "ob_imb_perp_100_200",
    "ob_vel_perp_bids_0_50", "ob_vel_perp_asks_0_50",
    "span_perp",
    "ret_1", "ret_5", "ret_15", "ret_30", "ret_60",
    "rsi_6", "rsi_14", "macd_hist",
    "bb_pct_b", "bb_width",
    "vwap_dev_60", "vwap_dev_240",
    "taker_imb_1", "taker_imb_5", "taker_imb_15",
    "taker_net_5", "taker_net_15",
    "ofi_proxy",
    "oi_vel_5", "oi_vel_15",
    "fund_rate", "fund_mom_480",
    "perp_spread_bps", "perp_imbalance",
]

OB_LEVELS = 50   # top-N bins per side for DeepLOB (spot+perp × bids+asks = 4 channels)


# ── label computation (identical to direction.py) ─────────────────────────────

def _compute_labels(price_arr: np.ndarray) -> dict[str, np.ndarray]:
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

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Convert flat feature matrix and label vector to overlapping sequences.
    Returns (X_seq, y_seq) where X_seq shape = (n, seq_len, n_features).
    Drops rows where label is NaN or sequence extends before data start.
    """
    n = len(X)
    idx = [i for i in range(seq_len, n) if not np.isnan(y[i])]
    if not idx:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    X_seq = np.stack([X[i - seq_len:i] for i in idx])
    y_seq = y[np.array(idx)]
    return X_seq, y_seq


# ── CNN-LSTM architecture ─────────────────────────────────────────────────────

def build_cnn_lstm(seq_len: int, n_features: int) -> Model:
    inp = Input(shape=(seq_len, n_features))
    x = layers.Conv1D(64, kernel_size=5, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.GRU(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


# ── DeepLOB architecture ──────────────────────────────────────────────────────

def _inception_block(x, filters=32):
    b1 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    b2 = layers.Conv2D(filters, (3, 1), padding="same", activation="relu")(x)
    b3 = layers.MaxPooling2D((3, 1), strides=(1, 1), padding="same")(x)
    b3 = layers.Conv2D(filters, (1, 1), padding="same", activation="relu")(b3)
    return layers.Concatenate()([b1, b2, b3])


def build_deeplob(seq_len: int, ob_levels: int) -> Model:
    # input: (seq_len, ob_levels*4, 1) — 4 channels = spot/perp × bids/asks
    inp = Input(shape=(seq_len, ob_levels * 4, 1))
    x = layers.Conv2D(32, (1, 2), strides=(1, 2), activation="relu")(inp)
    x = layers.Conv2D(32, (4, 1), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (4, 1), padding="same", activation="relu")(x)
    x = _inception_block(x, 32)
    x = _inception_block(x, 32)
    x = layers.Reshape((x.shape[1], -1))(x)   # (seq, features)
    x = layers.LSTM(64)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model


# ── training ──────────────────────────────────────────────────────────────────

def _fit(model, X_tr, y_tr, X_val, y_val, epochs=20, batch=256):
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True, mode="max"
        )
    ]
    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch,
        callbacks=cb, verbose=0,
        class_weight={0: 1.0, 1: (1 - y_tr.mean()) / (y_tr.mean() + 1e-9)},
    )
    return model


def _auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


# ── CNN-LSTM run ──────────────────────────────────────────────────────────────

def run_cnn_lstm(ticker: str):
    print(f"\n{'='*60}")
    print(f"  CNN-LSTM — {ticker.upper()}")
    print(f"{'='*60}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)

    # select sequential features
    sel_idx = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    missing = [f for f in SEQ_FEATURES if f not in feat_cols]
    if missing:
        print(f"  Warning: {len(missing)} features not found: {missing[:5]}")
    X_tr  = X_train[:, sel_idx]
    X_v   = X_val[:,   sel_idx]
    X_te  = X_test[:,  sel_idx]
    n_feat = X_tr.shape[1]

    meta     = load_meta(ticker)
    ts_map   = dict(zip(meta["timestamp"].values, range(len(meta))))
    price    = meta["perp_ask_price"].values
    all_lbl  = _compute_labels(price)
    fmt      = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")

    rows = []
    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n── {col} ──────────────────────────────────────────")

            def _lbl(ts_arr):
                return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

            y_tr = _lbl(ts_train)
            y_v  = _lbl(ts_val)
            y_te = _lbl(ts_test)

            X_seq_tr, y_seq_tr = _build_sequences(X_tr, y_tr, SEQ_LEN)
            X_seq_v,  y_seq_v  = _build_sequences(X_v,  y_v,  SEQ_LEN)
            X_seq_te, y_seq_te = _build_sequences(X_te, y_te, SEQ_LEN)

            if len(y_seq_tr) < 200:
                print("  Too few samples, skipping.")
                continue

            pos = y_seq_tr.mean()
            print(f"  Train: {len(y_seq_tr):,} seqs  positives={pos:.1%}  "
                  f"{fmt(ts_train[SEQ_LEN])} → {fmt(ts_train[-1])}")

            model = build_cnn_lstm(SEQ_LEN, n_feat)
            _fit(model, X_seq_tr, y_seq_tr, X_seq_v, y_seq_v)

            auc_val  = _auc(y_seq_v,  model.predict(X_seq_v,  verbose=0).flatten())
            auc_test = _auc(y_seq_te, model.predict(X_seq_te, verbose=0).flatten())
            gap = abs(auc_val - auc_test)

            print(f"  VAL   AUC={auc_val:.4f}  ({fmt(ts_val[SEQ_LEN])} → {fmt(ts_val[-1])})")
            print(f"  TEST  AUC={auc_test:.4f}  ({fmt(ts_test[SEQ_LEN])} → {fmt(ts_test[-1])})")
            flag = "✓" if gap <= 0.03 else "⚠"
            print(f"  {flag}  val/test gap={gap:.3f}")

            rows.append({"model": "cnn_lstm", "label": col, "split": "val",  "auc": auc_val})
            rows.append({"model": "cnn_lstm", "label": col, "split": "test", "auc": auc_test})

    return pd.DataFrame(rows)


# ── DeepLOB run ───────────────────────────────────────────────────────────────

def run_deeplob(ticker: str):
    print(f"\n{'='*60}")
    print(f"  DeepLOB — {ticker.upper()}")
    print(f"{'='*60}\n")

    meta, ob = load(ticker, include_ob=True)
    ts_map   = dict(zip(meta["timestamp"].values, range(len(meta))))
    price    = meta["perp_ask_price"].values
    all_lbl  = _compute_labels(price)

    # build OB matrix: top OB_LEVELS bins × 4 channels (spot_bids, spot_asks, perp_bids, perp_asks)
    def _get_ob_channels():
        channels = []
        for inst in ["spot", "perp"]:
            for side in ["bids", "asks"]:
                cols = [f"{inst}_{side}_amount_{i}" for i in range(OB_LEVELS)]
                channels.append(ob[cols].values)
        return np.stack(channels, axis=-1)   # (n, OB_LEVELS, 4)

    print("  Building OB channel matrix ...")
    ob_mat = _get_ob_channels()   # (n, OB_LEVELS, 4)

    # scale each channel by its own rolling stats — fit on train only
    meta_ts = meta["timestamp"].values
    gap_ok  = _cmask(pd.Series(meta_ts), max_lookback=1440)
    assembled = pd.read_parquet(CACHE_DIR / f"{ticker}_features_assembled.parquet")
    fc   = [c for c in assembled.columns if c != "timestamp"]
    X_all = assembled[fc].values
    row_ok = gap_ok & ~np.isnan(X_all).any(axis=1)
    ts_clean = assembled["timestamp"].values[row_ok]

    sp = sequential(row_ok.sum())
    clean_idx = np.where(row_ok)[0]
    tr_idx  = clean_idx[sp.train]
    val_idx = clean_idx[sp.val]
    te_idx  = clean_idx[sp.test]

    # scale OB channels using train stats
    sc    = StandardScaler()
    flat  = ob_mat[tr_idx].reshape(-1, OB_LEVELS * 4)
    sc.fit(flat)
    def _scale(idx):
        m = ob_mat[idx].reshape(-1, OB_LEVELS * 4)
        return sc.transform(m).reshape(-1, OB_LEVELS, 4)
    ob_tr = _scale(tr_idx)
    ob_v  = _scale(val_idx)
    ob_te = _scale(te_idx)

    fmt  = lambda ts: datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    rows = []

    def _make_tf_dataset(ob_chunk: np.ndarray, y_chunk: np.ndarray,
                         seq_len: int, batch: int = 256, shuffle: bool = False):
        """Generator-based tf.data pipeline — never materialises all sequences in RAM."""
        valid_idx = np.array([i for i in range(seq_len, len(ob_chunk))
                              if not np.isnan(y_chunk[i])])
        if len(valid_idx) == 0:
            return None, np.empty(0)

        ob_levels_x4 = ob_chunk.shape[1] * ob_chunk.shape[2]   # OB_LEVELS * 4

        def gen():
            for i in valid_idx:
                x = ob_chunk[i - seq_len:i].reshape(seq_len, ob_levels_x4, 1)
                yield x.astype(np.float32), np.float32(y_chunk[i])

        sig = (
            tf.TensorSpec(shape=(seq_len, ob_levels_x4, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(),                          dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(5000, len(valid_idx)))
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE), y_chunk[valid_idx]

    def _predict_deeplob(model, ob_chunk, y_chunk, seq_len, batch=512):
        valid_idx = np.array([i for i in range(seq_len, len(ob_chunk))
                              if not np.isnan(y_chunk[i])])
        ob_levels_x4 = ob_chunk.shape[1] * ob_chunk.shape[2]
        preds = []
        for start in range(0, len(valid_idx), batch):
            idxs = valid_idx[start:start + batch]
            xs   = np.stack([ob_chunk[i - seq_len:i].reshape(seq_len, ob_levels_x4, 1)
                             for i in idxs]).astype(np.float32)
            preds.append(model.predict(xs, verbose=0).flatten())
        return np.concatenate(preds), y_chunk[valid_idx]

    for H in HORIZONS:
        for direction in ["up", "down"]:
            col = f"{direction}_{H}"
            print(f"\n── {col} ──────────────────────────────────────────")

            y_tr = all_lbl[col][tr_idx]
            y_v  = all_lbl[col][val_idx]
            y_te = all_lbl[col][te_idx]

            ds_tr, ys_tr = _make_tf_dataset(ob_tr, y_tr, SEQ_LEN, shuffle=True)
            ds_v,  ys_v  = _make_tf_dataset(ob_v,  y_v,  SEQ_LEN)

            if ds_tr is None or len(ys_tr) < 200:
                print("  Too few samples, skipping.")
                continue

            pos = ys_tr.mean()
            print(f"  Train: {len(ys_tr):,} seqs  positives={pos:.1%}")

            model = build_deeplob(SEQ_LEN, OB_LEVELS)
            cw    = {0: 1.0, 1: float((1 - pos) / (pos + 1e-9))}
            cb    = [tf.keras.callbacks.EarlyStopping(
                        monitor="val_auc", patience=5,
                        restore_best_weights=True, mode="max")]
            model.fit(ds_tr, validation_data=ds_v,
                      epochs=20, callbacks=cb, verbose=0,
                      class_weight=cw)

            prob_val,  y_val_true  = _predict_deeplob(model, ob_v,  y_v,  SEQ_LEN)
            prob_test, y_test_true = _predict_deeplob(model, ob_te, y_te, SEQ_LEN)
            auc_val  = _auc(y_val_true,  prob_val)
            auc_test = _auc(y_test_true, prob_test)
            gap = abs(auc_val - auc_test)

            print(f"  VAL   AUC={auc_val:.4f}")
            print(f"  TEST  AUC={auc_test:.4f}")
            flag = "✓" if gap <= 0.03 else "⚠"
            print(f"  {flag}  val/test gap={gap:.3f}")

            rows.append({"model": "deeplob", "label": col, "split": "val",  "auc": auc_val})
            rows.append({"model": "deeplob", "label": col, "split": "test", "auc": auc_test})

    return pd.DataFrame(rows)


# ── comparison summary ────────────────────────────────────────────────────────

def _compare(lgbm_path: Path, dl_rows: pd.DataFrame):
    lgbm = pd.read_parquet(lgbm_path)[["label", "split", "auc"]].copy()
    lgbm["model"] = "lightgbm"
    all_res = pd.concat([lgbm, dl_rows], ignore_index=True)
    pivot = all_res.pivot_table(index="label", columns=["model", "split"], values="auc").round(4)
    print(f"\n\n{'='*60}")
    print("  MODEL COMPARISON")
    print(f"{'='*60}")
    print(pivot.to_string())
    out = CACHE_DIR / "direction_dl_eval.parquet"
    dl_rows.to_parquet(out, index=False)
    print(f"\nDL results → {out.name}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    mode   = sys.argv[2] if len(sys.argv) > 2 else "both"

    rows = pd.DataFrame()
    if mode in ("cnn_lstm", "both"):
        rows = pd.concat([rows, run_cnn_lstm(ticker)], ignore_index=True)
    if mode in ("deeplob", "both"):
        rows = pd.concat([rows, run_deeplob(ticker)], ignore_index=True)

    lgbm_path = CACHE_DIR / f"{ticker}_direction_eval.parquet"
    if lgbm_path.exists() and not rows.empty:
        _compare(lgbm_path, rows)
