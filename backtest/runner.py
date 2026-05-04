"""
Strategy runner — pre-computes all ML signals and runs strategies 1–6.

Data: val period (Oct→Dec 2025) — ML models are out-of-sample here.
Results saved to cache/strategy_results.parquet and cache/strategy_equity.parquet.

Run: python3 -m backtest.runner
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import load_meta
from features.assembly import assemble
from models.splits import sequential
from models.direction_dl import (
    SEQ_FEATURES, SEQ_LEN, HORIZONS, _compute_labels, _build_sequences,
)
from models.volatility import _compute_targets
from backtest.engine import run as bt_run
from strategy.agent import STRATEGIES, DEFAULT_PARAMS

CACHE_DIR = Path(__file__).parent.parent / "cache"
RESULTS_FILE = CACHE_DIR / "strategy_results.parquet"
EQUITY_FILE  = CACHE_DIR / "strategy_equity.parquet"


# ── ML prediction pre-computation ────────────────────────────────────────────

def _get_ml_predictions(ticker, X_val, feat_cols, ts_val, meta):
    print("  Pre-computing ML predictions on val ...")
    ts_map = dict(zip(meta["timestamp"].values, range(len(meta))))
    price  = meta["perp_ask_price"].values

    # ATR predictions (vol model)
    vol_path = CACHE_DIR / f"{ticker}_lgbm_atr_30.txt"
    vol_model = lgb.Booster(model_file=str(vol_path))

    # need ATR rank — fit on train predictions
    X_train, _, _, _, ts_train, _, _ = assemble(ticker)
    atr_train = vol_model.predict(X_train)
    atr_val   = vol_model.predict(X_val)
    atr_rank  = np.clip(
        np.searchsorted(np.sort(atr_train), atr_val) / len(atr_train), 0, 1)

    # raw ATR predictions in dollars (for TP/SL sizing)
    all_vol = _compute_targets(price)
    atr_actual_scale = meta["perp_ask_price"].values.mean() * 0.001   # ~$90 typical

    # use model prediction × scale factor
    atr_pred_dollar = atr_val  # already in $ from vol model

    # direction model predictions
    X_val_aug = np.column_stack([X_val, atr_rank])
    sel_idx   = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_v_seq   = X_val[:, sel_idx]

    all_lbl = _compute_labels(price)

    def _lbl(ts_arr, col):
        return all_lbl[col][np.array([ts_map[t] for t in ts_arr])]

    # LightGBM direction predictions (with ATR)
    lgbm_preds = {}
    for H in HORIZONS:
        for direction in ["up", "down"]:
            col      = f"{direction}_{H}"
            lgb_path = CACHE_DIR / f"{ticker}_direction_lgbm_{col}.txt"
            if lgb_path.exists():
                lgbm_preds[col] = lgb.Booster(model_file=str(lgb_path)).predict(X_val_aug)
            else:
                lgbm_preds[col] = np.full(len(X_val), 0.5)

    # CNN-LSTM predictions (two-stage)
    cnn_preds = {}
    for H in HORIZONS:
        for direction in ["up", "down"]:
            col      = f"{direction}_{H}"
            cnn_path = CACHE_DIR / f"{ticker}_cnn2s_dir_{direction}_{H}.keras"
            if cnn_path.exists():
                cnn = tf.keras.models.load_model(str(cnn_path))
                y_v = _lbl(ts_val, col)
                # build sequences with ATR appended
                n = len(X_v_seq)
                valid_idx = np.array([i for i in range(SEQ_LEN, n) if not np.isnan(y_v[i])])
                atr_col   = atr_rank[valid_idx, np.newaxis, np.newaxis]
                atr_tiled = np.tile(atr_col, (1, SEQ_LEN, 1))
                Xs = np.stack([X_v_seq[i - SEQ_LEN:i] for i in valid_idx])
                Xs = np.concatenate([Xs, atr_tiled], axis=2)
                probs = cnn.predict(Xs, verbose=0).flatten()
                # map back to full val array (NaN for first SEQ_LEN bars)
                full = np.full(n, np.nan)
                full[valid_idx] = probs
                cnn_preds[col] = full
            else:
                cnn_preds[col] = np.full(len(X_val), 0.5)

    # ensemble (50/50 where both available)
    ens_preds = {}
    for col in lgbm_preds:
        lp = lgbm_preds[col]
        cp = cnn_preds[col]
        valid = ~np.isnan(cp)
        ep    = np.where(valid, 0.5 * lp + 0.5 * cp, lp)
        ens_preds[col] = ep

    return atr_rank, atr_pred_dollar, ens_preds


# ── build strategy DataFrame ──────────────────────────────────────────────────

def _build_strategy_df(ticker, X_val, feat_cols, ts_val, meta,
                        atr_rank, atr_dollar, ens_preds):
    print("  Building strategy DataFrame ...")

    def _col(name):
        if name in feat_cols:
            return X_val[:, feat_cols.index(name)]
        return np.zeros(len(X_val))

    price = meta["perp_ask_price"].values
    # map val timestamps to meta index
    ts_map = dict(zip(meta["timestamp"].values, range(len(meta))))
    val_idx = np.array([ts_map[t] for t in ts_val])
    prices_val = price[val_idx]

    df = pd.DataFrame({
        "timestamp":        ts_val,
        "price":            prices_val,
        "atr_pred":         atr_dollar,
        "vol_pred":         atr_rank,
        # direction ensemble
        "p_up_60":          ens_preds["up_60"],
        "p_dn_60":          ens_preds["down_60"],
        "p_up_100":         ens_preds["up_100"],
        "p_dn_100":         ens_preds["down_100"],
        # technical indicators (from feature matrix)
        "bb_pct_b":         _col("bb_pct_b"),
        "bb_width":         _col("bb_width"),
        "macd_hist":        _col("macd_hist"),
        "rsi_6":            _col("rsi_6"),
        "rsi_14":           _col("rsi_14"),
        "ofi_perp_10_r15":  _col("ofi_perp_10_r15"),
        "ofi_perp_10":      _col("ofi_perp_10"),
        "taker_imb_5":      _col("taker_imb_5"),
        "taker_net_15":     _col("taker_net_15"),
        "fund_rate":        _col("fund_rate"),
        "fund_mom_480":     _col("fund_mom_480"),
        "ret_sma_200":      _col("ret_sma_200"),
        "vwap_dev_1440":    _col("vwap_dev_1440"),
        "sma_50":           _col("sma_50"),
        "sma_200":          _col("sma_200"),
    })

    # forward-fill NaN predictions (from CNN sequence warmup)
    for col in ["p_up_60", "p_dn_60", "p_up_100", "p_dn_100"]:
        df[col] = df[col].fillna(method="ffill").fillna(0.5)

    return df.reset_index(drop=True)


# ── print progress ────────────────────────────────────────────────────────────

def _print_header():
    print(f"\n  {'Strategy':<26}  {'Return':>8}  {'Sharpe':>7}  "
          f"{'Calmar':>7}  {'MaxDD':>7}  {'Trades':>7}  {'WinRate':>8}  {'PF':>6}")
    print(f"  {'─'*85}")


def _print_row(name, desc, result):
    s = result.summary()
    ret_str = f"{s['total_return']:+.2f}%"
    print(f"  {name:<10} {desc:<16}  {ret_str:>8}  {s['sharpe']:>7.3f}  "
          f"{s['calmar']:>7.3f}  {s['max_drawdown']:>6.2f}%  "
          f"{s['n_trades']:>7,}  {s['win_rate']:>7.1f}%  {s['profit_factor']:>6.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    print(f"\n{'='*90}")
    print(f"  STRATEGY BACKTEST — {ticker.upper()} — Val period (Oct→Dec 2025)")
    print(f"{'='*90}\n")

    X_train, X_val, X_test, feat_cols, ts_train, ts_val, ts_test = assemble(ticker)
    meta = load_meta(ticker)

    # ── pre-compute signals ───────────────────────────────────────────────────
    atr_rank, atr_dollar, ens_preds = _get_ml_predictions(
        ticker, X_val, feat_cols, ts_val, meta)

    sdf = _build_strategy_df(
        ticker, X_val, feat_cols, ts_val, meta,
        atr_rank, atr_dollar, ens_preds)

    prices     = sdf["price"].values
    timestamps = sdf["timestamp"].values
    atr_d      = sdf["atr_pred"].values

    all_results = []
    all_equity  = {"timestamp": timestamps}

    print(f"  Val bars: {len(sdf):,}  "
          f"({datetime.utcfromtimestamp(ts_val[0]).strftime('%Y-%m-%d')} → "
          f"{datetime.utcfromtimestamp(ts_val[-1]).strftime('%Y-%m-%d')})")
    print(f"  Price range: ${prices.min():,.0f} – ${prices.max():,.0f}\n")

    _print_header()

    for name, (fn, desc) in STRATEGIES.items():
        params  = DEFAULT_PARAMS[name]
        signals, tp_pct, sl_pct = fn(sdf, params)
        result  = bt_run(signals, prices, tp_pct, sl_pct, timestamps)

        _print_row(name, desc, result)

        s = result.summary()
        all_results.append({
            "strategy_code": name,
            "strategy_name": desc,
            "ticker":        ticker,
            "split":         "val",
            "period_start":  datetime.utcfromtimestamp(ts_val[0]).strftime("%Y-%m-%d"),
            "period_end":    datetime.utcfromtimestamp(ts_val[-1]).strftime("%Y-%m-%d"),
            **s,
            "params": json.dumps(params),
        })
        all_equity[name] = result.equity[:len(timestamps)]

    # ── also run on test ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  TEST period (Dec 2025→Apr 2026) — params fixed from val")
    print(f"{'='*90}")

    _, X_test_, _, _, _, _, ts_test_ = assemble(ticker)

    atr_train    = lgb.Booster(model_file=str(CACHE_DIR / f"{ticker}_lgbm_atr_30.txt")).predict(X_train)
    atr_test_raw = lgb.Booster(model_file=str(CACHE_DIR / f"{ticker}_lgbm_atr_30.txt")).predict(X_test_)
    atr_rank_te  = np.clip(np.searchsorted(np.sort(atr_train), atr_test_raw) / len(atr_train), 0, 1)

    X_test_aug   = np.column_stack([X_test_, atr_rank_te])
    sel_idx      = [feat_cols.index(f) for f in SEQ_FEATURES if f in feat_cols]
    X_te_seq     = X_test_[:, sel_idx]
    ts_map       = dict(zip(meta["timestamp"].values, range(len(meta))))
    price_full   = meta["perp_ask_price"].values
    all_lbl      = _compute_labels(price_full)

    ens_preds_te = {}
    for H in HORIZONS:
        for direction in ["up", "down"]:
            col      = f"{direction}_{H}"
            lgb_path = CACHE_DIR / f"{ticker}_direction_lgbm_{col}.txt"
            cnn_path = CACHE_DIR / f"{ticker}_cnn2s_dir_{direction}_{H}.keras"
            lp = lgb.Booster(model_file=str(lgb_path)).predict(X_test_aug) if lgb_path.exists() \
                 else np.full(len(X_test_), 0.5)
            if cnn_path.exists():
                y_te    = all_lbl[col][np.array([ts_map[t] for t in ts_test_])]
                cnn     = tf.keras.models.load_model(str(cnn_path))
                n       = len(X_te_seq)
                vi      = np.array([i for i in range(SEQ_LEN, n) if not np.isnan(y_te[i])])
                ac      = atr_rank_te[vi, np.newaxis, np.newaxis]
                at      = np.tile(ac, (1, SEQ_LEN, 1))
                Xs      = np.concatenate([np.stack([X_te_seq[i-SEQ_LEN:i] for i in vi]), at], axis=2)
                pr      = cnn.predict(Xs, verbose=0).flatten()
                full    = np.full(n, np.nan); full[vi] = pr
                cp      = full
            else:
                cp = np.full(len(X_test_), 0.5)
            valid = ~np.isnan(cp)
            ens_preds_te[col] = np.where(valid, 0.5*lp + 0.5*cp, lp)

    sdf_te = _build_strategy_df(
        ticker, X_test_, feat_cols, ts_test_, meta,
        atr_rank_te, atr_test_raw, ens_preds_te)

    prices_te = sdf_te["price"].values
    ts_te     = sdf_te["timestamp"].values
    all_equity["timestamp_test"] = ts_te

    _print_header()

    for name, (fn, desc) in STRATEGIES.items():
        params  = DEFAULT_PARAMS[name]
        signals, tp_pct, sl_pct = fn(sdf_te, params)
        result  = bt_run(signals, prices_te, tp_pct, sl_pct, ts_te)

        _print_row(name, desc, result)

        s = result.summary()
        all_results.append({
            "strategy_code": name,
            "strategy_name": desc,
            "ticker":        ticker,
            "split":         "test",
            "period_start":  datetime.utcfromtimestamp(ts_test_[0]).strftime("%Y-%m-%d"),
            "period_end":    datetime.utcfromtimestamp(ts_test_[-1]).strftime("%Y-%m-%d"),
            **s,
            "params": json.dumps(params),
        })
        all_equity[f"{name}_test"] = result.equity[:len(ts_te)]

    # ── save ──────────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    df_results.to_parquet(RESULTS_FILE, index=False)
    pd.DataFrame(all_equity).to_parquet(EQUITY_FILE, index=False)

    print(f"\n  Results  → {RESULTS_FILE.name}")
    print(f"  Equity   → {EQUITY_FILE.name}")
    return df_results


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    run(ticker)
