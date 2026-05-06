"""
Backtest runner — wires vol + direction models → execution layer → engine.

No regime layer. All strategies fire freely (subject to their own signal logic).

Data flow:
  raw features (unscaled)  → strategy signal conditions
  scaled features (scaled) → ML predictions (vol LightGBM + CNN-LSTM direction)
  execution layer          → ATR-dynamic TP/SL, breakeven, time stop, sizing, entry confirm

Saved outputs:
  cache/{ticker}_backtest_results.parquet  — per-strategy metrics + metadata
  model_registry.json                      — updated with backtest run info

Run: python3 -m backtest.run [ticker]
"""

import sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader   import load_meta
from data.gaps     import clean_mask
from models.splits import sequential
from backtest.engine import run as _engine
from strategy.agent  import STRATEGIES, DEFAULT_PARAMS
from models.direction_dl import SEQ_FEATURES, SEQ_LEN, HORIZONS
from execution.config import EXECUTION_CONFIG
from execution.exit   import ATRDynamicExit, ComboExit
from execution.sizing import VolScaledSizer
from backtest.preds import _vol_preds, _dir_preds, _regime_preds  # cached versions

CACHE_DIR    = Path(__file__).parent.parent / "cache"
REGISTRY     = Path(__file__).parent.parent / "model_registry.json"
MAX_LOOKBACK = 1440
VOL_MODEL    = "atr_30"   # must exist as cache/{ticker}_lgbm_{VOL_MODEL}.txt
REGIME_FILE  = "regime_cusum"   # "regime_cusum", "regime_hmm", or None


# Per-regime-model strategy gates. Allowed states differ per model since
# state names are model-specific (CUSUM uses {trend_up/down, ranging, chop, calm};
# HMM uses {trend_bull/bear, ranging, high_vol_chop, fund_long/short}).
REGIME_GATES = {
    "regime_cusum": {
        "S1_VolDir":    {"trend_up", "trend_down", "chop"},
        "S2_Funding":   {"ranging", "calm"},
        "S3_BBRevert":  {"ranging", "calm"},
        "S4_MACDTrend": {"trend_up", "trend_down", "chop"},
        "S6_TwoSignal": {"trend_up", "trend_down", "chop"},
        "S7_OIDiverg":  {"ranging", "calm"},
        "S8_TakerFlow": {"trend_up", "trend_down", "chop"},
        "S10_Squeeze":  {"trend_up", "trend_down", "chop"},
        "S12_VWAPVol":  {"ranging", "calm"},
    },
    "regime_hmm": {
        "S1_VolDir":    {"trend_bull", "trend_bear", "high_vol_chop"},
        "S2_Funding":   {"ranging", "fund_long", "fund_short"},
        "S3_BBRevert":  {"ranging"},
        "S4_MACDTrend": {"trend_bull", "trend_bear", "high_vol_chop"},
        "S6_TwoSignal": {"trend_bull", "trend_bear", "high_vol_chop"},
        "S7_OIDiverg":  {"ranging", "fund_long", "fund_short"},
        "S8_TakerFlow": {"trend_bull", "trend_bear", "high_vol_chop"},
        "S10_Squeeze":  {"trend_bull", "trend_bear", "high_vol_chop"},
        "S12_VWAPVol":  {"ranging"},
    },
}
REGIME_GATE = REGIME_GATES.get(REGIME_FILE, {}) if REGIME_FILE else {}


def _fmt(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


# ── data loading ──────────────────────────────────────────────────────────────

# Meta-only columns not in assembled features — loaded separately
_META_EXTRA = [
    "spot_large_bid_count", "spot_large_ask_count",
    "perp_large_bid_count", "perp_large_ask_count",
    "diff_price",
]


def _load_splits(ticker: str) -> dict:
    """Load assembled features, apply gap mask, and split 50/25/25."""
    pq        = pd.read_parquet(CACHE_DIR / f"{ticker}_features_assembled.parquet")
    feat_cols = [c for c in pq.columns if c != "timestamp"]

    meta       = load_meta(ticker)
    ts_meta    = meta["timestamp"].values
    price_meta = meta["perp_ask_price"].values
    gap_ok     = clean_mask(pd.Series(ts_meta), max_lookback=MAX_LOOKBACK)

    X_raw  = pq[feat_cols].values
    ts_all = pq["timestamp"].values
    row_ok = gap_ok & ~np.isnan(X_raw).any(axis=1)

    X_clean  = X_raw[row_ok]
    ts_clean = ts_all[row_ok]

    n  = len(X_clean)
    sp = sequential(n, 0.50, 0.25)

    scaler  = StandardScaler()
    X_sc_tr = scaler.fit_transform(X_clean[sp.train])
    X_sc_v  = scaler.transform(X_clean[sp.val])
    X_sc_te = scaler.transform(X_clean[sp.test])

    ts_to_price = dict(zip(ts_meta, price_meta))

    meta_extra = {}
    for col in _META_EXTRA:
        if col in meta.columns:
            meta_extra[col] = dict(zip(ts_meta, meta[col].values))

    return dict(
        feat_cols  = feat_cols,
        meta_extra = meta_extra,
        X_raw_tr   = X_clean[sp.train],
        X_raw_v    = X_clean[sp.val],
        X_raw_te   = X_clean[sp.test],
        X_sc_tr    = X_sc_tr,
        X_sc_v     = X_sc_v,
        X_sc_te    = X_sc_te,
        ts_tr      = ts_clean[sp.train],
        ts_v       = ts_clean[sp.val],
        ts_te      = ts_clean[sp.test],
        price_tr   = np.array([ts_to_price[t] for t in ts_clean[sp.train]]),
        price_v    = np.array([ts_to_price[t] for t in ts_clean[sp.val]]),
        price_te   = np.array([ts_to_price[t] for t in ts_clean[sp.test]]),
    )


# ── strategy DataFrame builder ────────────────────────────────────────────────

_STRAT_COLS = [
    # S1–S6 original
    "bb_pct_b", "bb_width", "rsi_6", "rsi_14", "macd_hist",
    "ofi_perp_10_r15", "ofi_perp_10", "taker_imb_5", "taker_net_15",
    "fund_rate", "fund_mom_480", "ret_sma_200", "vwap_dev_1440",
    "sma_50", "sma_200",
    # S7–S13 additions
    "oi_price_div_15", "taker_net_30", "taker_net_60",
    "taker_imb_30", "ret_15", "vwap_dev_240",
    "vol_z_spot_60", "spot_imbalance", "perp_imbalance",
    # meta-only
    "spot_large_bid_count", "spot_large_ask_count",
    "perp_large_bid_count", "perp_large_ask_count",
    "diff_price",
]


def _make_df(X_raw, feat_cols, price, atr_pred, vol_pred, dir_split,
              ts_arr=None, meta_extra=None):
    """Assemble per-bar strategy inputs."""
    col_idx = {c: i for i, c in enumerate(feat_cols)}
    df = pd.DataFrame({"price": price, "atr_pred": atr_pred, "vol_pred": vol_pred})

    for c in _STRAT_COLS:
        if c in col_idx:
            df[c] = X_raw[:, col_idx[c]]
        elif meta_extra and c in meta_extra and ts_arr is not None:
            df[c] = [meta_extra[c].get(t, 0.0) for t in ts_arr]
        else:
            df[c] = 0.0

    df["p_up_60"]  = dir_split.get("up_60",   np.full(len(X_raw), 0.5))
    df["p_dn_60"]  = dir_split.get("down_60", np.full(len(X_raw), 0.5))
    df["p_up_100"] = dir_split.get("up_100",  np.full(len(X_raw), 0.5))
    df["p_dn_100"] = dir_split.get("down_100",np.full(len(X_raw), 0.5))

    return df


# ── summary printer ───────────────────────────────────────────────────────────

def _print_summary(df_out: pd.DataFrame, sp: dict, ticker: str):
    print(f"\n\n{'='*80}")
    print(f"  BACKTEST RESULTS — {ticker.upper()}")
    print(f"{'='*80}")
    print(f"  Train : {sp['ts_tr'][0]:s} → {sp['ts_tr'][-1]:s}  ({sp['n_tr']:,} bars)")
    print(f"  Val   : {sp['ts_v'][0]:s} → {sp['ts_v'][-1]:s}  ({sp['n_v']:,} bars)")
    print(f"  Test  : {sp['ts_te'][0]:s} → {sp['ts_te'][-1]:s}  ({sp['n_te']:,} bars)")
    print(f"  Features: {sp['n_feat']}  |  Vol: {ticker}_lgbm_{VOL_MODEL}  "
          f"|  Dir: {ticker}_cnn2s_dir")

    hdr = (f"\n  {'Strategy':<18}  {'Sharpe':>7}  {'Calmar':>7}  {'MaxDD%':>7}  "
           f"{'Trades':>7}  {'Win%':>6}  {'PFactor':>8}  {'Return%':>8}")
    sep = "  " + "─" * 76

    gate_pass = []
    for split in ["val", "test"]:
        print(f"\n  ── {split.upper()} ──")
        print(hdr)
        print(sep)
        sub = df_out[df_out["split"] == split].sort_values("sharpe", ascending=False)
        for _, r in sub.iterrows():
            flag = "✓" if r["sharpe"] > 0.5 else " "
            print(f"  {r['strategy_key']:<18}  "
                  f"{r['sharpe']:>7.3f}  "
                  f"{r['calmar']:>7.3f}  "
                  f"{r['max_drawdown']:>7.1f}  "
                  f"{int(r['n_trades']):>7}  "
                  f"{r['win_rate']:>6.1f}  "
                  f"{r['profit_factor']:>8.3f}  "
                  f"{r['total_return']:>8.2f}  {flag}")
            if split == "val" and r["sharpe"] > 0.5:
                gate_pass.append(r["strategy_key"])

    # ── gate + conclusions ────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  PHASE 1 GATE — Sharpe > 0.5 on val")
    print(f"{'='*80}")
    if gate_pass:
        print(f"  PASS ✓ — {len(gate_pass)} strategy/ies: {', '.join(gate_pass)}")
    else:
        print(f"  FAIL ✗ — no strategy clears Sharpe > 0.5 on val")

    val_df  = df_out[df_out["split"] == "val"]
    test_df = df_out[df_out["split"] == "test"]
    best_v  = val_df.loc[val_df["sharpe"].idxmax()]
    best_te = test_df.loc[test_df["sharpe"].idxmax()]
    low_tr  = test_df[test_df["n_trades"] < 50]

    print(f"\n  Conclusions:")
    print(f"  - Best val  : {best_v['strategy_key']}  "
          f"Sharpe={best_v['sharpe']:.3f}  Trades={int(best_v['n_trades'])}  "
          f"WinRate={best_v['win_rate']:.1f}%")
    print(f"  - Best test : {best_te['strategy_key']}  "
          f"Sharpe={best_te['sharpe']:.3f}  Trades={int(best_te['n_trades'])}  "
          f"WinRate={best_te['win_rate']:.1f}%")
    if not low_tr.empty:
        print(f"  - Low-trade strategies on test (<50 trades): "
              f"{', '.join(low_tr['strategy_key'].tolist())}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc", regime_file: str = None):
    """Run the strategy backtest. `regime_file` overrides REGIME_FILE module default."""
    rf = regime_file if regime_file is not None else REGIME_FILE
    rg = REGIME_GATES.get(rf, {}) if rf else {}

    print(f"\n{'='*70}")
    print(f"  BACKTEST RUNNER — {ticker.upper()}  (regime: {rf or 'none'})")
    print(f"{'='*70}\n")

    # ── 1. Load splits ────────────────────────────────────────────────────────
    print("  Loading features ...")
    data = _load_splits(ticker)
    fc   = data["feat_cols"]

    ts_info = {
        "ts_tr":  (_fmt(data["ts_tr"][0]),  _fmt(data["ts_tr"][-1])),
        "ts_v":   (_fmt(data["ts_v"][0]),   _fmt(data["ts_v"][-1])),
        "ts_te":  (_fmt(data["ts_te"][0]),  _fmt(data["ts_te"][-1])),
        "n_tr":   len(data["ts_tr"]),
        "n_v":    len(data["ts_v"]),
        "n_te":   len(data["ts_te"]),
        "n_feat": len(fc),
    }
    print(f"  Train : {ts_info['ts_tr'][0]} → {ts_info['ts_tr'][1]}  ({ts_info['n_tr']:,})")
    print(f"  Val   : {ts_info['ts_v'][0]}  → {ts_info['ts_v'][1]}  ({ts_info['n_v']:,})")
    print(f"  Test  : {ts_info['ts_te'][0]}  → {ts_info['ts_te'][1]}  ({ts_info['n_te']:,})")

    # ── 2. Vol predictions ────────────────────────────────────────────────────
    print(f"\n  Loading vol model {ticker}_lgbm_{VOL_MODEL} ...")
    atr_tr, atr_v, atr_te, rank_tr, rank_v, rank_te = _vol_preds(
        ticker, data["X_sc_tr"], data["X_sc_v"], data["X_sc_te"])
    print(f"  Vol rank  val p50={np.median(rank_v):.2f}  "
          f"val p80={np.percentile(rank_v, 80):.2f}  "
          f"test p50={np.median(rank_te):.2f}")

    # ── 3. Direction predictions ──────────────────────────────────────────────
    print(f"\n  Loading direction models (CNN-LSTM two-stage) ...")
    dir_all = _dir_preds(
        ticker,
        data["X_sc_tr"], data["X_sc_v"], data["X_sc_te"],
        fc, rank_tr, rank_v, rank_te,
    )
    for col in ["up_60", "down_60", "up_100", "down_100"]:
        if col in dir_all:
            _, v_arr, te_arr = dir_all[col]
            print(f"  {col:<10}  val  p50={np.median(v_arr):.3f}  "
                  f"p80={np.percentile(v_arr, 80):.3f}  |  "
                  f"test p50={np.median(te_arr):.3f}  "
                  f"p80={np.percentile(te_arr, 80):.3f}")

    # ── 4. Build strategy DataFrames ──────────────────────────────────────────
    dir_v  = {col: arrs[1] for col, arrs in dir_all.items()}
    dir_te = {col: arrs[2] for col, arrs in dir_all.items()}

    me = data.get("meta_extra", {})
    df_val  = _make_df(data["X_raw_v"],  fc, data["price_v"],
                       atr_v,  rank_v,  dir_v,  data["ts_v"],  me)
    df_test = _make_df(data["X_raw_te"], fc, data["price_te"],
                       atr_te, rank_te, dir_te, data["ts_te"], me)

    # ── 4b. Regime gate (optional, cached) ────────────────────────────────────
    regime_v, regime_te = (None, None)
    if rf:
        regime_v, regime_te = _regime_preds(
            ticker, data["ts_v"], data["ts_te"], regime_file=rf)
        if regime_v is not None:
            print(f"\n  Regime gate active ({rf})")
            for sname, regs in [("val", regime_v), ("test", regime_te)]:
                uniq, cnts = np.unique(regs, return_counts=True)
                print(f"    {sname}: " + "  ".join(f"{u}={c/len(regs):.1%}" for u, c in zip(uniq, cnts)))
        else:
            print(f"\n  Regime gate disabled (no {rf} parquet)")

    # ── 5. Run strategies ─────────────────────────────────────────────────────
    print(f"\n  Running {len(STRATEGIES)} strategies × 2 splits ...\n")
    rows    = []
    run_at  = datetime.utcnow().isoformat()
    meta_kv = {
        "ticker":      ticker,
        "train_start": ts_info["ts_tr"][0], "train_end": ts_info["ts_tr"][1],
        "val_start":   ts_info["ts_v"][0],  "val_end":   ts_info["ts_v"][1],
        "test_start":  ts_info["ts_te"][0], "test_end":  ts_info["ts_te"][1],
        "train_rows":  ts_info["n_tr"],
        "val_rows":    ts_info["n_v"],
        "test_rows":   ts_info["n_te"],
        "n_features":  ts_info["n_feat"],
        "vol_model":   f"{ticker}_lgbm_{VOL_MODEL}",
        "dir_model":   f"{ticker}_cnn2s_dir",
        "run_at":      run_at,
    }

    atr_median = float(np.median(atr_tr))   # training median for relative scaling

    for key, (fn, name) in STRATEGIES.items():
        params   = DEFAULT_PARAMS[key]
        exec_cfg = EXECUTION_CONFIG.get(key)
        allowed  = rg.get(key)            # set of allowed regime states (or None)

        for split_label, df_s, ts_arr, atr_arr, regimes in [
            ("val",  df_val,  data["ts_v"],  atr_v,  regime_v),
            ("test", df_test, data["ts_te"], atr_te, regime_te),
        ]:
            price_arr = df_s["price"].values
            n         = len(price_arr)

            # ── entry strategy: preprocess signals ────────────────────────────
            raw_sigs, _, _ = fn(df_s, params)
            if exec_cfg:
                sigs = exec_cfg.entry.apply(raw_sigs)
            else:
                sigs = raw_sigs

            # ── regime gate: zero signals outside allowed states ──────────────
            n_pre_gate = int((sigs != 0).sum())
            if regimes is not None and allowed:
                gate_mask = np.isin(regimes, list(allowed))
                sigs      = sigs * gate_mask
            n_gated_out = n_pre_gate - int((sigs != 0).sum())

            # ── exit strategy: compute per-bar TP/SL arrays ───────────────────
            if exec_cfg and hasattr(exec_cfg.exit, "arrays"):
                tp_arr_e, sl_arr_e = exec_cfg.exit.arrays(atr_arr, price_arr, atr_median)
                plan0   = exec_cfg.exit.plan(atr_median, float(np.median(price_arr)),
                                              atr_median)
                trail_arr = np.zeros(n, dtype=np.float32)
                tab_arr   = np.full(n, plan0.tab_pct,        dtype=np.float32)
                be_arr    = np.full(n, plan0.breakeven_pct,  dtype=np.float32)
                ts_arr_e  = np.full(n, plan0.time_stop_bars, dtype=np.int32)
            else:
                tp_p  = params.get("tp_pct", 0.020)
                sl_p  = params.get("sl_pct", 0.007)
                tp_arr_e  = np.full(n, tp_p,  dtype=np.float32)
                sl_arr_e  = np.full(n, sl_p,  dtype=np.float32)
                trail_arr = np.full(n, params.get("trail_pct", 0.0), dtype=np.float32)
                tab_arr   = np.zeros(n, dtype=np.float32)
                be_arr    = np.zeros(n, dtype=np.float32)
                ts_arr_e  = np.zeros(n, dtype=np.int32)

            # ── sizing: per-bar position size ─────────────────────────────────
            if exec_cfg:
                if isinstance(exec_cfg.sizing, VolScaledSizer):
                    sz = exec_cfg.sizing
                    size_arr = np.clip(sz.target_risk / np.maximum(sl_arr_e, 1e-6),
                                        sz.min_size, sz.max_size).astype(np.float32)
                else:
                    size_arr = np.full(n, exec_cfg.sizing.fraction, dtype=np.float32)
            else:
                size_arr = np.full(n, 0.10, dtype=np.float32)

            # ── run engine ────────────────────────────────────────────────────
            result = _engine(
                sigs, price_arr, tp_arr_e, sl_arr_e, ts_arr,
                trail_pct_arr     = trail_arr,
                tab_pct_arr       = tab_arr,
                breakeven_pct_arr = be_arr,
                time_stop_arr     = ts_arr_e,
                position_size_arr = size_arr,
                force_exit_arr    = None,
            )
            summ  = result.summary()
            n_sig = int((sigs != 0).sum())
            avg_t = (sum(t.pnl_pct for t in result.trades) / len(result.trades) * 100
                     if result.trades else 0.0)

            reasons = {}
            for t in result.trades:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

            rows.append({
                **meta_kv,
                "strategy_key":  key,
                "strategy_name": name,
                "split":         split_label,
                "n_signals":     n_sig,
                "n_pre_gate":    n_pre_gate,
                "n_gated_out":   n_gated_out,
                "avg_trade_pct": round(avg_t, 4),
                "exit_TP":   reasons.get("TP",   0),
                "exit_SL":   reasons.get("SL",   0),
                "exit_TSL":  reasons.get("TSL",  0),
                "exit_BE":   reasons.get("BE",   0),
                "exit_TIME": reasons.get("TIME", 0),
                "exit_EOD":  reasons.get("EOD",  0),
                **summ,
            })

            gate_info = f" [gate {n_gated_out}/{n_pre_gate}]" if n_pre_gate else ""
            print(f"  {key:<18}  {split_label:<5}  "
                  f"Sharpe={summ['sharpe']:>6.3f}  Tr={summ['n_trades']:>4}  "
                  f"Win={summ['win_rate']:>4.0f}%  "
                  f"TP={reasons.get('TP',0):>3} SL={reasons.get('SL',0):>3} "
                  f"BE={reasons.get('BE',0):>3} T={reasons.get('TIME',0):>3}{gate_info}")

    # ── 6. Save results ───────────────────────────────────────────────────────
    df_out   = pd.DataFrame(rows)
    out_path = CACHE_DIR / f"{ticker}_backtest_results.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\n  Results → {out_path.name}")

    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}
    best_val_sharpe = df_out[df_out["split"] == "val"]["sharpe"].max()
    best_val_strat  = df_out.loc[(df_out["split"] == "val") &
                                  (df_out["sharpe"] == best_val_sharpe),
                                  "strategy_key"].iloc[0]
    reg[f"{ticker}_backtest"] = {
        **{k: meta_kv[k] for k in [
            "run_at", "vol_model", "dir_model",
            "train_start", "train_end", "val_start", "val_end",
            "test_start",  "test_end",
            "train_rows",  "val_rows", "test_rows", "n_features",
        ]},
        "best_val_strategy": best_val_strat,
        "best_val_sharpe":   round(float(best_val_sharpe), 4),
        "phase1_gate_pass":  bool(best_val_sharpe > 0.5),
    }
    REGISTRY.write_text(json.dumps(reg, indent=2))

    _print_summary(df_out, ts_info, ticker)


if __name__ == "__main__":
    ticker      = sys.argv[1] if len(sys.argv) > 1 else "btc"
    regime_file = sys.argv[2] if len(sys.argv) > 2 else None
    run(ticker, regime_file=regime_file)
