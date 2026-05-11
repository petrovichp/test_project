"""
Per-strategy hyperparameter grid search.

Searches signal thresholds and exit (TP / SL) per strategy, picks val-best by
Sharpe, then locks and evaluates on DQN-test.

Sequential-trade simulator (numba) walks bars, opens a trade on each non-zero
signal (1-bar lag), holds until TP/SL/BE/trail/time-stop exit, then resumes.
Mirrors backtest/engine.run semantics for single-strategy mode.

Run: python3 -m models.grid_search [ticker]
"""

import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader            import load_meta
from strategy.agent         import STRATEGIES
from backtest.single_trade  import simulate_one_trade

CACHE       = ROOT / "cache"
WARMUP      = 1440
BARS_PER_YR = 525_960

# bar-index splits (matching dqn_state.py)
DQN_TRAIN_E = 281_440 - WARMUP    # 280,000
DQN_VAL_E   = 332_307 - WARMUP    # 330,867


# ── per-strategy parameter grids ─────────────────────────────────────────────

GRIDS = {
    "S1_VolDir": [
        dict(vol_thresh=v, dir_thresh=d, tp_pct=t, sl_pct=s,
             base_tp_pct=t, base_sl_pct=s,
             breakeven_pct=0.005, time_stop_bars=0,
             trail_after_breakeven=True)
        for v in [0.55, 0.65]
        for d in [0.70, 0.75, 0.80]
        for t in [0.015, 0.020, 0.025]
        for s in [0.005, 0.007, 0.010]
    ],
    "S4_MACDTrend": [
        dict(vol_thresh=v, dir_thresh=d, sma_dev=0.002, tp_pct=t, sl_pct=s,
             base_tp_pct=t, base_sl_pct=s,
             breakeven_pct=0.006, time_stop_bars=0,
             trail_after_breakeven=True)
        for v in [0.55, 0.65]
        for d in [0.65, 0.70, 0.75]
        for t in [0.020, 0.025, 0.030]
        for s in [0.006, 0.008, 0.010]
    ],
    "S6_TwoSignal": [
        dict(vol_thresh=v, dir_req=d, dir_opp=0.20, tp_pct=t, sl_pct=s,
             base_tp_pct=t, base_sl_pct=s,
             breakeven_pct=0.005, time_stop_bars=0,
             trail_after_breakeven=True)
        for v in [0.50, 0.60]
        for d in [0.65, 0.70, 0.75]
        for t in [0.020, 0.025, 0.030]
        for s in [0.006, 0.008, 0.010]
    ],
    "S7_OIDiverg": [
        dict(div_sigma=ds, vol_floor=vf, tp_pct=t, sl_pct=s,
             base_tp_pct=t, base_sl_pct=s,
             breakeven_pct=0.003, time_stop_bars=45,
             trail_after_breakeven=False)
        for ds in [1.0, 1.5, 2.0, 2.5]
        for vf in [0.40, 0.50]
        for t in [0.015, 0.020, 0.025]
        for s in [0.005, 0.007, 0.010]
    ],
    "S8_TakerFlow": [
        dict(taker_sigma=ts, vol_floor=vf, tp_pct=t, sl_pct=s,
             base_tp_pct=t, base_sl_pct=s,
             breakeven_pct=0.004, time_stop_bars=0,
             trail_after_breakeven=True)
        for ts in [0.7, 1.0, 1.5, 2.0]
        for vf in [0.40, 0.50]
        for t in [0.012, 0.015, 0.020]
        for s in [0.005, 0.006, 0.008]
    ],
}


# ── exit-array builder (param-driven, no ExecutionConfig dependency) ────────

def _exit_arrays(atr_arr: np.ndarray, base_tp: float, base_sl: float,
                  atr_median: float, breakeven_pct: float,
                  time_stop_bars: int, trail_after_breakeven: bool):
    scale = np.clip(atr_arr / atr_median, 0.2, 5.0)
    tp_arr = np.clip(base_tp * scale, 0.005, 0.060).astype(np.float32)
    sl_arr = np.clip(base_sl * scale, 0.002, 0.025).astype(np.float32)
    n = len(atr_arr)
    trail_arr = np.zeros(n, dtype=np.float32)
    tab_val   = base_sl if trail_after_breakeven else 0.0
    tab_arr   = np.full(n, tab_val,        dtype=np.float32)
    be_arr    = np.full(n, breakeven_pct,  dtype=np.float32)
    ts_arr    = np.full(n, time_stop_bars, dtype=np.int32)
    return tp_arr, sl_arr, trail_arr, tab_arr, be_arr, ts_arr


# ── sequential-trade simulator ───────────────────────────────────────────────

@njit(cache=True)
def _simulate_sequential(signals, prices, tp, sl, tr, tab, be, ts_bars):
    n = len(signals)
    pnls = np.zeros(5000, dtype=np.float64)
    durs = np.zeros(5000, dtype=np.int32)
    cnt  = 0
    t = 0
    while t < n - 1:
        s = signals[t]
        if s != 0:
            pnl, n_held, _ = simulate_one_trade(
                prices, t + 1, int(s),
                float(tp[t]), float(sl[t]),
                float(tr[t]), float(tab[t]),
                float(be[t]), int(ts_bars[t]),
                0,
            )
            if cnt < pnls.shape[0]:
                pnls[cnt] = pnl
                durs[cnt] = n_held + 1
                cnt += 1
            t = t + 1 + n_held + 1
        else:
            t += 1
    return pnls[:cnt], durs[:cnt]


# ── strategy-DataFrame builder (reuses dqn_state shape) ─────────────────────

_STRAT_COLS = [
    "bb_pct_b", "bb_width", "rsi_6", "rsi_14", "macd_hist",
    "ofi_perp_10_r15", "ofi_perp_10", "taker_imb_5", "taker_net_15",
    "fund_rate", "fund_mom_480", "ret_sma_200", "vwap_dev_1440",
    "sma_50", "sma_200",
    "oi_price_div_15", "taker_net_30", "taker_net_60",
    "taker_imb_30", "ret_15", "vwap_dev_240",
    "vol_z_spot_60", "spot_imbalance", "perp_imbalance",
    "spot_large_bid_count", "spot_large_ask_count",
    "perp_large_bid_count", "perp_large_ask_count",
    "diff_price",
]


def _build_strategy_df(pq_use: pd.DataFrame, meta_use: pd.DataFrame,
                        price: np.ndarray, atr: np.ndarray, rank: np.ndarray,
                        dir_preds: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "price":    price,
        "atr_pred": atr,
        "vol_pred": rank,
    })
    for c in _STRAT_COLS:
        if c in pq_use.columns:
            df[c] = pq_use[c].values
        elif c in meta_use.columns:
            df[c] = meta_use[c].values
        else:
            df[c] = 0.0
    df["p_up_60"]   = dir_preds["up_60"]
    df["p_dn_60"]   = dir_preds["down_60"]
    df["p_up_100"]  = dir_preds["up_100"]
    df["p_dn_100"]  = dir_preds["down_100"]
    return df


# ── Sharpe ───────────────────────────────────────────────────────────────────

def _sharpe(pnls: np.ndarray, n_total_bars: int) -> float:
    """Engine-style Sharpe: per-bar returns (mostly zero) annualized for 1-min."""
    if len(pnls) == 0:
        return 0.0
    rets = np.zeros(n_total_bars, dtype=np.float64)
    n_t  = min(len(pnls), n_total_bars)
    rets[:n_t] = pnls[:n_t]
    if rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YR))


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*72}\n  HYPERPARAM GRID SEARCH  ({ticker.upper()})\n{'='*72}")

    # ── load full feature parquet + meta + vol + dir ─────────────────────────
    pq    = pd.read_parquet(CACHE / "features" / f"{ticker}_features_assembled.parquet")
    meta  = load_meta(ticker)
    assert (pq["timestamp"].values == meta["timestamp"].values).all()

    vol      = np.load(CACHE / "preds" / f"{ticker}_pred_vol_v4.npz")
    atr_full = pd.Series(vol["atr"]).ffill().bfill().values.astype(np.float32)
    rk_full  = pd.Series(vol["rank"]).ffill().bfill().values.astype(np.float32)
    atr_med  = float(vol["atr_train_median"])

    dir_preds = {}
    for col in ["up_60", "down_60", "up_100", "down_100"]:
        dir_preds[col] = np.load(CACHE / "preds" / f"{ticker}_pred_dir_{col}_v4.npz")["preds"]

    pq_use   = pq.iloc[WARMUP:].reset_index(drop=True)
    meta_use = meta.iloc[WARMUP:].reset_index(drop=True)
    price    = meta_use["perp_ask_price"].values.astype(np.float64)

    # build full-period strategy DataFrame (covers DQN-train + DQN-val + DQN-test)
    df_full = _build_strategy_df(pq_use, meta_use, price, atr_full, rk_full,
                                  dir_preds)
    print(f"  features built: df_full shape {df_full.shape}")

    # split slices into pq_use index space
    sl_train = (100_000, DQN_TRAIN_E)              # 100,000 → 280,000
    sl_val   = (DQN_TRAIN_E, DQN_VAL_E)            # 280,000 → 330,867
    sl_test  = (DQN_VAL_E, len(df_full))           # 330,867 → 383,174
    n_val    = sl_val[1]  - sl_val[0]
    n_test   = sl_test[1] - sl_test[0]
    print(f"  splits: train={sl_train[1]-sl_train[0]:,}  val={n_val:,}  test={n_test:,}")

    # ── jit warmup ───────────────────────────────────────────────────────────
    print("  JIT-compiling sequential simulator ...")
    t1 = time.perf_counter()
    _ = _simulate_sequential(
        np.zeros(20, dtype=np.int8), price[:20],
        np.full(20, 0.02, dtype=np.float32), np.full(20, 0.005, dtype=np.float32),
        np.zeros(20, dtype=np.float32),       np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32),       np.zeros(20, dtype=np.int32),
    )
    print(f"    compiled in {time.perf_counter()-t1:.2f}s")

    # ── per-strategy grid eval ──────────────────────────────────────────────
    all_results = []
    best_per_strategy = {}

    for key, grid in GRIDS.items():
        if key not in STRATEGIES:
            print(f"  skip {key}: not in active strategies")
            continue
        fn, _ = STRATEGIES[key]
        print(f"\n  ── {key}  ({len(grid)} combos) ─────────────────────────────────")

        # Per-combo: recompute signals (depends on threshold params)
        # Pre-slice the full DataFrame view per split to avoid repeated copies.
        df_train = df_full.iloc[sl_train[0]:sl_train[1]].reset_index(drop=True)
        df_val   = df_full.iloc[sl_val  [0]:sl_val  [1]].reset_index(drop=True)
        df_test  = df_full.iloc[sl_test [0]:sl_test [1]].reset_index(drop=True)
        atr_val  = atr_full[sl_val [0]:sl_val [1]]
        atr_test = atr_full[sl_test[0]:sl_test[1]]
        price_val  = price [sl_val [0]:sl_val [1]]
        price_test = price [sl_test[0]:sl_test[1]]

        rows = []
        t_strat = time.perf_counter()
        for params in grid:
            # signals on val
            sigs_v, _, _ = fn(df_val, params)
            sigs_v       = np.asarray(sigs_v, dtype=np.int8)
            # exit arrays on val
            tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _exit_arrays(
                atr_val,
                params["base_tp_pct"], params["base_sl_pct"], atr_med,
                params["breakeven_pct"], params["time_stop_bars"],
                params["trail_after_breakeven"],
            )
            pnls_v, durs_v = _simulate_sequential(
                sigs_v, price_val, tp_v, sl_v, tr_v, tab_v, be_v, ts_v)
            sharpe_v = _sharpe(pnls_v, n_val)
            n_v      = len(pnls_v)
            wr_v     = float((pnls_v > 0).mean()) if n_v else 0.0
            tot_v    = float(pnls_v.sum()) if n_v else 0.0

            rows.append(dict(params=params, n_trades=n_v,
                              sharpe=sharpe_v, win_rate=wr_v, total=tot_v))

        # filter for min trade count, then pick best by val Sharpe
        valid = [r for r in rows if r["n_trades"] >= 30]
        if not valid:
            valid = [r for r in rows if r["n_trades"] >= 5]
        if not valid:
            print(f"    no combos with enough trades; skip")
            continue
        best = max(valid, key=lambda r: r["sharpe"])

        # locked test eval at best val params
        bp = best["params"]
        sigs_te, _, _ = fn(df_test, bp)
        sigs_te = np.asarray(sigs_te, dtype=np.int8)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _exit_arrays(
            atr_test, bp["base_tp_pct"], bp["base_sl_pct"], atr_med,
            bp["breakeven_pct"], bp["time_stop_bars"],
            bp["trail_after_breakeven"],
        )
        pnls_te, durs_te = _simulate_sequential(
            sigs_te, price_test, tp_t, sl_t, tr_t, tab_t, be_t, ts_t)
        sharpe_te = _sharpe(pnls_te, n_test)
        n_te      = len(pnls_te)
        wr_te     = float((pnls_te > 0).mean()) if n_te else 0.0
        tot_te    = float(pnls_te.sum()) if n_te else 0.0

        # default-params reference (using STRATEGIES default params)
        from strategy.agent import DEFAULT_PARAMS
        from execution.config import EXECUTION_CONFIG
        def_params = dict(DEFAULT_PARAMS[key])
        # build default exit params from EXECUTION_CONFIG
        cfg = EXECUTION_CONFIG[key]
        def_params["base_tp_pct"]            = cfg.exit.base_tp
        def_params["base_sl_pct"]            = cfg.exit.base_sl
        def_params["breakeven_pct"]          = cfg.exit._be
        def_params["time_stop_bars"]         = cfg.exit._ts
        def_params["trail_after_breakeven"]  = cfg.exit._tab_en
        sigs_d_v, _, _ = fn(df_val,  def_params)
        sigs_d_t, _, _ = fn(df_test, def_params)
        sigs_d_v = np.asarray(sigs_d_v, dtype=np.int8)
        sigs_d_t = np.asarray(sigs_d_t, dtype=np.int8)
        tp_dv, sl_dv, tr_dv, tab_dv, be_dv, ts_dv = _exit_arrays(
            atr_val, def_params["base_tp_pct"], def_params["base_sl_pct"], atr_med,
            def_params["breakeven_pct"], def_params["time_stop_bars"],
            def_params["trail_after_breakeven"])
        tp_dt, sl_dt, tr_dt, tab_dt, be_dt, ts_dt = _exit_arrays(
            atr_test, def_params["base_tp_pct"], def_params["base_sl_pct"], atr_med,
            def_params["breakeven_pct"], def_params["time_stop_bars"],
            def_params["trail_after_breakeven"])
        pnls_dv, _ = _simulate_sequential(sigs_d_v, price_val,  tp_dv, sl_dv, tr_dv, tab_dv, be_dv, ts_dv)
        pnls_dt, _ = _simulate_sequential(sigs_d_t, price_test, tp_dt, sl_dt, tr_dt, tab_dt, be_dt, ts_dt)
        sharpe_def_v = _sharpe(pnls_dv, n_val)
        sharpe_def_t = _sharpe(pnls_dt, n_test)

        elapsed = time.perf_counter() - t_strat
        improvement_v = best["sharpe"] - sharpe_def_v
        improvement_t = sharpe_te      - sharpe_def_t

        print(f"    default params           val Sharpe = {sharpe_def_v:>+7.3f}  "
              f"test Sharpe = {sharpe_def_t:>+7.3f}  "
              f"({len(pnls_dv)}/{len(pnls_dt)} trades)")
        print(f"    BEST  combo (n={best['n_trades']:>3} val trades)  "
              f"val Sharpe = {best['sharpe']:>+7.3f}  test Sharpe = {sharpe_te:>+7.3f}  "
              f"({n_te} test trades)  Δ_v={improvement_v:+.2f}  Δ_t={improvement_t:+.2f}  "
              f"[{elapsed:.1f}s]")
        print(f"      params: " + ", ".join(f"{k}={v}" for k, v in bp.items()
                                              if k not in ("base_tp_pct","base_sl_pct")))

        best_per_strategy[key] = dict(
            best_params         = bp,
            default_val_sharpe  = sharpe_def_v,
            default_test_sharpe = sharpe_def_t,
            best_val_sharpe     = best["sharpe"],
            best_test_sharpe    = sharpe_te,
            best_val_trades     = best["n_trades"],
            best_test_trades    = n_te,
            best_val_winrate    = best["win_rate"],
            best_test_winrate   = wr_te,
            improvement_val     = improvement_v,
            improvement_test    = improvement_t,
            n_combos            = len(grid),
        )
        all_results.append(dict(strategy=key, **best_per_strategy[key]))

    # ── save ─────────────────────────────────────────────────────────────────
    out_json = CACHE / "results" / f"{ticker}_grid_search_results.json"
    out_json.write_text(json.dumps(best_per_strategy, indent=2, default=str))
    pd.DataFrame(all_results).to_parquet(
        CACHE / "results" / f"{ticker}_grid_search_results.parquet", index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}\n  SUMMARY\n{'='*72}")
    print(f"\n  {'strategy':<14}  {'def_val':>8}  {'best_val':>9}  {'def_test':>9}  "
          f"{'best_test':>10}  {'Δ_val':>7}  {'Δ_test':>7}  {'tr_v/tr_t':>10}")
    print("  " + "─" * 84)
    for r in all_results:
        flag_v = "★" if r["improvement_val"]  > 0 else " "
        flag_t = "★" if r["improvement_test"] > 0 else " "
        print(f"  {r['strategy']:<14}  "
              f"{r['default_val_sharpe']:>+8.3f}  "
              f"{r['best_val_sharpe']:>+8.3f}{flag_v}  "
              f"{r['default_test_sharpe']:>+9.3f}  "
              f"{r['best_test_sharpe']:>+9.3f}{flag_t}  "
              f"{r['improvement_val']:>+7.2f}  "
              f"{r['improvement_test']:>+7.2f}  "
              f"{r['best_val_trades']:>3}/{r['best_test_trades']:<6}")

    print(f"\n  ── BASELINE (CUSUM gate) ──")
    print(f"    S4_MACDTrend test Sharpe = +3.130   (per CLAUDE.md, prior eval window)")

    if all_results:
        best_test_strat = max(all_results, key=lambda r: r["best_test_sharpe"])
        print(f"\n  ── BEST GRID-TUNED TEST RESULT ──")
        print(f"    {best_test_strat['strategy']}: "
              f"val={best_test_strat['best_val_sharpe']:+.3f}  "
              f"test={best_test_strat['best_test_sharpe']:+.3f}  "
              f"({best_test_strat['best_test_trades']} test trades)")
        if best_test_strat["best_test_sharpe"] > 3.13:
            print(f"  ✓ BEATS CUSUM baseline. Deployable.")
        elif best_test_strat["best_test_sharpe"] > 0.5:
            print(f"  ⚠ Positive and meaningful but under CUSUM. Still useful.")
        elif best_test_strat["best_test_sharpe"] > 0:
            print(f"  ⚠ Marginally positive on test.")
        else:
            print(f"  ✗ Negative on test even with grid-best val params.")

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
