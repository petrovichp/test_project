"""
Walk-forward validation across the RL period.

Tests whether strategies have persistent edge across time, or whether the
prior DQN-test window (Mar–Apr 2026) was anomalous.

Splits RL period [bar 101,440, 384,614) into 6 contiguous folds; for each
(strategy × fold × mode) runs the sequential-trade simulator and records
per-fold Sharpe.

Modes:
  default          — DEFAULT_PARAMS + EXECUTION_CONFIG (current production)
  grid_best        — val-best params from cache/btc_grid_search_results.json
  default+cusum    — DEFAULT_PARAMS with CUSUM regime gate applied

Stability gate per (strategy, mode): ≥4 of 6 folds positive AND mean > 0.

Run: python3 -m models.walk_forward [ticker]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader        import load_meta
from strategy.agent     import STRATEGIES, DEFAULT_PARAMS
from execution.config   import EXECUTION_CONFIG
from models.grid_search import (_build_strategy_df, _exit_arrays,
                                  _simulate_sequential, _sharpe)

CACHE        = ROOT / "cache"
WARMUP       = 1440
RL_START_REL = 100_000        # in pq_use index space (bar 101,440 absolute)
RL_END_REL   = 383_174        # in pq_use index space (bar 384,614 absolute)
N_FOLDS      = 6

# CUSUM regime gates (per backtest/run.py:50-61)
CUSUM_GATES = {
    "S1_VolDir":    {"trend_up", "trend_down", "chop"},
    "S2_Funding":   {"ranging", "calm"},
    "S3_BBRevert":  {"ranging", "calm"},
    "S4_MACDTrend": {"trend_up", "trend_down", "chop"},
    "S6_TwoSignal": {"trend_up", "trend_down", "chop"},
    "S7_OIDiverg":  {"ranging", "calm"},
    "S8_TakerFlow": {"trend_up", "trend_down", "chop"},
    "S10_Squeeze":  {"trend_up", "trend_down", "chop"},
    "S12_VWAPVol":  {"ranging", "calm"},
}


def _build_default_full_params(key: str) -> dict:
    """Combine signal default params + execution config exit params into one dict."""
    p = dict(DEFAULT_PARAMS[key])
    cfg = EXECUTION_CONFIG[key]
    p["base_tp_pct"]           = cfg.exit.base_tp
    p["base_sl_pct"]           = cfg.exit.base_sl
    p["breakeven_pct"]         = cfg.exit._be
    p["time_stop_bars"]        = cfg.exit._ts
    p["trail_after_breakeven"] = cfg.exit._tab_en
    return p


def _fmt(ts) -> str:
    return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")


def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*78}")
    print(f"  WALK-FORWARD VALIDATION  ({ticker.upper()})  N_FOLDS={N_FOLDS}")
    print(f"{'='*78}")

    # ── load aligned source data ─────────────────────────────────────────────
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

    df_full = _build_strategy_df(pq_use, meta_use, price, atr_full, rk_full,
                                   dir_preds)

    rg_pq = pd.read_parquet(CACHE / "preds" / f"{ticker}_regime_cusum_v4.parquet")
    regime_full = rg_pq["state_name"].values
    assert len(regime_full) == len(pq_use)

    # ── load grid-best params ────────────────────────────────────────────────
    grid_path = CACHE / "results" / f"{ticker}_grid_search_results.json"
    if grid_path.exists():
        grid_best = json.loads(grid_path.read_text())
        print(f"  grid_search_results loaded: {list(grid_best.keys())}")
    else:
        grid_best = {}
        print("  warning: grid_search_results.json not found — skipping grid_best mode")

    # ── compute fold boundaries ──────────────────────────────────────────────
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        a = RL_START_REL + i * fold_size
        b = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        folds.append((a, b))

    ts_arr = pq_use["timestamp"].values
    print(f"\n  RL period: bars [{RL_START_REL:,}, {RL_END_REL:,})  "
          f"({RL_END_REL - RL_START_REL:,} bars)")
    print(f"  Folds (~{fold_size:,} bars each):")
    for i, (a, b) in enumerate(folds):
        print(f"    fold {i+1}:  bars [{a:>7,}, {b:>7,})  "
              f"{_fmt(ts_arr[a])} → {_fmt(ts_arr[b-1])}  ({b-a:,} bars)")

    # ── evaluate per (strategy × mode × fold) ────────────────────────────────
    strats = ["S1_VolDir", "S4_MACDTrend", "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow"]
    modes  = ["default", "grid_best", "default+cusum"]

    # jit warmup
    print(f"\n  JIT-warm sequential simulator ...")
    _ = _simulate_sequential(
        np.zeros(20, dtype=np.int8), price[:20],
        np.full(20, 0.02, dtype=np.float32), np.full(20, 0.005, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.int32),
    )

    print(f"\n  Running {len(strats)} strats × {len(modes)} modes × {N_FOLDS} folds ...")
    rows = []

    for strat_key in strats:
        fn, _ = STRATEGIES[strat_key]
        for mode in modes:
            if mode == "default":
                params = _build_default_full_params(strat_key)
                use_cusum = False
            elif mode == "grid_best":
                if strat_key not in grid_best:
                    continue
                params = grid_best[strat_key]["best_params"]
                use_cusum = False
            else:                                # default+cusum
                params = _build_default_full_params(strat_key)
                use_cusum = True

            for i, (a, b) in enumerate(folds):
                df_fold     = df_full.iloc[a:b].reset_index(drop=True)
                price_fold  = price[a:b]
                atr_fold    = atr_full[a:b]
                regime_fold = regime_full[a:b]
                n_fold      = b - a

                sigs, _, _ = fn(df_fold, params)
                sigs       = np.asarray(sigs, dtype=np.int8)

                if use_cusum:
                    allowed   = CUSUM_GATES.get(strat_key, set())
                    gate_mask = np.isin(regime_fold, list(allowed))
                    sigs      = sigs * gate_mask.astype(np.int8)

                tp, sl, tr, tab, be, ts_bars = _exit_arrays(
                    atr_fold,
                    params["base_tp_pct"], params["base_sl_pct"], atr_med,
                    params["breakeven_pct"], params["time_stop_bars"],
                    params["trail_after_breakeven"],
                )
                pnls, durs = _simulate_sequential(
                    sigs, price_fold, tp, sl, tr, tab, be, ts_bars)

                sharpe = _sharpe(pnls, n_fold)
                n_t    = len(pnls)
                wr     = float((pnls > 0).mean()) if n_t else 0.0
                tot    = float(pnls.sum()) if n_t else 0.0

                rows.append(dict(
                    strategy=strat_key, mode=mode, fold=i + 1,
                    fold_start=_fmt(ts_arr[a]), fold_end=_fmt(ts_arr[b - 1]),
                    n_bars=n_fold, n_trades=n_t,
                    sharpe=sharpe, win_rate=wr, total_pnl=tot,
                ))

    df_results = pd.DataFrame(rows)
    df_results.to_parquet(CACHE / "results" / f"{ticker}_walk_forward_results.parquet", index=False)

    # ── per-fold tables ──────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  WALK-FORWARD SHARPE PER FOLD\n{'='*78}")
    for mode in modes:
        print(f"\n  ── mode: {mode} ──")
        hdr = f"  {'strategy':<14} " + " ".join(f"{f'fold{i+1}':>8}"
                                                 for i in range(N_FOLDS))
        print(hdr + f"  {'mean':>7}  {'std':>5}  pos/{N_FOLDS}")
        print("  " + "─" * (len(hdr) + 22))
        for strat_key in strats:
            sub = df_results[(df_results["strategy"] == strat_key) &
                              (df_results["mode"] == mode)]
            if len(sub) == 0:
                continue
            sharps = sub.sort_values("fold")["sharpe"].values
            mean   = float(sharps.mean())
            std    = float(sharps.std())
            n_pos  = int((sharps > 0).sum())
            stable = "★" if (n_pos >= N_FOLDS // 2 + 1 and mean > 0) else " "
            line   = f"  {strat_key:<14} " + " ".join(f"{s:>+8.2f}" for s in sharps)
            line  += f"  {mean:>+6.2f}  {std:>5.2f}  {n_pos}/{N_FOLDS} {stable}"
            print(line)

    # ── per-fold trade counts (sanity) ───────────────────────────────────────
    print(f"\n\n  Trade counts per fold (mode=default):")
    print(f"  {'strategy':<14} " + " ".join(f"{f'fold{i+1}':>7}" for i in range(N_FOLDS)))
    for strat_key in strats:
        sub = df_results[(df_results["strategy"] == strat_key) &
                          (df_results["mode"] == "default")]
        if len(sub) == 0:
            continue
        cnts = sub.sort_values("fold")["n_trades"].values
        print(f"  {strat_key:<14} " + " ".join(f"{c:>7,}" for c in cnts))

    # ── stability gate ───────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  STABILITY GATE  (≥{N_FOLDS//2 + 1}/{N_FOLDS} folds Sharpe>0 AND mean>0)\n{'='*78}")
    stable = []
    for strat_key in strats:
        for mode in modes:
            sub = df_results[(df_results["strategy"] == strat_key) &
                              (df_results["mode"] == mode)]
            if len(sub) == 0:
                continue
            sharps = sub["sharpe"].values
            n_pos  = int((sharps > 0).sum())
            mean   = float(sharps.mean())
            std    = float(sharps.std())
            if n_pos >= N_FOLDS // 2 + 1 and mean > 0:
                stable.append((strat_key, mode, n_pos, mean, std))

    if stable:
        print(f"\n  ✓ Stable strategy/mode combinations:")
        print(f"    {'strategy':<14}  {'mode':<14}  {'pos/' + str(N_FOLDS):>5}  "
              f"{'mean':>6}  {'std':>5}")
        for s, m, n_pos, mean, std in sorted(stable, key=lambda x: -x[3]):
            print(f"    {s:<14}  {m:<14}  {n_pos}/{N_FOLDS}    "
                  f"{mean:>+6.2f}  {std:>5.2f}")
    else:
        print(f"\n  ✗ NO strategy/mode combination passes the stability gate.")
        print(f"    No persistent edge across the RL period.")

    # ── extra: which folds are universally hard? ────────────────────────────
    print(f"\n\n  Per-fold mean Sharpe (across all strategy/mode combos):")
    for i in range(N_FOLDS):
        sub = df_results[df_results["fold"] == i + 1]
        line = f"    fold {i+1}: mean={sub['sharpe'].mean():>+6.2f}  "
        line += f"median={sub['sharpe'].median():>+6.2f}  "
        line += f"({sub['fold_start'].iloc[0]} → {sub['fold_end'].iloc[0]})"
        print(line)

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
