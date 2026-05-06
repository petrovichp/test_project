"""
Experiment 1 — widen time stops on mean-reversion strategies.

Hypothesis: S2/S3/S7 exit predominantly via TIME (60-90% of trades).
The signal may be right but reversion takes longer than current time stop.

Strategies tested (current vs candidates):
  S2_Funding:   60 → [60, 120, 240, 480]      bars
  S3_BBRevert:  30 → [30, 60, 90, 120]        bars
  S7_OIDiverg:  45 → [45, 90, 180, 360]       bars

Decision rule: keep variant if TIME% < 30% AND min(val_sharpe, test_sharpe) improves.

Run: python3 -m experiments.exp1_time_stops
"""

import sys, time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.run     import _load_splits, _make_df
from backtest.preds   import _vol_preds, _dir_preds
from backtest.engine  import run as engine_run
from strategy.agent   import STRATEGIES, DEFAULT_PARAMS
from execution.config import EXECUTION_CONFIG
from execution.sizing import VolScaledSizer

VARIANTS = {
    "S2_Funding":  [60, 120, 240, 480],
    "S3_BBRevert": [30, 60, 90, 120],
    "S7_OIDiverg": [45, 90, 180, 360],
}


def main(ticker: str = "btc"):
    print(f"\n{'='*88}")
    print(f"  EXPERIMENT 1 — widen time stops on mean-reversion strategies ({ticker.upper()})")
    print(f"{'='*88}\n")

    t0 = time.perf_counter()

    # ── load splits + cached predictions ──────────────────────────────────────
    data = _load_splits(ticker)
    fc   = data["feat_cols"]
    atr_tr, atr_v, atr_te, rank_tr, rank_v, rank_te = _vol_preds(
        ticker, data["X_sc_tr"], data["X_sc_v"], data["X_sc_te"])
    dir_all = _dir_preds(ticker, data["X_sc_tr"], data["X_sc_v"], data["X_sc_te"],
                          fc, rank_tr, rank_v, rank_te)
    dir_v  = {col: arrs[1] for col, arrs in dir_all.items()}
    dir_te = {col: arrs[2] for col, arrs in dir_all.items()}

    me     = data.get("meta_extra", {})
    df_val  = _make_df(data["X_raw_v"],  fc, data["price_v"],
                       atr_v,  rank_v,  dir_v,  data["ts_v"],  me)
    df_test = _make_df(data["X_raw_te"], fc, data["price_te"],
                       atr_te, rank_te, dir_te, data["ts_te"], me)
    atr_median = float(np.median(atr_tr))
    print(f"  Setup ready in {time.perf_counter()-t0:.1f}s — running variants ...\n")

    rows = []

    for strat_key, ts_list in VARIANTS.items():
        fn, _    = STRATEGIES[strat_key]
        params   = DEFAULT_PARAMS[strat_key]
        cfg      = EXECUTION_CONFIG[strat_key]
        cur_ts   = ts_list[0]   # current = first

        for split_label, df_s, ts_arr, atr_arr, price_arr in [
            ("val",  df_val,  data["ts_v"],  atr_v,  data["price_v"]),
            ("test", df_test, data["ts_te"], atr_te, data["price_te"]),
        ]:
            n = len(price_arr)

            # signals (cached per strategy/split — independent of time stop)
            raw_sigs = fn(df_s, params)[0]
            sigs     = cfg.entry.apply(raw_sigs)

            # TP/SL/BE arrays (also independent of time stop)
            tp_a, sl_a = cfg.exit.arrays(atr_arr, price_arr, atr_median)
            plan0      = cfg.exit.plan(atr_median, float(np.median(price_arr)), atr_median)
            trail_a    = np.zeros(n, dtype=np.float32)
            tab_a      = np.full(n, plan0.tab_pct,        dtype=np.float32)
            be_a       = np.full(n, plan0.breakeven_pct,  dtype=np.float32)

            # sizing (vectorised)
            if isinstance(cfg.sizing, VolScaledSizer):
                sz = cfg.sizing
                sz_a = np.clip(sz.target_risk / np.maximum(sl_a, 1e-6),
                                sz.min_size, sz.max_size).astype(np.float32)
            else:
                sz_a = np.full(n, cfg.sizing.fraction, dtype=np.float32)

            # iterate time-stop variants
            for ts_bars in ts_list:
                ts_a = np.full(n, ts_bars, dtype=np.int32)
                result = engine_run(
                    sigs, price_arr, tp_a, sl_a, ts_arr,
                    trail_pct_arr=trail_a, tab_pct_arr=tab_a,
                    breakeven_pct_arr=be_a, time_stop_arr=ts_a,
                    position_size_arr=sz_a, force_exit_arr=None)
                summ = result.summary()
                reasons = {}
                for t in result.trades:
                    reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
                tot = max(summ["n_trades"], 1)
                rows.append({
                    "strategy":    strat_key,
                    "ts_bars":     ts_bars,
                    "is_current":  ts_bars == cur_ts,
                    "split":       split_label,
                    "sharpe":      summ["sharpe"],
                    "n_trades":    summ["n_trades"],
                    "win_rate":    summ["win_rate"],
                    "total_ret":   summ["total_return"],
                    "max_dd":      summ["max_drawdown"],
                    "tp":          reasons.get("TP",   0),
                    "sl":          reasons.get("SL",   0),
                    "be":          reasons.get("BE",   0),
                    "tsl":         reasons.get("TSL",  0),
                    "time":        reasons.get("TIME", 0),
                    "time_pct":    100 * reasons.get("TIME", 0) / tot,
                })

    df = pd.DataFrame(rows)

    # ── per-strategy comparison ───────────────────────────────────────────────
    for strat_key, ts_list in VARIANTS.items():
        sub = df[df.strategy == strat_key]
        cur_ts = ts_list[0]
        print(f"\n  ── {strat_key}  (current TS={cur_ts} bars) ──")
        print(f"  {'TS':>5}  {'split':<5}  {'Sharpe':>7}  {'Tr':>4}  {'Win%':>5}  "
              f"{'Ret%':>7}  {'DD%':>6}  {'TP':>3} {'SL':>3} {'BE':>3} {'TSL':>3} {'TIME':>4}  "
              f"{'TIME%':>6}")
        print("  " + "─" * 80)
        for ts_bars in ts_list:
            for sp in ["val", "test"]:
                r = sub[(sub.ts_bars == ts_bars) & (sub.split == sp)].iloc[0]
                marker = "*" if ts_bars == cur_ts else " "
                print(f"  {ts_bars:>4}{marker} {sp:<5}  {r.sharpe:>7.3f}  {int(r.n_trades):>4}  "
                      f"{r.win_rate:>4.0f}%  {r.total_ret:>6.2f}%  {r.max_dd:>5.1f}  "
                      f"{int(r.tp):>3} {int(r.sl):>3} {int(r.be):>3} {int(r.tsl):>3} {int(r.time):>4}  "
                      f"{r.time_pct:>5.0f}%")

    # ── decision summary ──────────────────────────────────────────────────────
    print(f"\n\n{'='*88}")
    print(f"  DECISION SUMMARY  (rule: TIME% < 30% AND min(val_sharpe,test_sharpe) > current min)")
    print(f"{'='*88}")
    print(f"\n  {'Strategy':<14}  {'Variant':<10}  {'min(V,T) Sharpe':>16}  "
          f"{'Δ vs current':>13}  {'val TIME%':>10}  {'test TIME%':>10}  Verdict")
    print("  " + "─" * 90)

    for strat_key, ts_list in VARIANTS.items():
        cur_ts  = ts_list[0]
        cur_min = min(
            df[(df.strategy == strat_key) & (df.ts_bars == cur_ts) & (df.split == "val")].sharpe.iloc[0],
            df[(df.strategy == strat_key) & (df.ts_bars == cur_ts) & (df.split == "test")].sharpe.iloc[0],
        )
        for ts_bars in ts_list:
            v = df[(df.strategy == strat_key) & (df.ts_bars == ts_bars) & (df.split == "val")].iloc[0]
            t = df[(df.strategy == strat_key) & (df.ts_bars == ts_bars) & (df.split == "test")].iloc[0]
            mn   = min(v.sharpe, t.sharpe)
            dv   = mn - cur_min
            keep = (v.time_pct < 30 and t.time_pct < 30 and dv > 0 and ts_bars != cur_ts)
            verdict = "KEEP ✓" if keep else ("current" if ts_bars == cur_ts else "—")
            print(f"  {strat_key:<14}  {f'TS={ts_bars}':<10}  "
                  f"{mn:>15.3f}  {dv:>+12.3f}  {v.time_pct:>9.0f}%  {t.time_pct:>9.0f}%  {verdict}")

    print(f"\n  Total elapsed: {time.perf_counter()-t0:.1f}s")
    out = ROOT / "cache" / f"{ticker}_exp1_time_stops.parquet"
    df.to_parquet(out, index=False)
    print(f"  Detailed results → {out.name}")

    return df


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    main(ticker)
