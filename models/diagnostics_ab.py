"""
Path 1 diagnostics — 1a fee-free walk-forward + 1b optimal-exit oracle.

1a: Re-runs walk-forward with TAKER_FEE = 0. If strategies become positive,
    the fee model is the killer → execution research is the answer.
    If they remain negative, fees aren't the issue → signal is.

1b: For each bar where a strategy fires, computes the BEST achievable PnL
    if we could pick the optimal exit within H bars. Bounds the achievable
    Sharpe ceiling — answers "can ANY exit logic make these strategies work?"

Run: python3 -m models.diagnostics_ab [ticker]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader        import load_meta
from strategy.agent     import STRATEGIES, DEFAULT_PARAMS
from execution.config   import EXECUTION_CONFIG
from models.grid_search import (_build_strategy_df, _exit_arrays, _sharpe)
from models.walk_forward import (_build_default_full_params, CUSUM_GATES,
                                   RL_START_REL, RL_END_REL, N_FOLDS, _fmt)
from backtest.costs     import TAKER_FEE

CACHE = ROOT / "cache"
WARMUP = 1440
ORACLE_LOOKAHEAD = 60       # 60 1-min bars = 1 hour cap for oracle exit


# ── parameterized simulators (fee as argument) ───────────────────────────────

@njit(cache=True, fastmath=False)
def _simulate_one_trade_fee(prices, entry_bar, direction,
                              tp_pct, sl_pct, trail_pct, tab_pct,
                              breakeven_pct, time_stop_bars, max_lookahead, fee):
    """Same as backtest.single_trade.simulate_one_trade but `fee` is a parameter."""
    n = len(prices)
    entry = prices[entry_bar] * (1.0 + direction * fee)
    tp = entry * (1.0 + direction * tp_pct)
    sl = entry * (1.0 - direction * sl_pct)
    if direction == 1:
        if not (tp > entry and sl < entry):
            return 0.0, 0
    else:
        if not (tp < entry and sl > entry):
            return 0.0, 0

    cur_trail = trail_pct
    be_done   = False
    end       = n if max_lookahead <= 0 else min(n, entry_bar + 1 + max_lookahead)

    for i in range(entry_bar + 1, end):
        price = prices[i]
        if time_stop_bars > 0 and (i - entry_bar) >= time_stop_bars:
            return direction * (price / entry - 1.0) - 2.0 * fee, i - entry_bar
        if breakeven_pct > 0.0 and not be_done:
            if direction * (price / entry - 1.0) >= breakeven_pct:
                sl       = entry
                be_done  = True
                if tab_pct > 0.0:
                    cur_trail = tab_pct
        if cur_trail > 0.0:
            if direction == 1:
                cand = price * (1.0 - cur_trail)
                if cand > sl:
                    sl = cand
            else:
                cand = price * (1.0 + cur_trail)
                if cand < sl:
                    sl = cand
        hit_tp = (direction == 1 and price >= tp) or (direction == -1 and price <= tp)
        hit_sl = (direction == 1 and price <= sl) or (direction == -1 and price >= sl)
        if hit_tp or hit_sl:
            ep = tp if hit_tp else sl
            return direction * (ep / entry - 1.0) - 2.0 * fee, i - entry_bar

    last_price = prices[end - 1]
    return direction * (last_price / entry - 1.0) - 2.0 * fee, end - 1 - entry_bar


@njit(cache=True)
def _simulate_sequential_fee(signals, prices, tp, sl, tr, tab, be, ts_bars, fee):
    n = len(signals)
    pnls = np.zeros(5000, dtype=np.float64)
    durs = np.zeros(5000, dtype=np.int32)
    cnt = 0
    t = 0
    while t < n - 1:
        s = signals[t]
        if s != 0:
            pnl, n_held = _simulate_one_trade_fee(
                prices, t + 1, int(s),
                float(tp[t]), float(sl[t]),
                float(tr[t]), float(tab[t]),
                float(be[t]), int(ts_bars[t]),
                0, fee,
            )
            if cnt < pnls.shape[0]:
                pnls[cnt] = pnl
                durs[cnt] = n_held + 1
                cnt += 1
            t = t + 1 + n_held + 1
        else:
            t += 1
    return pnls[:cnt], durs[:cnt]


# ── oracle (perfect exit within H bars) ──────────────────────────────────────

@njit(cache=True)
def _simulate_sequential_oracle(signals, prices, lookahead, fee):
    """Sequential, non-overlapping. On each fire, exit at the best price within
    `lookahead` bars after entry. Returns per-trade PnLs."""
    n = len(signals)
    pnls = np.zeros(5000, dtype=np.float64)
    durs = np.zeros(5000, dtype=np.int32)
    cnt = 0
    t = 0
    while t < n - 1:
        s = signals[t]
        if s != 0:
            direction = int(s)
            entry_bar = t + 1
            entry = prices[entry_bar] * (1.0 + direction * fee)
            end = min(n, entry_bar + 1 + lookahead)
            if end <= entry_bar + 1:
                t += 1
                continue
            best_idx = entry_bar + 1
            best_pnl = direction * (prices[best_idx] / entry - 1.0) - 2.0 * fee
            for i in range(entry_bar + 1, end):
                p = direction * (prices[i] / entry - 1.0) - 2.0 * fee
                if p > best_pnl:
                    best_pnl = p
                    best_idx = i
            n_held = best_idx - entry_bar
            if cnt < pnls.shape[0]:
                pnls[cnt] = best_pnl
                durs[cnt] = n_held + 1
                cnt += 1
            t = entry_bar + n_held + 1
        else:
            t += 1
    return pnls[:cnt], durs[:cnt]


# ── shared loader ────────────────────────────────────────────────────────────

def _load_full(ticker: str):
    pq    = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    meta  = load_meta(ticker)
    assert (pq["timestamp"].values == meta["timestamp"].values).all()

    vol      = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_full = pd.Series(vol["atr"]).ffill().bfill().values.astype(np.float32)
    rk_full  = pd.Series(vol["rank"]).ffill().bfill().values.astype(np.float32)
    atr_med  = float(vol["atr_train_median"])

    dir_preds = {}
    for col in ["up_60", "down_60", "up_100", "down_100"]:
        dir_preds[col] = np.load(CACHE / f"{ticker}_pred_dir_{col}_v4.npz")["preds"]

    pq_use   = pq.iloc[WARMUP:].reset_index(drop=True)
    meta_use = meta.iloc[WARMUP:].reset_index(drop=True)
    price    = meta_use["perp_ask_price"].values.astype(np.float64)

    df_full = _build_strategy_df(pq_use, meta_use, price, atr_full, rk_full,
                                   dir_preds)

    rg = pd.read_parquet(CACHE / f"{ticker}_regime_cusum_v4.parquet")
    return dict(pq_use=pq_use, meta_use=meta_use, price=price,
                df_full=df_full, atr_full=atr_full, rk_full=rk_full,
                atr_med=atr_med, regime_full=rg["state_name"].values,
                ts_arr=pq_use["timestamp"].values)


def _fold_boundaries():
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        a = RL_START_REL + i * fold_size
        b = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        folds.append((a, b))
    return folds


# ── 1a: fee-free walk-forward ────────────────────────────────────────────────

def fee_free_walk_forward(d: dict, ticker: str):
    print(f"\n{'='*78}\n  1a — FEE-FREE WALK-FORWARD  (fee=0 vs fee={TAKER_FEE*100:.2f}%/side)\n{'='*78}")

    # jit warmup
    _ = _simulate_sequential_fee(
        np.zeros(20, dtype=np.int8), d["price"][:20],
        np.full(20, 0.02, dtype=np.float32), np.full(20, 0.005, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.int32),
        0.0,
    )

    folds = _fold_boundaries()
    strats = ["S1_VolDir", "S4_MACDTrend", "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow"]

    # We compare: default-params @ fee=TAKER_FEE  vs  default-params @ fee=0
    # (Mode = "default" only; grid-best and CUSUM are independent dimensions
    # and reusing them here would conflate effects.)
    rows = []
    for strat_key in strats:
        fn, _ = STRATEGIES[strat_key]
        params = _build_default_full_params(strat_key)
        for fee_label, fee_val in [("with_fee", TAKER_FEE), ("fee_free", 0.0)]:
            for i, (a, b) in enumerate(folds):
                df_fold     = d["df_full"].iloc[a:b].reset_index(drop=True)
                price_fold  = d["price"][a:b]
                atr_fold    = d["atr_full"][a:b]
                n_fold      = b - a

                sigs, _, _ = fn(df_fold, params)
                sigs       = np.asarray(sigs, dtype=np.int8)

                tp, sl, tr, tab, be, ts_bars = _exit_arrays(
                    atr_fold,
                    params["base_tp_pct"], params["base_sl_pct"], d["atr_med"],
                    params["breakeven_pct"], params["time_stop_bars"],
                    params["trail_after_breakeven"],
                )
                pnls, _ = _simulate_sequential_fee(
                    sigs, price_fold, tp, sl, tr, tab, be, ts_bars, fee_val)
                sharpe = _sharpe(pnls, n_fold)
                n_t    = len(pnls)
                wr     = float((pnls > 0).mean()) if n_t else 0.0

                rows.append(dict(
                    strategy=strat_key, fee_mode=fee_label, fold=i + 1,
                    n_trades=n_t, sharpe=sharpe, win_rate=wr,
                    total_pnl=float(pnls.sum()) if n_t else 0.0,
                    fold_start=_fmt(d["ts_arr"][a]),
                    fold_end=_fmt(d["ts_arr"][b - 1]),
                ))

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / f"{ticker}_diag_1a_fee_free.parquet", index=False)

    # ── compare: per-strategy delta ──────────────────────────────────────────
    print(f"\n  Per-fold Sharpe: with-fee → fee-free  (Δ shown)")
    print(f"  {'strategy':<14}  {'fold1':>14} {'fold2':>14} {'fold3':>14} "
          f"{'fold4':>14} {'fold5':>14} {'fold6':>14}    {'mean Δ':>7}")
    print("  " + "─" * 110)
    deltas_all = []
    for strat_key in strats:
        wf = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "with_fee")].sort_values("fold")
        ff = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "fee_free")].sort_values("fold")
        line = f"  {strat_key:<14} "
        deltas = []
        for i in range(N_FOLDS):
            wfsh = wf.iloc[i]["sharpe"]; ffsh = ff.iloc[i]["sharpe"]
            d_   = ffsh - wfsh
            deltas.append(d_)
            line += f" {wfsh:>+5.2f}→{ffsh:>+5.2f}"
        deltas = np.array(deltas)
        mean_delta = float(deltas.mean())
        deltas_all.append((strat_key, mean_delta))
        line += f"    {mean_delta:>+7.2f}"
        print(line)

    print(f"\n  Per-strategy mean Sharpe (across folds):")
    for strat_key in strats:
        wf = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "with_fee")]["sharpe"].mean()
        ff = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "fee_free")]["sharpe"].mean()
        n_pos_wf = ((df[(df["strategy"]==strat_key) & (df["fee_mode"]=="with_fee")]["sharpe"] > 0).sum())
        n_pos_ff = ((df[(df["strategy"]==strat_key) & (df["fee_mode"]=="fee_free")]["sharpe"] > 0).sum())
        flag = "★" if (n_pos_ff >= 4 and ff > 0) else " "
        print(f"    {strat_key:<14}  with-fee mean={wf:>+6.2f} (pos {n_pos_wf}/6)  "
              f"fee-free mean={ff:>+6.2f} (pos {n_pos_ff}/6)  Δ={ff-wf:>+5.2f}  {flag}")

    # ── interpretation hint ─────────────────────────────────────────────────
    grand_wf = df[df["fee_mode"]=="with_fee"]["sharpe"].mean()
    grand_ff = df[df["fee_mode"]=="fee_free"]["sharpe"].mean()
    print(f"\n  Grand mean Sharpe:  with-fee = {grand_wf:+.2f}    fee-free = {grand_ff:+.2f}    "
          f"Δ = {grand_ff-grand_wf:+.2f}")
    if grand_ff > 0.5 and grand_wf < 0:
        print(f"  → STRONG SIGNAL: fees are a primary cause. Execution research path opens.")
    elif grand_ff > 0:
        print(f"  → Partial signal: fee-free mean is positive but small. Mixed root cause.")
    else:
        print(f"  → Fees alone do NOT account for negative results. Signal is too weak even fee-free.")

    return df


# ── 1b: optimal-exit oracle ─────────────────────────────────────────────────

def optimal_exit_oracle(d: dict, ticker: str):
    print(f"\n\n{'='*78}\n  1b — OPTIMAL-EXIT ORACLE  (lookahead={ORACLE_LOOKAHEAD} bars, fee=0 + fee=on)\n{'='*78}")

    _ = _simulate_sequential_oracle(
        np.zeros(20, dtype=np.int8), d["price"][:20], 30, 0.0)

    folds = _fold_boundaries()
    strats = ["S1_VolDir", "S4_MACDTrend", "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow"]

    rows = []
    for strat_key in strats:
        fn, _ = STRATEGIES[strat_key]
        params = _build_default_full_params(strat_key)
        for fee_label, fee_val in [("with_fee", TAKER_FEE), ("fee_free", 0.0)]:
            for i, (a, b) in enumerate(folds):
                df_fold    = d["df_full"].iloc[a:b].reset_index(drop=True)
                price_fold = d["price"][a:b]
                n_fold     = b - a

                sigs, _, _ = fn(df_fold, params)
                sigs       = np.asarray(sigs, dtype=np.int8)

                pnls, _ = _simulate_sequential_oracle(
                    sigs, price_fold, ORACLE_LOOKAHEAD, fee_val)
                sharpe = _sharpe(pnls, n_fold)
                n_t    = len(pnls)
                wr     = float((pnls > 0).mean()) if n_t else 0.0

                rows.append(dict(
                    strategy=strat_key, fee_mode=fee_label, fold=i + 1,
                    n_trades=n_t, sharpe=sharpe, win_rate=wr,
                    total_pnl=float(pnls.sum()) if n_t else 0.0,
                    fold_start=_fmt(d["ts_arr"][a]),
                    fold_end=_fmt(d["ts_arr"][b - 1]),
                    mean_trade_pnl=float(pnls.mean()) if n_t else 0.0,
                ))

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / f"{ticker}_diag_1b_oracle.parquet", index=False)

    print(f"\n  Oracle Sharpe (best exit within {ORACLE_LOOKAHEAD} bars), with-fee:")
    print(f"  {'strategy':<14} " + " ".join(f"{f'fold{i+1}':>8}" for i in range(N_FOLDS))
          + f"  {'mean':>7}  {'mean win%':>9}  {'mean pnl%':>10}")
    print("  " + "─" * 95)
    for strat_key in strats:
        sub = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "with_fee")].sort_values("fold")
        sharps = sub["sharpe"].values
        line = f"  {strat_key:<14} " + " ".join(f"{s:>+8.2f}" for s in sharps)
        wr_mean = sub["win_rate"].mean() * 100
        pnl_mean = sub["mean_trade_pnl"].mean() * 100
        line += f"  {sharps.mean():>+6.2f}  {wr_mean:>8.1f}%  {pnl_mean:>+9.3f}%"
        print(line)

    print(f"\n  Oracle Sharpe (best exit within {ORACLE_LOOKAHEAD} bars), fee-free:")
    print(f"  {'strategy':<14} " + " ".join(f"{f'fold{i+1}':>8}" for i in range(N_FOLDS))
          + f"  {'mean':>7}  {'mean win%':>9}  {'mean pnl%':>10}")
    print("  " + "─" * 95)
    for strat_key in strats:
        sub = df[(df["strategy"] == strat_key) & (df["fee_mode"] == "fee_free")].sort_values("fold")
        sharps = sub["sharpe"].values
        line = f"  {strat_key:<14} " + " ".join(f"{s:>+8.2f}" for s in sharps)
        wr_mean = sub["win_rate"].mean() * 100
        pnl_mean = sub["mean_trade_pnl"].mean() * 100
        line += f"  {sharps.mean():>+6.2f}  {wr_mean:>8.1f}%  {pnl_mean:>+9.3f}%"
        print(line)

    print(f"\n  Interpretation:")
    print(f"   • Oracle Sharpe is the UPPER BOUND given the strategy's entry signals.")
    print(f"   • If oracle Sharpe is high but actual is low → exit logic is the bottleneck.")
    print(f"   • If oracle Sharpe is also low → entry signals lack predictive power.")

    avg_oracle_with_fee = df[df["fee_mode"] == "with_fee"]["sharpe"].mean()
    avg_oracle_no_fee   = df[df["fee_mode"] == "fee_free"]["sharpe"].mean()
    print(f"\n  Grand-mean oracle Sharpe (with fee): {avg_oracle_with_fee:+.2f}")
    print(f"  Grand-mean oracle Sharpe (no fee):   {avg_oracle_no_fee:+.2f}")

    return df


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*78}\n  PATH 1 DIAGNOSTICS — 1a + 1b  ({ticker.upper()})\n{'='*78}")
    d = _load_full(ticker)
    print(f"  loaded data: bars {len(d['price']):,}")

    df_a = fee_free_walk_forward(d, ticker)
    df_b = optimal_exit_oracle(d, ticker)

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
