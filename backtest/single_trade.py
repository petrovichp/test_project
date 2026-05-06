"""
Single-trade simulator (numba-jit) — DQN training inner loop.

Mirrors the 5-step exit hierarchy in backtest/engine.py:
  1. Force exit (not used in single-trade variant — DQN re-decides each bar)
  2. Time stop
  3. Breakeven ratchet (then trail-after-breakeven)
  4. Trailing SL ratchet
  5. TP / SL touch

Differences from engine.py:
  - One trade per call (no equity tracking, no position_size scaling — DQN
    treats `pnl_pct` directly as the reward).
  - 1-bar entry lag handled by caller: caller passes `entry_bar` = T+1, where
    the signal fired at bar T.
  - No per-bar breakeven/trail/time_stop arrays — single scalars passed in
    (these are constant for a given strategy/trade per the EXECUTION_CONFIG).

Exit reason IDs:
  0 = TP, 1 = SL, 2 = TSL, 3 = BE, 4 = TIME, 5 = EOD (hit price-window end)

Parity with engine.py is verified by `parity_test()` in this file:
  random sample of trades from the existing backtest run is replayed through
  this simulator and PnL is compared to engine output (must match within 1e-9).

Run: python3 -m backtest.single_trade   # runs the parity test
"""

import sys, time
import numpy as np
from pathlib import Path
from numba import njit, types
from numba.typed import List as NumbaList

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.costs import TAKER_FEE

EXIT_TP, EXIT_SL, EXIT_TSL, EXIT_BE, EXIT_TIME, EXIT_EOD = 0, 1, 2, 3, 4, 5
EXIT_NAMES = ("TP", "SL", "TSL", "BE", "TIME", "EOD")


@njit(cache=True, fastmath=False)
def simulate_one_trade(
    prices:         np.ndarray,    # full price array (read-only)
    entry_bar:      int,           # absolute bar index where entry executes
    direction:      int,           # +1 long, -1 short
    tp_pct:         float,
    sl_pct:         float,
    trail_pct:      float,
    tab_pct:        float,
    breakeven_pct:  float,
    time_stop_bars: int,
    max_lookahead:  int,           # cap on i - entry_bar; 0 = no cap (use prices end)
):
    """Simulate one trade. Returns (pnl_pct, n_bars_held, exit_reason).

    Mirrors backtest/engine.py logic exactly. Entry executes at `entry_bar`
    close; exit is checked starting at `entry_bar + 1`.

    Returns:
      pnl_pct       : net PnL fraction (after 2× taker fee)
      n_bars_held   : exit_bar - entry_bar
      exit_reason   : 0=TP, 1=SL, 2=TSL, 3=BE, 4=TIME, 5=EOD
    """
    n = len(prices)

    # Entry priced with taker fee slippage (matches engine.py:220)
    entry = prices[entry_bar] * (1.0 + direction * TAKER_FEE)

    # Initial TP/SL prices (engine.py:223-224)
    tp = entry * (1.0 + direction * tp_pct)
    sl = entry * (1.0 - direction * sl_pct)

    # Validity check (engine.py:232-233) — match engine's silent-skip behavior
    if direction == 1:
        if not (tp > entry and sl < entry):
            return 0.0, 0, EXIT_EOD
    else:
        if not (tp < entry and sl > entry):
            return 0.0, 0, EXIT_EOD

    cur_trail   = trail_pct                  # mutable (tab_pct may overwrite at BE)
    be_done     = False                      # breakeven triggered?

    end = n if max_lookahead <= 0 else min(n, entry_bar + 1 + max_lookahead)

    for i in range(entry_bar + 1, end):
        price = prices[i]

        # ── 2. Time stop ──────────────────────────────────────────────────────
        if time_stop_bars > 0 and (i - entry_bar) >= time_stop_bars:
            raw_pnl = direction * (price / entry - 1.0)
            return raw_pnl - 2.0 * TAKER_FEE, i - entry_bar, EXIT_TIME

        # ── 3. Breakeven ratchet ──────────────────────────────────────────────
        if breakeven_pct > 0.0 and not be_done:
            unrealised = direction * (price / entry - 1.0)
            if unrealised >= breakeven_pct:
                sl       = entry
                be_done  = True
                if tab_pct > 0.0:           # trail-after-breakeven activates
                    cur_trail = tab_pct

        # ── 4. Trailing SL ratchet ────────────────────────────────────────────
        if cur_trail > 0.0:
            if direction == 1:
                cand = price * (1.0 - cur_trail)
                if cand > sl:
                    sl = cand
            else:
                cand = price * (1.0 + cur_trail)
                if cand < sl:
                    sl = cand

        # ── 5. TP / SL check ──────────────────────────────────────────────────
        hit_tp = (direction ==  1 and price >= tp) or \
                 (direction == -1 and price <= tp)
        hit_sl = (direction ==  1 and price <= sl) or \
                 (direction == -1 and price >= sl)

        if hit_tp or hit_sl:
            exit_price = tp if hit_tp else sl
            raw_pnl    = direction * (exit_price / entry - 1.0)
            net_pnl    = raw_pnl - 2.0 * TAKER_FEE
            if hit_tp:
                reason = EXIT_TP
            elif cur_trail > 0.0 and not be_done:
                reason = EXIT_TSL
            elif be_done:
                reason = EXIT_BE
            else:
                reason = EXIT_SL
            return net_pnl, i - entry_bar, reason

    # ── 6. Forced close at end of price window (EOD) ──────────────────────────
    last_price = prices[end - 1]
    raw_pnl    = direction * (last_price / entry - 1.0)
    return raw_pnl - 2.0 * TAKER_FEE, end - 1 - entry_bar, EXIT_EOD


# ── parity test ──────────────────────────────────────────────────────────────

def parity_test(ticker: str = "btc", n_sample: int = 50, seed: int = 42):
    """Replay a sample of trades from the existing backtest through both
    `backtest.engine.run` (single-strategy) and this simulator. Compare PnL
    bar-by-bar.

    Strategy: re-run engine.run on the test split with one strategy, capture
    each trade's (entry_bar, direction, tp_pct, sl_pct, exit_reason, pnl_pct),
    then call simulate_one_trade() with the same inputs and assert match.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    from data.loader        import load_meta
    from data.gaps          import clean_mask
    from models.splits      import sequential
    from backtest.engine    import run as engine_run
    from strategy.agent     import STRATEGIES, DEFAULT_PARAMS
    from execution.config   import EXECUTION_CONFIG
    from execution.sizing   import VolScaledSizer

    print(f"\n{'='*70}\n  SINGLE-TRADE SIMULATOR — PARITY TEST  ({ticker.upper()})\n{'='*70}")

    # ── load existing v3 backtest data path (faster + closer to engine.py runtime) ─
    pq        = pd.read_parquet(ROOT / "cache" / f"{ticker}_features_assembled.parquet")
    feat_cols = [c for c in pq.columns if c != "timestamp"]
    meta      = load_meta(ticker)
    ts_meta   = meta["timestamp"].values
    price_meta= meta["perp_ask_price"].values
    gap_ok    = clean_mask(pd.Series(ts_meta), max_lookback=1440)
    X_raw     = pq[feat_cols].values
    ts_all    = pq["timestamp"].values
    row_ok    = gap_ok & ~np.isnan(X_raw).any(axis=1)
    X_clean   = X_raw[row_ok]
    ts_clean  = ts_all[row_ok]
    n         = len(X_clean)
    sp        = sequential(n, 0.50, 0.25)
    ts_to_pr  = dict(zip(ts_meta, price_meta))
    price_te  = np.array([ts_to_pr[t] for t in ts_clean[sp.test]])

    # vol preds (re-use cached v3)
    import lightgbm as lgb
    sc = StandardScaler(); sc.fit(X_clean[sp.train])
    X_sc_te = sc.transform(X_clean[sp.test])
    vol_model = lgb.Booster(model_file=str(ROOT / "cache" / f"{ticker}_lgbm_atr_30.txt"))
    atr_te    = vol_model.predict(X_sc_te)
    sorted_tr = np.sort(vol_model.predict(sc.transform(X_clean[sp.train])))
    rank_te   = np.clip(np.searchsorted(sorted_tr, atr_te) / len(sorted_tr), 0, 1)

    # use a strategy that fires reasonably often: S8_TakerFlow (no direction needed)
    strat_key  = "S8_TakerFlow"
    fn, _      = STRATEGIES[strat_key]
    params     = DEFAULT_PARAMS[strat_key]
    exec_cfg   = EXECUTION_CONFIG[strat_key]

    # build strategy DataFrame
    df = pd.DataFrame({"price": price_te, "atr_pred": atr_te, "vol_pred": rank_te})
    for c in ["bb_pct_b", "bb_width", "rsi_6", "rsi_14", "macd_hist",
              "ofi_perp_10_r15", "ofi_perp_10", "taker_imb_5", "taker_net_15",
              "fund_rate", "fund_mom_480", "ret_sma_200", "vwap_dev_1440",
              "sma_50", "sma_200",
              "oi_price_div_15", "taker_net_30", "taker_net_60",
              "taker_imb_30", "ret_15", "vwap_dev_240",
              "vol_z_spot_60", "spot_imbalance", "perp_imbalance",
              "diff_price"]:
        idx = feat_cols.index(c) if c in feat_cols else None
        if idx is not None:
            df[c] = X_clean[sp.test, idx]
        else:
            df[c] = 0.0
    df["p_up_60"] = df["p_dn_60"] = df["p_up_100"] = df["p_dn_100"] = 0.5

    raw_sigs, _, _ = fn(df, params)
    sigs = exec_cfg.entry.apply(raw_sigs)

    atr_med = float(np.median(vol_model.predict(sc.transform(X_clean[sp.train]))))
    n_bars  = len(price_te)

    if hasattr(exec_cfg.exit, "arrays"):
        tp_arr_e, sl_arr_e = exec_cfg.exit.arrays(atr_te, price_te, atr_med)
        plan0 = exec_cfg.exit.plan(atr_med, float(np.median(price_te)), atr_med)
        trail_arr = np.zeros(n_bars, dtype=np.float32)
        tab_arr   = np.full(n_bars, plan0.tab_pct,        dtype=np.float32)
        be_arr    = np.full(n_bars, plan0.breakeven_pct,  dtype=np.float32)
        ts_arr_e  = np.full(n_bars, plan0.time_stop_bars, dtype=np.int32)
    else:
        tp_arr_e  = np.full(n_bars, params["tp_pct"],  dtype=np.float32)
        sl_arr_e  = np.full(n_bars, params["sl_pct"],  dtype=np.float32)
        trail_arr = np.full(n_bars, params.get("trail_pct", 0.0), dtype=np.float32)
        tab_arr   = np.zeros(n_bars, dtype=np.float32)
        be_arr    = np.zeros(n_bars, dtype=np.float32)
        ts_arr_e  = np.zeros(n_bars, dtype=np.int32)

    if isinstance(exec_cfg.sizing, VolScaledSizer):
        sz = exec_cfg.sizing
        size_arr = np.clip(sz.target_risk / np.maximum(sl_arr_e, 1e-6),
                            sz.min_size, sz.max_size).astype(np.float32)
    else:
        size_arr = np.full(n_bars, exec_cfg.sizing.fraction, dtype=np.float32)

    # ── run engine ────────────────────────────────────────────────────────────
    res = engine_run(
        sigs, price_te, tp_arr_e, sl_arr_e, ts_clean[sp.test],
        trail_pct_arr     = trail_arr,
        tab_pct_arr       = tab_arr,
        breakeven_pct_arr = be_arr,
        time_stop_arr     = ts_arr_e,
        position_size_arr = size_arr,
        force_exit_arr    = None,
    )
    print(f"  Engine produced {len(res.trades):,} trades. Sampling {n_sample} for parity check ...")
    if len(res.trades) < n_sample:
        n_sample = len(res.trades)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(res.trades), size=n_sample, replace=False)

    # ── compile + warm up jit ─────────────────────────────────────────────────
    print("  JIT-compiling simulator (first call) ...")
    t0 = time.perf_counter()
    _ = simulate_one_trade(price_te, 100, 1, 0.01, 0.005, 0.0, 0.0, 0.0, 0, 0)
    print(f"    compiled in {time.perf_counter()-t0:.2f}s")

    # ── compare ───────────────────────────────────────────────────────────────
    n_pass = 0; n_fail = 0
    print(f"\n  {'#':>4}  {'bar_in':>7} {'dir':>4}  {'engine PnL%':>11}  {'sim PnL%':>11}  "
          f"{'engine reason':>13}  {'sim reason':>11}  {'ΔPnL':>10}  {'match':>7}")
    print("  " + "─"*85)

    for k, idx in enumerate(idxs):
        t = res.trades[idx]
        bar_in    = t.bar_in              # this is the bar where entry EXECUTED (engine.py:218 → signals[i-1])
        direction = t.direction
        tp_pct    = float(tp_arr_e[bar_in - 1])
        sl_pct    = float(sl_arr_e[bar_in - 1])
        trail     = float(trail_arr[bar_in - 1])
        tab       = float(tab_arr[bar_in - 1])
        be_pct    = float(be_arr[bar_in - 1])
        ts_bars   = int(ts_arr_e[bar_in - 1])

        sim_pnl, sim_n, sim_reason = simulate_one_trade(
            price_te, bar_in, direction, tp_pct, sl_pct,
            trail, tab, be_pct, ts_bars, max_lookahead=0,
        )

        diff = abs(sim_pnl - t.pnl_pct)
        same_reason = (EXIT_NAMES[sim_reason] == t.exit_reason)
        ok = (diff < 1e-9) and same_reason
        if ok: n_pass += 1
        else:  n_fail += 1
        if k < 30 or not ok:
            print(f"  {k:>4}  {bar_in:>7} {direction:>+4}  "
                  f"{t.pnl_pct*100:>+10.4f}%  {sim_pnl*100:>+10.4f}%  "
                  f"{t.exit_reason:>13}  {EXIT_NAMES[sim_reason]:>11}  "
                  f"{diff:>10.2e}  {'✓' if ok else '✗':>7}")

    print(f"\n  Pass: {n_pass}/{n_sample}   Fail: {n_fail}/{n_sample}")
    if n_fail == 0:
        print(f"  ✓ Parity verified — single-trade sim matches engine.run() within 1e-9 PnL")
    else:
        print(f"  ✗ Parity FAILED on {n_fail} trades — investigate before proceeding")

    # ── speed benchmark ───────────────────────────────────────────────────────
    print(f"\n  Speed benchmark — {n_sample*100} simulated trades:")
    t0 = time.perf_counter()
    for _ in range(100):
        for idx in idxs:
            t = res.trades[idx]
            bar_in = t.bar_in
            simulate_one_trade(
                price_te, bar_in, t.direction,
                float(tp_arr_e[bar_in-1]), float(sl_arr_e[bar_in-1]),
                float(trail_arr[bar_in-1]), float(tab_arr[bar_in-1]),
                float(be_arr[bar_in-1]),    int(ts_arr_e[bar_in-1]),
                0,
            )
    elapsed = time.perf_counter() - t0
    n_calls = n_sample * 100
    per_call_us = elapsed / n_calls * 1e6
    print(f"    {n_calls:,} calls in {elapsed:.3f}s  →  {per_call_us:.1f} µs/call")
    if per_call_us < 100:
        print(f"    ✓ <100 µs/call — DQN training-loop budget OK (540k steps × 100 µs = {540_000 * per_call_us / 1e6:.0f}s)")
    else:
        print(f"    ⚠ >100 µs/call — DQN training will be slow")

    return n_fail == 0


if __name__ == "__main__":
    ok = parity_test(sys.argv[1] if len(sys.argv) > 1 else "btc")
    sys.exit(0 if ok else 1)
