"""
Plot BTC price vs strategy equity curves over the RL period (Sep 2025 → Apr 2026).

Three equity curves shown for each strategy:
  - "with fees"   : actual production result (default params, taker fee 0.16%)
  - "fee-free"    : same strategy with TAKER_FEE = 0  (the latent alpha)
  - "oracle"      : best possible exit within 60 bars (signal-quality ceiling)

Run: python3 -m models.plot_results [ticker] [strategy_key]
"""

import sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader        import load_meta
from strategy.agent     import STRATEGIES
from models.grid_search import _build_strategy_df, _exit_arrays
from models.walk_forward import (_build_default_full_params,
                                   RL_START_REL, RL_END_REL)
from models.diagnostics_ab import (_simulate_one_trade_fee,
                                     _simulate_sequential_oracle,
                                     ORACLE_LOOKAHEAD)
from backtest.costs     import TAKER_FEE
from numba import njit

CACHE = ROOT / "cache"
WARMUP = 1440


# ── per-bar equity curve builders ────────────────────────────────────────────

@njit(cache=True)
def _equity_curve_strategy(signals, prices, tp, sl, tr, tab, be, ts_bars, fee):
    """Walk bars, simulate sequential trades, return per-bar equity (flat
    between trades, multiplicative on close)."""
    n = len(signals)
    equity = np.ones(n, dtype=np.float64)
    eq = 1.0
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
            close_bar = min(t + 1 + n_held, n - 1)
            for i in range(t, close_bar + 1):
                equity[i] = eq
            eq *= (1.0 + pnl)
            if close_bar + 1 < n:
                for i in range(close_bar + 1, n):
                    equity[i] = eq
            t = close_bar + 1
        else:
            equity[t] = eq
            t += 1
    if t < n:
        for i in range(t, n):
            equity[i] = eq
    return equity


@njit(cache=True)
def _equity_curve_oracle(signals, prices, lookahead, fee):
    """Same shape as _equity_curve_strategy but using oracle exit."""
    n = len(signals)
    equity = np.ones(n, dtype=np.float64)
    eq = 1.0
    t = 0
    while t < n - 1:
        s = signals[t]
        if s != 0:
            direction = int(s)
            entry_bar = t + 1
            entry = prices[entry_bar] * (1.0 + direction * fee)
            end = min(n, entry_bar + 1 + lookahead)
            best_idx = entry_bar + 1
            best_pnl = direction * (prices[best_idx] / entry - 1.0) - 2.0 * fee
            for i in range(entry_bar + 1, end):
                p = direction * (prices[i] / entry - 1.0) - 2.0 * fee
                if p > best_pnl:
                    best_pnl = p
                    best_idx = i
            n_held = best_idx - entry_bar
            close_bar = min(t + 1 + n_held, n - 1)
            for i in range(t, close_bar + 1):
                equity[i] = eq
            eq *= (1.0 + best_pnl)
            if close_bar + 1 < n:
                for i in range(close_bar + 1, n):
                    equity[i] = eq
            t = close_bar + 1
        else:
            equity[t] = eq
            t += 1
    if t < n:
        for i in range(t, n):
            equity[i] = eq
    return equity


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc", strategy_key: str = "S1_VolDir"):
    t0 = time.perf_counter()
    print(f"\n  Plotting BTC vs {strategy_key} equity curves over RL period ...")

    # ── load source ──────────────────────────────────────────────────────────
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

    df_full = _build_strategy_df(pq_use, meta_use, price, atr_full, rk_full, dir_preds)

    # ── slice RL period ──────────────────────────────────────────────────────
    a, b = RL_START_REL, RL_END_REL
    df_rl    = df_full.iloc[a:b].reset_index(drop=True)
    price_rl = price[a:b]
    atr_rl   = atr_full[a:b]
    ts_rl    = pq_use["timestamp"].values[a:b]
    dt_rl    = pd.to_datetime(ts_rl, unit="s")

    # ── strategy signals & exits ─────────────────────────────────────────────
    fn, _   = STRATEGIES[strategy_key]
    params  = _build_default_full_params(strategy_key)

    sigs, _, _ = fn(df_rl, params)
    sigs       = np.asarray(sigs, dtype=np.int8)

    tp, sl, tr, tab, be, ts_bars = _exit_arrays(
        atr_rl,
        params["base_tp_pct"], params["base_sl_pct"], atr_med,
        params["breakeven_pct"], params["time_stop_bars"],
        params["trail_after_breakeven"],
    )

    # jit warmup
    _ = _equity_curve_strategy(
        np.zeros(20, dtype=np.int8), price_rl[:20],
        np.full(20, 0.02, dtype=np.float32), np.full(20, 0.005, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.int32),
        TAKER_FEE,
    )
    _ = _equity_curve_oracle(
        np.zeros(20, dtype=np.int8), price_rl[:20], 30, 0.0,
    )

    print(f"  computing equity curves ...")
    eq_with_fee = _equity_curve_strategy(
        sigs, price_rl, tp, sl, tr, tab, be, ts_bars, TAKER_FEE)
    eq_no_fee   = _equity_curve_strategy(
        sigs, price_rl, tp, sl, tr, tab, be, ts_bars, 0.0)
    eq_oracle_no_fee = _equity_curve_oracle(
        sigs, price_rl, ORACLE_LOOKAHEAD, 0.0)
    eq_oracle_with_fee = _equity_curve_oracle(
        sigs, price_rl, ORACLE_LOOKAHEAD, TAKER_FEE)

    # buy-and-hold reference (normalized to 1.0 at start)
    bh = price_rl / price_rl[0]

    n_signals = int((sigs != 0).sum())
    print(f"  signals fired: {n_signals:,}  ({n_signals/len(sigs)*100:.2f}% of bars)")
    print(f"  final equity:")
    print(f"    with-fee strategy   : {eq_with_fee[-1]:.4f}")
    print(f"    fee-free strategy   : {eq_no_fee[-1]:.4f}")
    print(f"    oracle with fee     : {eq_oracle_with_fee[-1]:.4f}")
    print(f"    oracle fee-free     : {eq_oracle_no_fee[-1]:.4f}")
    print(f"    buy-and-hold        : {bh[-1]:.4f}  (BTC ${price_rl[0]:,.0f} → ${price_rl[-1]:,.0f})")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.07},
    )

    # ─ top: BTC price ─
    ax1.plot(dt_rl, price_rl, color="#1f77b4", linewidth=0.6, label="BTC perp_ask_price")
    ax1.set_ylabel("BTC USDT", fontsize=11)
    ax1.set_title(f"BTC perpetual price vs {strategy_key} simulation  "
                   f"({dt_rl[0].strftime('%Y-%m-%d')} → {dt_rl[-1].strftime('%Y-%m-%d')})",
                   fontsize=13, pad=12)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ─ bottom: equity curves (drop oracle-no-fee because it dominates scale) ─
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax2.plot(dt_rl, eq_oracle_with_fee, color="#2ca02c", linewidth=1.4, alpha=0.95,
             label=f"oracle WITH fee  (signal-quality ceiling)  →  final {eq_oracle_with_fee[-1]:.2f}×")
    ax2.plot(dt_rl, eq_no_fee,   color="#ff7f0e", linewidth=1.5, alpha=0.95,
             label=f"strategy fee-free   (latent alpha)         →  final {eq_no_fee[-1]:.2f}×")
    ax2.plot(dt_rl, bh,          color="#1f77b4", linewidth=1.0, alpha=0.7,
             linestyle="-.",
             label=f"BTC buy-and-hold                            →  final {bh[-1]:.2f}×")
    ax2.plot(dt_rl, eq_with_fee, color="#d62728", linewidth=1.5, alpha=0.95,
             label=f"strategy WITH fee  (actual production)     →  final {eq_with_fee[-1]:.2f}×")
    ax2.set_ylabel("Equity (start = 1.00)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, linestyle=":", which="both")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}×"))

    # date formatting
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    # caption
    fig.text(0.5, 0.02,
              f"Signals: {n_signals:,} fires ({n_signals/len(sigs)*100:.2f}% of bars).  "
              f"Default params from EXECUTION_CONFIG[{strategy_key!r}].  "
              f"Sequential trades, 1-bar entry lag, ATR-scaled TP/SL, BE+trail-after-BE.  "
              f"Y-axis log-scaled.\n"
              f"Oracle fee-free upper bound (perfect 60-bar exits, no fees) reaches "
              f"{eq_oracle_no_fee[-1]:,.0f}× — dropped from plot for legibility.",
              ha="center", fontsize=8, style="italic", color="#444")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out = CACHE / f"{ticker}_{strategy_key}_equity_vs_price.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    print(f"\n  → {out}")
    print(f"  total time {time.perf_counter()-t0:.1f}s")

    plt.close()


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "btc"
    strat  = sys.argv[2] if len(sys.argv) > 2 else "S1_VolDir"
    run(ticker, strat)
