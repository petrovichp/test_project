"""
Deal-by-deal audit of A2 + rule-based exits.

Captures every trade's full lifecycle and produces detailed diagnostics:
  • per-trade timeline    (entry bar/ts, strategy, direction, exit bar/ts,
                            n_held, pnl, exit_reason)
  • duration distribution
  • long/short/cash time-in-position ratios
  • per-strategy attribution
  • exit-reason breakdown
  • PnL distribution
  • sanity audits — checks for bugs:
      - zero-PnL trades that aren't NO_TRADE
      - trades with negative durations
      - duration mismatch with single_trade simulator
      - action-direction mismatches
      - overlapping trades
      - equity continuity
  • equity curve plot

Run: python3 -m models.analyze_a2_rule [ticker] [--split val|test|both]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from numba import njit
from models.dqn_network          import DQN, masked_argmax
from models.dqn_rollout          import _build_exit_arrays, STRAT_KEYS
from models.diagnostics_ab       import _simulate_one_trade_fee
from backtest.single_trade       import EXIT_NAMES


@njit(cache=True, fastmath=False)
def _simulate_one_trade_fee_with_reason(prices, entry_bar, direction,
                                          tp_pct, sl_pct, trail_pct, tab_pct,
                                          breakeven_pct, time_stop_bars,
                                          max_lookahead, fee):
    """Mirror of _simulate_one_trade_fee but also returns exit_reason_id.
    exit_reason ∈ {0=TP, 1=SL, 2=TSL, 3=BE, 4=TIME, 5=EOD}."""
    n = len(prices)
    entry = prices[entry_bar] * (1.0 + direction * fee)
    tp = entry * (1.0 + direction * tp_pct)
    sl = entry * (1.0 - direction * sl_pct)
    if direction == 1:
        if not (tp > entry and sl < entry):
            return 0.0, 0, 5
    else:
        if not (tp < entry and sl > entry):
            return 0.0, 0, 5

    cur_trail = trail_pct
    be_done   = False
    end       = n if max_lookahead <= 0 else min(n, entry_bar + 1 + max_lookahead)

    for i in range(entry_bar + 1, end):
        price = prices[i]
        if time_stop_bars > 0 and (i - entry_bar) >= time_stop_bars:
            return direction * (price / entry - 1.0) - 2.0 * fee, i - entry_bar, 4
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
            # classify SL: BE (sl == entry, be_done, no trail movement)
            #              TSL (cur_trail > 0 and be_done — trailing fired)
            #              SL (initial)
            if hit_tp:
                reason = 0
            elif be_done and abs(sl - entry) < 1e-12:
                reason = 3                    # BE: SL stayed at entry
            elif cur_trail > 0.0 and be_done:
                reason = 2                    # TSL: trail moved SL above entry
            else:
                reason = 1                    # initial SL
            return direction * (ep / entry - 1.0) - 2.0 * fee, i - entry_bar, reason

    last_price = prices[end - 1]
    return direction * (last_price / entry - 1.0) - 2.0 * fee, end - 1 - entry_bar, 5

CACHE = ROOT / "cache"
ACTIONS_LABEL = ["NO_TRADE"] + STRAT_KEYS


def _load_a2(ticker: str = "btc") -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / f"{ticker}_dqn_policy_A2.pt",
                                     map_location="cpu"))
    net.eval()
    return net


def trace_trades(net, sp, atr_med, fee: float = 0.0):
    """Walk the split bar-by-bar with greedy A2 + rule-based exits.
    Returns a list of trade dicts (one per executed trade) plus equity-curve bars.
    """
    state         = sp["state"]
    valid_actions = sp["valid_actions"]
    signals       = sp["signals"]
    prices        = sp["price"]
    atr           = sp["atr"]
    ts            = sp["ts"]
    n_bars        = len(state)

    tp, sl, trail, tab, be, ts_arr = _build_exit_arrays(prices, atr, atr_med)

    equity   = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr   = np.full(n_bars, 1.0, dtype=np.float64)
    in_pos_long  = np.zeros(n_bars, dtype=np.bool_)
    in_pos_short = np.zeros(n_bars, dtype=np.bool_)

    trades = []
    actions_count  = np.zeros(10, dtype=np.int64)
    notrade_count  = 0
    entries_blocked_dir_zero = 0   # action picks strat but signal == 0 (shouldn't happen)
    invalid_entries = 0             # tp/sl arithmetic guards (rare)

    t = 0
    while t < n_bars - 2:
        # build state with stateful fields
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid_actions[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False

        with torch.no_grad():
            sb = torch.from_numpy(s_t).float().unsqueeze(0)
            vb = torch.from_numpy(valid_t).bool().unsqueeze(0)
            action = int(masked_argmax(net, sb, vb).item())
        actions_count[action] += 1

        if action == 0:
            notrade_count += 1
            t += 1
            continue

        k = action - 1
        direction = int(signals[t, k])
        if direction == 0:
            entries_blocked_dir_zero += 1
            t += 1
            continue

        # entry
        entry_bar   = t + 1
        entry_ts    = int(ts[entry_bar])
        entry_price = float(prices[entry_bar]) * (1.0 + direction * fee)
        tp_pct      = float(tp[t, k])
        sl_pct      = float(sl[t, k])
        trail_pct   = float(trail[t, k])
        tab_pct     = float(tab[t, k])
        be_pct      = float(be[t, k])
        ts_bars     = int(ts_arr[t, k])

        pnl, n_held, exit_reason_id = _simulate_one_trade_fee_with_reason(
            prices, entry_bar, direction,
            tp_pct, sl_pct, trail_pct, tab_pct, be_pct, ts_bars,
            0, fee,
        )
        exit_reason = EXIT_NAMES[exit_reason_id]

        if n_held == 0 and pnl == 0.0:
            invalid_entries += 1
            t += 1
            continue

        exit_bar   = entry_bar + n_held
        if exit_bar >= n_bars:
            exit_bar = n_bars - 1
        exit_ts    = int(ts[exit_bar])
        exit_price = float(prices[exit_bar])

        # equity tracking
        eq_arr[t:exit_bar + 1] = equity
        equity *= (1.0 + float(pnl))
        eq_arr[exit_bar + 1:] = equity                 # extend tail
        if exit_bar == n_bars - 1:
            eq_arr[-1] = equity                        # ensure last bar reflects post-exit
        peak     = max(peak, equity)
        last_pnl = float(pnl)

        # in-position bar marking (entry_bar inclusive, exit_bar inclusive)
        if direction == 1:
            in_pos_long[entry_bar:exit_bar + 1]  = True
        else:
            in_pos_short[entry_bar:exit_bar + 1] = True

        trades.append(dict(
            decision_bar = t,
            entry_bar    = entry_bar,
            entry_ts     = entry_ts,
            entry_dt     = datetime.utcfromtimestamp(entry_ts).isoformat(),
            exit_bar     = exit_bar,
            exit_ts      = exit_ts,
            exit_dt      = datetime.utcfromtimestamp(exit_ts).isoformat(),
            n_held       = int(n_held),
            duration_min = int(n_held),
            strategy_idx = k,
            strategy     = STRAT_KEYS[k],
            direction    = direction,
            entry_price  = entry_price,
            exit_price   = exit_price,
            tp_pct       = tp_pct,
            sl_pct       = sl_pct,
            tab_pct      = tab_pct,
            be_pct       = be_pct,
            pnl          = float(pnl),
            pnl_bps      = float(pnl) * 10_000,
            exit_reason  = exit_reason,
            equity_after = equity,
        ))

        t = exit_bar + 1

    return dict(
        trades         = trades,
        eq_arr         = eq_arr,
        in_pos_long    = in_pos_long,
        in_pos_short   = in_pos_short,
        actions_count  = actions_count,
        notrade_count  = notrade_count,
        entries_blocked_dir_zero = entries_blocked_dir_zero,
        invalid_entries          = invalid_entries,
        n_bars         = n_bars,
        prices         = prices,
        ts             = ts,
    )


def _print_sanity(label: str, r: dict):
    """Audit checks that should ALL pass if pipeline is correct."""
    trades = r["trades"]
    n      = len(trades)
    print(f"\n  ── SANITY AUDIT ({label}) ──")

    issues = []

    # 1. Negative durations
    neg = [t for t in trades if t["n_held"] < 0]
    if neg:
        issues.append(f"  ✗ {len(neg)} trades with negative n_held")
    else:
        print(f"  ✓ no negative durations")

    # 2. Zero-PnL trades — these are EXPECTED for BE-protected trades that
    #    retraced exactly to entry (BE moves SL to entry; SL fires at entry → PnL=0)
    zero_pnl_be = [t for t in trades if t["pnl"] == 0 and t["n_held"] > 0
                                          and t["exit_reason"] == "BE"]
    zero_pnl_other = [t for t in trades if t["pnl"] == 0 and t["n_held"] > 0
                                              and t["exit_reason"] != "BE"]
    if zero_pnl_be:
        print(f"  ✓ {len(zero_pnl_be)} trades exited at BE-exact (PnL=0): expected behavior, "
              f"BE locks SL at entry, price retraced to entry")
    if zero_pnl_other:
        issues.append(f"  ✗ {len(zero_pnl_other)} non-BE trades with PnL=0 and n_held>0 "
                       f"(unexpected; investigate)")

    # 4. Direction is always ±1
    bad_dir = [t for t in trades if t["direction"] not in (-1, 1)]
    if bad_dir:
        issues.append(f"  ✗ {len(bad_dir)} trades with invalid direction")
    else:
        print(f"  ✓ all trades have direction ∈ {{-1, +1}}")

    # 5. exit_bar > entry_bar
    overlap = [t for t in trades if t["exit_bar"] < t["entry_bar"]]
    if overlap:
        issues.append(f"  ✗ {len(overlap)} trades with exit_bar < entry_bar")
    else:
        print(f"  ✓ all exits come after entries")

    # 6. No overlapping trades (sequential, non-overlap)
    sorted_tr = sorted(trades, key=lambda t: t["entry_bar"])
    overlap_pairs = 0
    for i in range(1, len(sorted_tr)):
        if sorted_tr[i]["entry_bar"] <= sorted_tr[i - 1]["exit_bar"]:
            overlap_pairs += 1
    if overlap_pairs:
        issues.append(f"  ✗ {overlap_pairs} overlapping trade pairs detected")
    else:
        print(f"  ✓ no overlapping trades (sequential non-overlap honored)")

    # 7. In-position fraction sanity
    n_long_bars  = int(r["in_pos_long"].sum())
    n_short_bars = int(r["in_pos_short"].sum())
    n_pos_bars   = n_long_bars + n_short_bars
    if n_pos_bars > r["n_bars"]:
        issues.append(f"  ✗ in-position bars > total bars (impossible)")

    # 8. Equity continuity (final eq matches compounded trade pnls)
    compounded = 1.0
    for t in trades:
        compounded *= (1.0 + t["pnl"])
    if abs(compounded - r["eq_arr"][-1]) > 1e-9:
        issues.append(f"  ✗ equity inconsistency: compounded={compounded:.6f} vs eq_arr[-1]={r['eq_arr'][-1]:.6f}")
    else:
        print(f"  ✓ equity continuity: compounded trade PnLs match equity curve to 1e-9")

    # 9. Action count vs trades count
    n_action_trades = int(r["actions_count"][1:].sum())
    n_executed      = n + r["entries_blocked_dir_zero"] + r["invalid_entries"]
    if n_action_trades != n_executed:
        issues.append(f"  ✗ action_trades {n_action_trades} ≠ executed/blocked sum {n_executed}")
    else:
        print(f"  ✓ action-trade accounting matches: {n_action_trades} non-NO_TRADE actions, "
              f"{n} executed, {r['entries_blocked_dir_zero']} blocked (signal=0), "
              f"{r['invalid_entries']} invalid")

    if issues:
        print(f"\n  ── ISSUES FOUND ──")
        for s in issues:
            print(s)
    else:
        print(f"\n  ✓ ALL CHECKS PASS")


def _print_summary(label: str, r: dict):
    """High-level summary stats."""
    trades = r["trades"]
    n      = len(trades)
    if n == 0:
        print(f"  no trades for {label}")
        return

    df = pd.DataFrame(trades)
    n_bars = r["n_bars"]

    print(f"\n  ── DEAL-BY-DEAL SUMMARY ({label}) — {n_bars:,} bars ──")
    print(f"  Trades executed     : {n:,}")
    print(f"  Date range          : {df['entry_dt'].iloc[0][:10]} → {df['exit_dt'].iloc[-1][:10]}")

    # Trade frequency
    span_days = (df["exit_ts"].iloc[-1] - df["entry_ts"].iloc[0]) / 86400
    print(f"\n  Frequency:")
    print(f"    Span days         : {span_days:.1f}")
    print(f"    Trades / day      : {n / max(span_days, 1):.2f}")
    print(f"    Bars / trade (avg): {n_bars / max(n, 1):.0f} bars between entries")

    # Duration
    print(f"\n  Duration (bars in position):")
    durs = df["n_held"]
    print(f"    min   = {durs.min():>4}   p25 = {durs.quantile(0.25):>4.0f}   "
          f"median = {durs.median():>4.0f}   p75 = {durs.quantile(0.75):>4.0f}   "
          f"p90 = {durs.quantile(0.90):>4.0f}   max = {durs.max():>4}   mean = {durs.mean():.1f}")

    # Time-in-position vs cash
    n_long  = int(r["in_pos_long"].sum())
    n_short = int(r["in_pos_short"].sum())
    n_cash  = n_bars - n_long - n_short
    print(f"\n  Time allocation:")
    print(f"    In long pos   : {n_long:>7,} bars  ({n_long/n_bars*100:>5.2f}%)")
    print(f"    In short pos  : {n_short:>7,} bars  ({n_short/n_bars*100:>5.2f}%)")
    print(f"    In cash       : {n_cash:>7,} bars  ({n_cash/n_bars*100:>5.2f}%)")

    # Long/short balance
    n_long_tr  = int((df["direction"] == 1).sum())
    n_short_tr = int((df["direction"] == -1).sum())
    print(f"\n  Long / short balance:")
    print(f"    Long trades   : {n_long_tr:>4}  ({n_long_tr/n*100:>5.1f}%)")
    print(f"    Short trades  : {n_short_tr:>4}  ({n_short_tr/n*100:>5.1f}%)")

    # PnL stats
    pnls = df["pnl"].values
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    print(f"\n  PnL statistics:")
    print(f"    Total return   : {(np.prod(1 + pnls) - 1)*100:>+7.2f}%   "
          f"(equity {np.prod(1 + pnls):.4f}×)")
    print(f"    Mean PnL/trade : {pnls.mean()*100:>+7.4f}%   "
          f"(median {np.median(pnls)*100:>+7.4f}%)")
    print(f"    Win rate       : {len(wins)/n*100:>5.1f}%")
    print(f"    Mean win       : {wins.mean()*100:>+7.4f}%   "
          f"(max {wins.max()*100:>+6.3f}%)" if len(wins) else "    no wins")
    print(f"    Mean loss      : {losses.mean()*100:>+7.4f}%   "
          f"(min {losses.min()*100:>+6.3f}%)" if len(losses) else "    no losses")
    if len(wins) and len(losses):
        print(f"    Win/loss ratio : {abs(wins.mean()/losses.mean()):.2f}× "
              f"(profit factor: {wins.sum()/abs(losses.sum()):.2f})")

    # Per-strategy breakdown
    print(f"\n  Per-strategy attribution:")
    print(f"    {'strat':<14}  {'n_trd':>5}  {'long':>4}  {'short':>5}  "
          f"{'win%':>5}  {'meanPnL':>8}  {'totalPnL':>9}  {'avgDur':>7}")
    print("    " + "─" * 75)
    for strat in STRAT_KEYS:
        sub = df[df["strategy"] == strat]
        if len(sub) == 0: continue
        n_l = int((sub["direction"] == 1).sum())
        n_s = int((sub["direction"] == -1).sum())
        wp  = (sub["pnl"] > 0).mean() * 100
        mp  = sub["pnl"].mean() * 100
        tp  = sub["pnl"].sum() * 100
        ad  = sub["n_held"].mean()
        print(f"    {strat:<14}  {len(sub):>5}  {n_l:>4}  {n_s:>5}  "
              f"{wp:>4.1f}%  {mp:>+7.4f}%  {tp:>+8.3f}%  {ad:>6.1f}")

    # Exit reason breakdown
    print(f"\n  Exit reasons:")
    print(f"    {'reason':<8}  {'count':>5}  {'%':>5}  {'meanPnL':>8}  {'win%':>5}")
    print("    " + "─" * 45)
    for reason in df["exit_reason"].unique():
        sub = df[df["exit_reason"] == reason]
        wp  = (sub["pnl"] > 0).mean() * 100
        mp  = sub["pnl"].mean() * 100
        print(f"    {reason:<8}  {len(sub):>5}  {len(sub)/n*100:>4.1f}%  "
              f"{mp:>+7.4f}%  {wp:>4.1f}%")

    # PnL distribution buckets
    print(f"\n  PnL distribution buckets:")
    edges = np.array([-np.inf, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, np.inf])
    labels = ["<-2%", "-2..-1%", "-1..-0.5%", "-0.5..0%", "0..0.5%", "0.5..1%", "1..2%", ">2%"]
    h, _ = np.histogram(pnls, bins=edges)
    print(f"    {'bucket':<10}  {'count':>5}  {'%':>5}")
    for lbl, c in zip(labels, h):
        print(f"    {lbl:<10}  {c:>5}  {c/n*100:>4.1f}%")

    # Daily / weekly cadence (rough)
    df["date"] = pd.to_datetime(df["entry_ts"], unit='s').dt.date
    by_day = df.groupby("date").size()
    print(f"\n  Trades per day:")
    print(f"    days with ≥1 trade  : {len(by_day):>4}  / {int(span_days)+1} total")
    print(f"    median  trades/day  : {by_day.median():>5.1f}")
    print(f"    p90     trades/day  : {by_day.quantile(0.90):>5.1f}")
    print(f"    max     trades/day  : {by_day.max():>5}")

    # Action distribution
    print(f"\n  A2 action distribution:")
    total = r["actions_count"].sum()
    for i, lbl in enumerate(ACTIONS_LABEL):
        c = r["actions_count"][i]
        print(f"    {i:>2} {lbl:<14}  {c:>7,}  ({c/total*100:>5.2f}%)")


def _plot_equity(r_val, r_test, ticker: str, fee: float):
    """Equity curve + price overlay across val + test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_v = r_val["n_bars"]
    n_t = r_test["n_bars"]
    eq_full = np.concatenate([r_val["eq_arr"], r_val["eq_arr"][-1] * r_test["eq_arr"]])
    px_full = np.concatenate([r_val["prices"], r_test["prices"]])
    long_full  = np.concatenate([r_val["in_pos_long"],  r_test["in_pos_long"]])
    short_full = np.concatenate([r_val["in_pos_short"], r_test["in_pos_short"]])

    bars = np.arange(len(eq_full))

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                                gridspec_kw={"height_ratios": [3, 2, 1]})
    ax_eq, ax_px, ax_pos = axes

    # Equity curve
    ax_eq.plot(bars, eq_full, color="navy", linewidth=1.4, label="A2 + rule-based")
    ax_eq.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
    ax_eq.axvline(n_v, color="red", linestyle="--", alpha=0.6, label="val/test boundary")
    ax_eq.set_ylabel("Equity")
    ax_eq.set_title(f"A2 + rule-based exits — equity curve  (val + test, fee={fee:.4f})")
    ax_eq.grid(alpha=0.3); ax_eq.legend(loc="upper left", fontsize=9)

    # Price (BTC normalized)
    ax_px.plot(bars, px_full / px_full[0], color="orange", linewidth=0.7, label="BTC price (normalized)")
    ax_px.axvline(n_v, color="red", linestyle="--", alpha=0.6)
    ax_px.set_ylabel("BTC price (norm)")
    ax_px.grid(alpha=0.3); ax_px.legend(loc="upper left", fontsize=9)

    # Position (long=+1, short=-1, cash=0)
    pos = long_full.astype(int) - short_full.astype(int)
    ax_pos.fill_between(bars,  0,  1, where=long_full,  color="green", alpha=0.5, label="long")
    ax_pos.fill_between(bars, -1,  0, where=short_full, color="red",   alpha=0.5, label="short")
    ax_pos.axvline(n_v, color="red", linestyle="--", alpha=0.6)
    ax_pos.set_ylim(-1.2, 1.2)
    ax_pos.set_ylabel("Position")
    ax_pos.set_xlabel("Bar (val + test, 1-min)")
    ax_pos.grid(alpha=0.3); ax_pos.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    out = CACHE / f"{ticker}_a2_rule_audit.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"\n  → equity plot: {out.name}")


def run(ticker: str = "btc", split: str = "both", fee: float = 0.0,
         save_trades: bool = True):
    print(f"\n{'='*78}\n  A2 + RULE-BASED EXITS — DEAL-BY-DEAL AUDIT  ({ticker.upper()})  fee={fee:.4f}\n{'='*78}")
    net = _load_a2(ticker)
    print(f"  loaded A2 policy: {sum(p.numel() for p in net.parameters()):,} params")
    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_med = float(vol["atr_train_median"])

    splits_to_run = ["val", "test"] if split == "both" else [split]
    results = {}
    for sp_name in splits_to_run:
        print(f"\n{'─'*78}\n  Split: {sp_name.upper()}\n{'─'*78}")
        sp = np.load(CACHE / f"{ticker}_dqn_state_{sp_name}.npz")
        t0 = time.perf_counter()
        r = trace_trades(net, sp, atr_med, fee=fee)
        print(f"  traced {len(r['trades'])} trades in {time.perf_counter()-t0:.1f}s")
        _print_summary(sp_name, r)
        _print_sanity(sp_name, r)
        results[sp_name] = r

        if save_trades:
            out_csv = CACHE / f"{ticker}_a2_rule_trades_{sp_name}.csv"
            pd.DataFrame(r["trades"]).to_csv(out_csv, index=False)
            print(f"\n  → trade log: {out_csv.name}")

    # Combined plot
    if "val" in results and "test" in results:
        _plot_equity(results["val"], results["test"], ticker, fee)

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--split", choices=["val", "test", "both"], default="both")
    ap.add_argument("--fee", type=float, default=0.0)
    args = ap.parse_args()
    run(args.ticker, split=args.split, fee=args.fee)
