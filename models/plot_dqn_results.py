"""
Plot BTC price vs DQN policy equity curves over DQN-val + DQN-test (OOS portion).

Shows three policies from Group A:
  A0  taker fee 0.0008 + no penalty  (Phase 3 baseline — failure case)
  A2  fee=0           + 0.1% penalty (best overall — Sharpe +7.30 on val)
  A4  maker fee 0.0004 + no penalty  (deployable target — Sharpe +1.72 on val)

Plus buy-and-hold reference.

Period: bars [DQN-val_start, end) = Feb 12 2026 → Apr 25 2026
        (~103k 1-min bars, OOS for the DQN training)

Run: python3 -m models.plot_dqn_results [ticker]
"""

import sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax
from models.dqn_rollout    import _build_exit_arrays
from models.diagnostics_ab import _simulate_one_trade_fee
from numba import njit

CACHE = ROOT / "cache"


# ── per-bar equity from DQN policy (greedy) ─────────────────────────────────

@njit(cache=True)
def _dqn_equity_curve(actions_per_step, action_bars,
                       signals_strat, prices,
                       tp, sl, trail, tab, be, ts_bars, fee):
    """Walk decision points (from action_bars) and simulate trades using
    pre-decided actions. Returns per-bar equity curve (length len(prices)).

    actions_per_step : (M,) int8     — chosen action at each decision point
    action_bars      : (M,) int64    — bar index at each decision point
    """
    n_bars = len(prices)
    equity = np.ones(n_bars, dtype=np.float64)
    eq = 1.0
    last_filled = 0
    M = len(actions_per_step)

    for j in range(M):
        t = action_bars[j]
        if t < 0 or t >= n_bars - 1:
            continue
        # fill flat from last_filled..t
        for i in range(last_filled, t + 1):
            equity[i] = eq

        a = actions_per_step[j]
        if a == 0:
            last_filled = t + 1
            if last_filled < n_bars:
                equity[last_filled] = eq
            continue

        k = a - 1
        direction = signals_strat[t, k]
        if direction == 0:
            last_filled = t + 1
            continue

        pnl, n_held = _simulate_one_trade_fee(
            prices, t + 1, int(direction),
            float(tp[t, k]), float(sl[t, k]),
            float(trail[t, k]), float(tab[t, k]),
            float(be[t, k]), int(ts_bars[t, k]),
            0, fee,
        )
        close_bar = min(t + 1 + n_held, n_bars - 1)
        for i in range(t, close_bar + 1):
            equity[i] = eq
        eq *= (1.0 + pnl)
        if close_bar + 1 < n_bars:
            for i in range(close_bar + 1, n_bars):
                equity[i] = eq
        last_filled = close_bar + 1

    if last_filled < n_bars:
        for i in range(last_filled, n_bars):
            equity[i] = eq
    return equity


def run_policy_through_bars(net: DQN, state, valid, fee: float):
    """Greedy policy over `state`, returns (actions_per_step, action_bars).
    Skips through trade durations (no decision until trade closes)."""
    n = len(state)
    actions = np.zeros(n, dtype=np.int8)
    bars    = np.zeros(n, dtype=np.int64)
    cnt     = 0

    equity = 1.0; peak = 1.0; last_pnl = 0.0
    t = 0
    while t < n - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))

        sb = torch.from_numpy(s_t[None, :]).float()
        vb = torch.from_numpy(valid[t][None, :]).bool()
        a  = int(masked_argmax(net, sb, vb).item())

        actions[cnt] = a
        bars[cnt]    = t
        cnt += 1

        if a == 0:
            t += 1
        else:
            # need to advance past trade duration; we don't know it here without
            # executing — store decision and let caller sim & advance.
            # For simplicity: assume trade lasts ~average. We'll instead build
            # actions for ALL bars and let _dqn_equity_curve handle the advance.
            # So just advance 1 bar; non-overlapping enforced by simulation logic.
            # NOTE: simpler to rebuild full action array (one per bar) and have
            # the simulator handle skipping.
            t += 1
    return actions[:cnt], bars[:cnt]


def run_policy_with_skip(net: DQN, state, valid, signals_strat, prices,
                           tp, sl, trail, tab, be, ts_bars, fee: float):
    """Walk bars, when DQN selects a trade action, simulate it and skip past
    its duration. Returns (actions_per_step, action_bars) where bars are the
    DECISION bars (post-skip)."""
    n_bars = len(state)
    actions = []
    bars    = []

    equity = 1.0; peak = 1.0; last_pnl = 0.0
    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))

        sb = torch.from_numpy(s_t[None, :]).float()
        vb = torch.from_numpy(valid[t][None, :]).bool()
        a  = int(masked_argmax(net, sb, vb).item())

        actions.append(a)
        bars.append(t)

        if a == 0:
            t += 1
        else:
            k = a - 1
            direction = int(signals_strat[t, k])
            if direction == 0:
                t += 1
                continue
            pnl, n_held = _simulate_one_trade_fee(
                prices, t + 1, direction,
                float(tp[t, k]), float(sl[t, k]),
                float(trail[t, k]), float(tab[t, k]),
                float(be[t, k]), int(ts_bars[t, k]),
                0, fee,
            )
            equity   *= (1.0 + float(pnl))
            peak      = max(peak, equity)
            last_pnl  = float(pnl)
            t = t + 1 + n_held + 1
    return (np.array(actions, dtype=np.int8),
             np.array(bars,    dtype=np.int64))


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n  Loading DQN policies + state arrays ...")

    # ── concat DQN-val + DQN-test for OOS plotting ──────────────────────────
    sp_v = np.load(CACHE / "state" / f"{ticker}_dqn_state_val.npz")
    sp_t = np.load(CACHE / "state" / f"{ticker}_dqn_state_test.npz")
    state   = np.concatenate([sp_v["state"],         sp_t["state"]])
    valid   = np.concatenate([sp_v["valid_actions"], sp_t["valid_actions"]])
    signals = np.concatenate([sp_v["signals"],       sp_t["signals"]])
    prices  = np.concatenate([sp_v["price"],         sp_t["price"]]).astype(np.float64)
    atr     = np.concatenate([sp_v["atr"],           sp_t["atr"]])
    ts      = np.concatenate([sp_v["ts"],            sp_t["ts"]])
    dt      = pd.to_datetime(ts, unit="s")
    print(f"  combined val+test: {len(state):,} bars  "
          f"({dt[0].strftime('%Y-%m-%d')} → {dt[-1].strftime('%Y-%m-%d')})")

    # ── exit arrays (precompute once) ───────────────────────────────────────
    vol = np.load(CACHE / "preds" / f"{ticker}_pred_vol_v4.npz")
    atr_med = float(vol["atr_train_median"])
    tp, sl, trail, tab, be, ts_bars = _build_exit_arrays(prices, atr, atr_med)

    # jit warmup
    _ = _simulate_one_trade_fee(prices[:20], 5, 1, 0.02, 0.005, 0.0, 0.0, 0.005, 0, 0, 0.0008)
    _ = _dqn_equity_curve(
        np.zeros(2, dtype=np.int8), np.array([0, 1], dtype=np.int64),
        signals[:20], prices[:20],
        tp[:20], sl[:20], trail[:20], tab[:20], be[:20], ts_bars[:20], 0.0008,
    )

    # ── run each policy ──────────────────────────────────────────────────────
    cells = [
        # (id, fee, color, label)
        ("A0", 0.0008, "#d62728",
         "A0 — DQN at taker fee 0.0008  (Phase 3 baseline — failure)"),
        ("A4", 0.0004, "#ff7f0e",
         "A4 — DQN at maker fee 0.0004  (deployable target)"),
        ("A2", 0.0000, "#2ca02c",
         "A2 — DQN at fee 0 + penalty 0.1%  (best — fee-free reference)"),
    ]

    curves = {}
    for cid, fee, color, label in cells:
        path = CACHE / "policies" / f"{ticker}_dqn_policy_{cid}.pt"
        if not path.exists():
            print(f"  ✗ missing {path.name}, skipping {cid}")
            continue
        print(f"  loading {cid} ...")
        net = DQN(state_dim=50, n_actions=10, hidden=64)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()

        actions, bars = run_policy_with_skip(
            net, state, valid, signals, prices,
            tp, sl, trail, tab, be, ts_bars, fee,
        )
        eq = _dqn_equity_curve(
            actions, bars, signals, prices,
            tp, sl, trail, tab, be, ts_bars, fee,
        )
        n_trades = int((actions != 0).sum())
        print(f"    {cid}: {n_trades:>5,} trades   final equity = {eq[-1]:.3f}")
        curves[cid] = dict(eq=eq, color=color, label=label, n_trades=n_trades)

    bh = prices / prices[0]

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.07},
    )

    # top: BTC price
    ax1.plot(dt, prices, color="#1f77b4", linewidth=0.6, label="BTC perp_ask_price")
    ax1.set_ylabel("BTC USDT", fontsize=11)
    ax1.set_title(f"BTC perpetual price vs DQN policy equity — Group A  "
                   f"({dt[0].strftime('%Y-%m-%d')} → {dt[-1].strftime('%Y-%m-%d')})  "
                   f"OOS portion only (DQN-val + DQN-test)",
                   fontsize=13, pad=12)
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # split val/test in chart with vertical line
    n_val = len(sp_v["state"])
    split_dt = dt[n_val]
    for ax in (ax1, ax2):
        ax.axvline(split_dt, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax2.text(split_dt, 0.98, "  val │ test  ", transform=ax2.get_xaxis_transform(),
              fontsize=8, ha="left", va="top", color="#444",
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    # bottom: equity curves
    ax2.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax2.plot(dt, bh, color="#1f77b4", linewidth=1.0, alpha=0.6,
             linestyle="-.",
             label=f"BTC buy-and-hold                                    →  final {bh[-1]:.2f}×")
    for cid, cdata in curves.items():
        ax2.plot(dt, cdata["eq"], color=cdata["color"], linewidth=1.5, alpha=0.95,
                  label=f"{cdata['label']}  →  final {cdata['eq'][-1]:.2f}× ({cdata['n_trades']:,} trades)")
    ax2.set_ylabel("Equity (start = 1.00)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, linestyle=":", which="both")
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}×"))

    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0, ha="center")

    fig.text(0.5, 0.02,
              f"DQN policies trained in Group A sweep (Phase 3.3, fee × penalty grid). "
              f"Equity curves shown on the OOS slice only — DQN-val (left of dashed line) + DQN-test (right). "
              f"DQN-train portion (Sep 2025–Feb 2026) excluded since it would be in-sample.",
              ha="center", fontsize=8, style="italic", color="#444")

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out = CACHE / "plots" / f"{ticker}_dqn_groupA_equity_vs_price.png"
    plt.savefig(str(out), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n  → {out}")
    print(f"  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
