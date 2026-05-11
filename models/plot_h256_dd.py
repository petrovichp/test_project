"""
Plot equity curves: VOTE5_H256_DD (Z1.1 winner) vs BASELINE_VOTE5
vs BTC buy & hold, on DQN-val and DQN-test splits.
"""
import json, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from models.dqn_network    import DQN, DuelingDQN
from models.dqn_rollout    import _build_exit_arrays
from models.voting_ensemble import _VotePolicy
from models.diagnostics_ab import _simulate_one_trade_fee

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]


def load_dqn(tag, hidden=64):
    net = DQN(50, 10, hidden)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def load_dueling(tag, hidden=64):
    net = DuelingDQN(50, 10, hidden)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def _eval_with_curve(policy_fn, sp, atr_median):
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    state = sp["state"]; valid = sp["valid_actions"]
    signals = sp["signals"]; prices = sp["price"]
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    n_trades = 0

    pol = policy_fn()
    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False
        action = pol(s_t, valid_t)
        if action == 0:
            t += 1; continue
        k = action - 1
        direction = int(signals[t, k])
        if direction == 0:
            t += 1; continue
        pnl, n_held = _simulate_one_trade_fee(
            prices, t + 1, direction,
            float(tp[t, k]), float(sl[t, k]),
            float(tr[t, k]), float(tab[t, k]),
            float(be[t, k]),   int(ts[t, k]),
            0, 0.0,
        )
        t_close = t + 1 + n_held
        if t_close >= n_bars: t_close = n_bars - 1
        eq_arr[t:t_close + 1] = equity
        equity *= (1.0 + float(pnl))
        eq_arr[t_close + 1:] = equity
        if t_close == n_bars - 1: eq_arr[-1] = equity
        peak = max(peak, equity); last_pnl = float(pnl)
        n_trades += 1
        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return eq_arr, float(sharpe), float(equity), int(n_trades)


def main():
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    # baseline VOTE5 (vanilla, h=64) — old champion
    vote5_nets = [load_dqn("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}")
                   for s in SEEDS]
    # NEW: VOTE5_H256_DD — Z1.1 winner
    h256_dd_nets = [load_dueling(f"VOTE5_H256_DD_seed{s}", hidden=256) for s in SEEDS]

    def make_vote5():    return _VotePolicy(vote5_nets,    mode="plurality")
    def make_h256_dd():  return _VotePolicy(h256_dd_nets,  mode="plurality")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    summary = []

    for ax, split in zip(axes, ["val", "test"]):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        eq_v5, sh_v5, eq_5, nt_5 = _eval_with_curve(make_vote5,   sp, atr_median)
        eq_h,  sh_h,  eq_h_,nt_h = _eval_with_curve(make_h256_dd, sp, atr_median)
        bh = sp["price"] / sp["price"][0]

        dates = pd.to_datetime(sp["ts"], unit="s")

        ax.plot(dates, eq_v5, color="#888", lw=1.2,
                label=f"BASELINE_VOTE5 (Sharpe={sh_v5:+.2f}, eq×{eq_5:.3f}, {nt_5} trades)")
        ax.plot(dates, eq_h, color="#0a7", lw=1.6,
                label=f"⭐ VOTE5_H256_DD (Sharpe={sh_h:+.2f}, eq×{eq_h_:.3f}, {nt_h} trades)")
        ax.plot(dates, bh, color="#aaa", lw=1.0, ls="--",
                label=f"BTC buy & hold (eq×{bh[-1]:.3f})")

        ax.set_title(f"DQN-{split}  ({len(sp['price']):,} bars,  "
                     f"{dates[0].date()} → {dates[-1].date()})", fontsize=11)
        ax.set_ylabel("Equity ×")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        for label in ax.get_xticklabels():
            label.set_rotation(30); label.set_ha("right")

        summary.append({
            "split": split,
            "vote5_sharpe": sh_v5, "vote5_eq": eq_5, "vote5_trades": nt_5,
            "h256_dd_sharpe": sh_h, "h256_dd_eq": eq_h_, "h256_dd_trades": nt_h,
            "bh_eq": float(bh[-1]),
        })

    fig.suptitle("VOTE5_H256_DD (Z1.1 winner) vs BASELINE_VOTE5 vs BTC B&H  (fee=0)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = CACHE / "plots" / "plot_h256_dd_vs_vote5.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")

    print(f"\n{'split':<6} {'VOTE5 Sharpe':>13} {'H256_DD Sharpe':>15} "
          f"{'VOTE5 eq':>9} {'H256_DD eq':>11} {'BH eq':>8} "
          f"{'VOTE5 trd':>10} {'H256_DD trd':>12}")
    print("-"*120)
    for r in summary:
        print(f"{r['split']:<6} {r['vote5_sharpe']:>+13.3f} {r['h256_dd_sharpe']:>+15.3f} "
              f"{r['vote5_eq']:>9.3f} {r['h256_dd_eq']:>11.3f} {r['bh_eq']:>8.3f} "
              f"{r['vote5_trades']:>10} {r['h256_dd_trades']:>12}")

    (CACHE / "plots" / "plot_h256_dd_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
