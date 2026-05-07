"""
Plot equity curves: BASELINE_VOTE5 (K=5 plurality) vs BASELINE_FULL vs BTC B&H,
on DQN-val and DQN-test splits.
"""
import json, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from models.dqn_network    import DQN
from models.dqn_rollout    import _build_exit_arrays
from models.voting_ensemble import _VotePolicy, evaluate_with_policy, load_net

CACHE = pathlib.Path("cache")

VOTE5_SEEDS = [42, 7, 123, 0, 99]
TAG_FOR = {42: "BASELINE_FULL", 7: "BASELINE_FULL_seed7", 123: "BASELINE_FULL_seed123",
            0: "BASELINE_FULL_seed0", 99: "BASELINE_FULL_seed99"}


def _eval_with_curve(policy_fn, sp, atr_median):
    """Run a policy, return (eq_curve, sharpe, equity_final, n_trades)."""
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)

    # mirror evaluate_with_policy but capture eq_arr
    from models.diagnostics_ab import _simulate_one_trade_fee
    state = sp["state"]; valid = sp["valid_actions"]
    signals = sp["signals"]; prices = sp["price"]
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []; n_trades = 0

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
        trade_pnls.append(float(pnl)); n_trades += 1
        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return eq_arr, float(sharpe), float(equity), int(n_trades)


def main():
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    # build VOTE5 policy factory + load FULL net once
    vote5_nets = [load_net(TAG_FOR[s]) for s in VOTE5_SEEDS]
    full_net   = load_net("BASELINE_FULL")

    def make_vote5():
        return _VotePolicy(vote5_nets, mode="plurality")

    # _Greedy wrapper for the single net to use the same eval function
    class _SingleGreedy:
        def __init__(self, net): self.net = net
        def __call__(self, s, v):
            with torch.no_grad():
                from models.dqn_network import masked_argmax
                sb = torch.from_numpy(s).float().unsqueeze(0)
                vb = torch.from_numpy(v).bool().unsqueeze(0)
                return int(masked_argmax(self.net, sb, vb).item())
    def make_full():
        return _SingleGreedy(full_net)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    summary = []

    for ax, split in zip(axes, ["val", "test"]):
        sp = np.load(CACHE / f"btc_dqn_state_{split}.npz")
        eq_full, sh_full, eq_f, nt_f = _eval_with_curve(make_full, sp, atr_median)
        eq_vote, sh_vote, eq_v, nt_v = _eval_with_curve(make_vote5, sp, atr_median)
        bh = sp["price"] / sp["price"][0]

        dates = pd.to_datetime(sp["ts"], unit="s")

        ax.plot(dates, eq_full, color="#0a7", lw=1.4,
                label=f"BASELINE_FULL (Sharpe={sh_full:+.2f}, eq×{eq_f:.3f}, {nt_f} trades)")
        ax.plot(dates, eq_vote, color="#d24", lw=1.4,
                label=f"BASELINE_VOTE5 (Sharpe={sh_vote:+.2f}, eq×{eq_v:.3f}, {nt_v} trades)")
        ax.plot(dates, bh, color="#888", lw=1.0, ls="--",
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
            "full_sharpe": sh_full, "full_eq": eq_f, "full_trades": nt_f,
            "vote5_sharpe": sh_vote, "vote5_eq": eq_v, "vote5_trades": nt_v,
            "bh_eq": float(bh[-1]),
        })

    fig.suptitle("BASELINE_VOTE5 (K=5 plurality) vs BASELINE_FULL vs BTC B&H (fee=0)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = CACHE / "plot_vote5_vs_baseline_vs_bh.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")

    print(f"\n{'split':<6} {'FULL Sharpe':>12} {'VOTE5 Sharpe':>13} {'FULL eq':>8} "
          f"{'VOTE5 eq':>9} {'BH eq':>8} {'FULL trades':>12} {'VOTE5 trades':>13}")
    print("-"*100)
    for r in summary:
        print(f"{r['split']:<6} {r['full_sharpe']:>+12.3f} {r['vote5_sharpe']:>+13.3f} "
              f"{r['full_eq']:>8.3f} {r['vote5_eq']:>9.3f} {r['bh_eq']:>8.3f} "
              f"{r['full_trades']:>12} {r['vote5_trades']:>13}")

    (CACHE / "plot_vote5_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
