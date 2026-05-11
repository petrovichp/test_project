"""
Plot equity curves: VOTE5_v8_H256_DD (Step 4 winner) vs VOTE5_H256_DD (prior)
vs BASELINE_VOTE5 vs BTC buy & hold, on DQN-val and DQN-test.
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


def load_dueling(tag, hidden, state_dim, n_actions):
    net = DuelingDQN(state_dim, n_actions, hidden)
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

    # 3 ensembles + BH
    vote5_nets = [load_dqn("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS]
    h256_dd_nets = [load_dueling(f"VOTE5_H256_DD_seed{s}", hidden=256, state_dim=50, n_actions=10)
                       for s in SEEDS]
    v8_nets = [load_dueling(f"VOTE5_v8_H256_DD_seed{s}", hidden=256, state_dim=52, n_actions=12)
                  for s in SEEDS]

    def make_vote5(): return _VotePolicy(vote5_nets,   mode="plurality")
    def make_h256():  return _VotePolicy(h256_dd_nets, mode="plurality")
    def make_v8():    return _VotePolicy(v8_nets,      mode="plurality")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    summary = []

    for ax, split in zip(axes, ["val", "test"]):
        # Use v8 state cache (52-dim) for v8 — others use their own cache
        sp_v5 = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        sp_v8 = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")

        eq_v5, sh_v5, eq_5_, nt_5 = _eval_with_curve(make_vote5, sp_v5, atr_median)
        eq_h,  sh_h,  eq_h_, nt_h = _eval_with_curve(make_h256,  sp_v5, atr_median)
        eq_8,  sh_8,  eq_8_, nt_8 = _eval_with_curve(make_v8,    sp_v8, atr_median)
        bh = sp_v5["price"] / sp_v5["price"][0]
        dates = pd.to_datetime(sp_v5["ts"], unit="s")

        ax.plot(dates, eq_v5, color="#888", lw=1.0,
                label=f"BASELINE_VOTE5 (Sharpe={sh_v5:+.2f}, eq×{eq_5_:.3f}, {nt_5} tr)")
        ax.plot(dates, eq_h,  color="#0a7", lw=1.2,
                label=f"VOTE5_H256_DD (Sharpe={sh_h:+.2f}, eq×{eq_h_:.3f}, {nt_h} tr)")
        ax.plot(dates, eq_8,  color="#d24", lw=1.8,
                label=f"VOTE5_v8_H256_DD NEW (Sharpe={sh_8:+.2f}, eq×{eq_8_:.3f}, {nt_8} tr)")
        ax.plot(dates, bh, color="#aaa", lw=1.0, ls="--",
                label=f"BTC B&H (eq×{bh[-1]:.3f})")

        ax.set_title(f"DQN-{split}  ({len(sp_v5['price']):,} bars,  "
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
            "vote5_sharpe": sh_v5, "vote5_eq": eq_5_, "vote5_trades": nt_5,
            "h256_dd_sharpe": sh_h, "h256_dd_eq": eq_h_, "h256_dd_trades": nt_h,
            "v8_sharpe": sh_8, "v8_eq": eq_8_, "v8_trades": nt_8,
            "bh_eq": float(bh[-1]),
        })

    fig.suptitle("VOTE5_v8_H256_DD (NEW, Step 4) vs prior baselines  (fee=0)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = CACHE / "plots" / "plot_v8_vs_baselines.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")

    print(f"\n{'split':<6} {'VOTE5':>14} {'H256_DD':>15} {'v8 NEW':>15} {'BH':>8}")
    for r in summary:
        print(f"{r['split']:<6} "
              f"Sh={r['vote5_sharpe']:+5.2f}×{r['vote5_eq']:.2f}({r['vote5_trades']:>4}) "
              f"Sh={r['h256_dd_sharpe']:+5.2f}×{r['h256_dd_eq']:.2f}({r['h256_dd_trades']:>4}) "
              f"Sh={r['v8_sharpe']:+5.2f}×{r['v8_eq']:.2f}({r['v8_trades']:>4}) "
              f"×{r['bh_eq']:.2f}")

    (CACHE / "plots" / "plot_v8_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
