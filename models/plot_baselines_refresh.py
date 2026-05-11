"""
Refresh baseline plot — show the current lineup post C2 + Z2.1:
  - VOTE5_v8_H256_DD (primary, max WF)
  - DISTILL_v8_seed42 (cheap-deployment alt, max test)
  - VOTE5_H256_DD (prior primary, best fold-6)
  - BASELINE_VOTE5 (h=64 reference)
  - BTC B&H
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
    n = DQN(50, 10, hidden)
    n.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    n.eval(); return n


def load_dueling(tag, hidden, state_dim, n_actions):
    n = DuelingDQN(state_dim, n_actions, hidden)
    n.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    n.eval(); return n


def _eval_curve(policy_fn, sp, atr_median):
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

    vote5     = [load_dqn("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS]
    h256_dd   = [load_dueling(f"VOTE5_H256_DD_seed{s}", 256, 50, 10)  for s in SEEDS]
    v8        = [load_dueling(f"VOTE5_v8_H256_DD_seed{s}",   256, 52, 12) for s in SEEDS]
    distill42 = load_dueling("DISTILL_v8_seed42", 256, 52, 12)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    summary = []

    for ax, split in zip(axes, ["val", "test"]):
        sp_v5 = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        sp_v8 = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        dates = pd.to_datetime(sp_v5["ts"], unit="s")

        # plot order: faded prior baselines first, headliners on top
        eq, sh, eqf, nt = _eval_curve(lambda: _VotePolicy(vote5, mode="plurality"), sp_v5, atr_median)
        ax.plot(dates, eq, color="#999", lw=0.9, alpha=0.7,
                label=f"BASELINE_VOTE5 h=64 (Sh={sh:+.2f}, eq×{eqf:.2f}, {nt} tr)")

        eq, sh, eqf, nt = _eval_curve(lambda: _VotePolicy(h256_dd, mode="plurality"), sp_v5, atr_median)
        ax.plot(dates, eq, color="#0a7", lw=1.5,
                label=f"VOTE5_H256_DD (Sh={sh:+.2f}, eq×{eqf:.2f}, {nt} tr)")

        eq, sh, eqf, nt = _eval_curve(lambda: _VotePolicy(v8, mode="plurality"), sp_v8, atr_median)
        ax.plot(dates, eq, color="#06f", lw=2.2,
                label=f"⭐ VOTE5_v8_H256_DD primary (Sh={sh:+.2f}, eq×{eqf:.2f}, {nt} tr)")

        eq, sh, eqf, nt = _eval_curve(lambda: _VotePolicy([distill42], mode="plurality"), sp_v8, atr_median)
        ax.plot(dates, eq, color="#d22", lw=2.2,
                label=f"⭐ DISTILL_v8_seed42 single-net (Sh={sh:+.2f}, eq×{eqf:.2f}, {nt} tr)")

        bh = sp_v5["price"] / sp_v5["price"][0]
        ax.plot(dates, bh, color="#aaa", lw=1.0, ls="--",
                label=f"BTC B&H (eq×{bh[-1]:.2f})")

        ax.set_title(f"DQN-{split}  ({len(sp_v5['price']):,} bars, "
                     f"{dates[0].date()} → {dates[-1].date()})", fontsize=11)
        ax.set_ylabel("Equity ×")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        for label in ax.get_xticklabels():
            label.set_rotation(30); label.set_ha("right")

    fig.suptitle("Refreshed baselines — 2026-05-11 (post C2 + Z2.1)  fee=0",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = CACHE / "plots" / "plot_baselines_refresh.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
