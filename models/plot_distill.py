"""
Plot equity curves for C2 distilled students vs VOTE5_v8 teacher on val + test.

Curves shown:
  - All 10 distilled single-net students (5 orig + 5 disjoint), thin lines
  - Best single (DISTILL_v8 seed=42), bold red
  - Best disjoint single (DISTILL_v8_DISJOINT seed=50), bold orange
  - Teacher VOTE5_v8_H256_DD ensemble, bold green
  - BTC buy & hold, dashed grey
"""
import json, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from models.dqn_network    import DuelingDQN
from models.dqn_rollout    import _build_exit_arrays
from models.voting_ensemble import _VotePolicy
from models.diagnostics_ab import _simulate_one_trade_fee

CACHE = pathlib.Path("cache")
ORIG_SEEDS     = [42, 7, 123, 0, 99]
DISJOINT_SEEDS = [1, 13, 25, 50, 77]


def load_net(tag):
    n = DuelingDQN(52, 12, 256)
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

    # all distilled student nets
    orig_nets = {s: load_net(f"DISTILL_v8_seed{s}") for s in ORIG_SEEDS}
    disj_nets = {s: load_net(f"DISTILL_v8_DISJOINT_seed{s}") for s in DISJOINT_SEEDS}
    teacher_nets = [load_net(f"VOTE5_v8_H256_DD_seed{s}") for s in ORIG_SEEDS]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    summary = []

    for ax, split in zip(axes, ["val", "test"]):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        dates = pd.to_datetime(sp["ts"], unit="s")

        # all 5 orig single students (thin grey-blue)
        for s, net in orig_nets.items():
            eq, sh, eq_f, nt = _eval_curve(lambda n=net: _VotePolicy([n], mode="plurality"),
                                            sp, atr_median)
            ax.plot(dates, eq, color="#6699cc", lw=0.7, alpha=0.5)
        # all 5 disjoint single students (thin grey-orange)
        for s, net in disj_nets.items():
            eq, sh, eq_f, nt = _eval_curve(lambda n=net: _VotePolicy([n], mode="plurality"),
                                            sp, atr_median)
            ax.plot(dates, eq, color="#cc9966", lw=0.7, alpha=0.5)

        # best single (orig seed=42)
        eq_42, sh_42, eq_42_f, nt_42 = _eval_curve(
            lambda: _VotePolicy([orig_nets[42]], mode="plurality"), sp, atr_median)
        ax.plot(dates, eq_42, color="#d22", lw=2.2,
                label=f"DISTILL_v8 single seed=42  (Sh={sh_42:+.2f}, eq×{eq_42_f:.3f}, {nt_42} tr)")

        # best single (disjoint seed=50)
        eq_50, sh_50, eq_50_f, nt_50 = _eval_curve(
            lambda: _VotePolicy([disj_nets[50]], mode="plurality"), sp, atr_median)
        ax.plot(dates, eq_50, color="#e80", lw=2.2,
                label=f"DISTILL_v8_DISJOINT single seed=50  (Sh={sh_50:+.2f}, eq×{eq_50_f:.3f}, {nt_50} tr)")

        # teacher VOTE5_v8
        eq_t, sh_t, eq_t_f, nt_t = _eval_curve(
            lambda: _VotePolicy(teacher_nets, mode="plurality"), sp, atr_median)
        ax.plot(dates, eq_t, color="#0a6", lw=2.2,
                label=f"TEACHER VOTE5_v8_H256_DD  (Sh={sh_t:+.2f}, eq×{eq_t_f:.3f}, {nt_t} tr)")

        # BH
        bh = sp["price"] / sp["price"][0]
        ax.plot(dates, bh, color="#888", lw=1.0, ls="--",
                label=f"BTC B&H  (eq×{bh[-1]:.3f})")

        # invisible spread legend marker
        ax.plot([], [], color="#6699cc", lw=2, alpha=0.5,
                label=f"5 orig single distilled (thin)")
        ax.plot([], [], color="#cc9966", lw=2, alpha=0.5,
                label=f"5 disjoint single distilled (thin)")

        ax.set_title(f"DQN-{split}  ({len(sp['price']):,} bars, "
                     f"{dates[0].date()} → {dates[-1].date()})", fontsize=11)
        ax.set_ylabel("Equity ×")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        for label in ax.get_xticklabels():
            label.set_rotation(30); label.set_ha("right")

        summary.append({
            "split": split,
            "best_orig_s42_sharpe": sh_42, "best_orig_s42_eq": eq_42_f, "best_orig_s42_trades": nt_42,
            "best_disj_s50_sharpe": sh_50, "best_disj_s50_eq": eq_50_f, "best_disj_s50_trades": nt_50,
            "teacher_sharpe": sh_t, "teacher_eq": eq_t_f, "teacher_trades": nt_t,
            "bh_eq": float(bh[-1]),
        })

    fig.suptitle("C2 distilled students vs VOTE5_v8 teacher  (fee=0, masked-CE distillation)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = CACHE / "plots" / "plot_distill_v8.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")

    print(f"\n{'split':<5} {'s=42 orig':>22} {'s=50 disjoint':>22} {'teacher':>22} {'BH':>7}")
    for r in summary:
        print(f"{r['split']:<5} "
              f"Sh={r['best_orig_s42_sharpe']:+5.2f}×{r['best_orig_s42_eq']:.2f}({r['best_orig_s42_trades']:>3})   "
              f"Sh={r['best_disj_s50_sharpe']:+5.2f}×{r['best_disj_s50_eq']:.2f}({r['best_disj_s50_trades']:>3})   "
              f"Sh={r['teacher_sharpe']:+5.2f}×{r['teacher_eq']:.2f}({r['teacher_trades']:>3})  "
              f"×{r['bh_eq']:.2f}")

    (CACHE / "plots" / "plot_distill_v8_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
