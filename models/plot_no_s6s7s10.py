"""
Plot equity curves: A2 baseline vs A2_no_s6_s7_s10 vs BTC buy-and-hold,
on DQN-val and DQN-test splits (the splits A2 was selected/evaluated on).
"""
import json, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from models.dqn_network  import DQN
from models.dqn_selector import evaluate_policy
from models.dqn_rollout  import _build_exit_arrays

CACHE = pathlib.Path("cache")


def _load_policy(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def _build_mask(ablate: list[int]) -> np.ndarray | None:
    if not ablate:
        return None
    m = np.ones(10, dtype=np.bool_)
    for i in ablate:
        m[i] = False
    m[0] = True
    return m


def run_split(split: str, policy_tag: str, ablate: list[int], atr_median: float):
    sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
    tp, sl, tr, tab, be, ts_arr = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    net = _load_policy(policy_tag)
    out = evaluate_policy(
        net, sp["state"], sp["valid_actions"], sp["signals"], sp["price"],
        tp, sl, tr, tab, be, ts_arr,
        valid_mask_override=_build_mask(ablate),
        fee=0.0,
    )
    return sp["ts"], sp["price"], out


def main():
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)
    splits = ["val", "test"]

    summary_rows = []
    for ax, split in zip(axes, splits):
        # baseline A2
        ts_b, price, base = run_split(split, "A2", [], atr_median)
        # triple ablation
        _, _, abl = run_split(split, "A2_no_s6_s7_s10", [5, 6, 8], atr_median)

        # BTC buy-and-hold equity (long, no fee, full split)
        bh = price / price[0]

        # convert ns timestamps
        dates = pd.to_datetime(ts_b)

        ax.plot(dates, base["eq_curve"], color="#0a7", lw=1.4,
                label=f"A2 baseline (Sharpe={base['sharpe']:+.2f}, "
                      f"eq×{base['equity_final']:.3f}, {base['n_trades']} trades)")
        ax.plot(dates, abl["eq_curve"], color="#d24", lw=1.4,
                label=f"A2 no_s6+s7+s10 (Sharpe={abl['sharpe']:+.2f}, "
                      f"eq×{abl['equity_final']:.3f}, {abl['n_trades']} trades)")
        ax.plot(dates, bh, color="#888", lw=1.0, ls="--",
                label=f"BTC buy & hold (eq×{bh[-1]:.3f})")

        ax.set_title(f"DQN-{split}  ({len(price):,} bars,  "
                     f"{dates[0].date()} → {dates[-1].date()})", fontsize=11)
        ax.set_ylabel("Equity ×")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

        summary_rows.append({
            "split":  split,
            "base_sharpe":   base["sharpe"],
            "base_eq":       base["equity_final"],
            "base_dd":       base["max_dd"],
            "base_trades":   base["n_trades"],
            "abl_sharpe":    abl["sharpe"],
            "abl_eq":        abl["equity_final"],
            "abl_dd":        abl["max_dd"],
            "abl_trades":    abl["n_trades"],
            "bh_eq":         float(bh[-1]),
        })

    fig.suptitle("A2 baseline vs A2 no_s6+s7+s10 vs BTC buy-and-hold (fee=0)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out_png = CACHE / "plots" / "plot_no_s6s7s10_vs_baseline_vs_bh.png"
    fig.savefig(out_png, dpi=130)
    print(f"\n  → {out_png}")

    # print summary
    print()
    print(f"{'split':<6} {'base Sharpe':>12} {'abl Sharpe':>12} {'base eq':>9} "
          f"{'abl eq':>9} {'BH eq':>9} {'base trades':>12} {'abl trades':>11}")
    print("-"*95)
    for r in summary_rows:
        print(f"{r['split']:<6} {r['base_sharpe']:>+12.3f} {r['abl_sharpe']:>+12.3f} "
              f"{r['base_eq']:>9.3f} {r['abl_eq']:>9.3f} {r['bh_eq']:>9.3f} "
              f"{r['base_trades']:>12} {r['abl_trades']:>11}")

    (CACHE / "plots" / "plot_no_s6s7s10_summary.json").write_text(
        json.dumps(summary_rows, indent=2, default=str))


if __name__ == "__main__":
    main()
