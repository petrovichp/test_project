"""
Test 5 — long-bias diagnostic + synthetic price-inversion stress test.

Two parts:

  Part A (default):  per-fold direction analysis using cached walkforward output
                     - Loads cache/btc_groupC2_walkforward_verify_baseline.json
                     - For each fold, reports BTC return, A2 fold Sharpe,
                       long-only PnL, short-only PnL
                     - Computes correlation between A2 fold Sharpe and BTC return
                     - Saves a scatter plot

  Part B (--invert-prices): synthetic price-inversion stress test
                     - Loads the full RL period; replaces price array with
                       2 * mean(prices) - prices (mirror around period mean)
                     - Re-runs A2 + rule-based eval per fold on the inverted
                       data. NO retraining — A2 sees the same state vector
                       (so its DECISIONS are the same), but the simulated
                       price path is inverted, revealing whether the entry
                       policy itself is direction-symmetric or long-biased.
                     - Caveat: because the state isn't inverted, this is a
                       lossy stress test. It tests whether trading on flipped
                       prices breaks down catastrophically.

Run:
  python3 -m models.test5_long_bias                       # Part A only
  python3 -m models.test5_long_bias --invert-prices       # Part A + Part B
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax
from models.dqn_rollout    import _build_exit_arrays, STRAT_KEYS
from models.dqn_selector   import evaluate_policy as evaluate_a2_rule_based
from models.group_c2_walkforward import (_load_a2_policy, _load_full_rl_period,
                                            _fold_boundaries,
                                            RL_START_REL)

CACHE = ROOT / "cache"


def _scatter_btc_vs_sharpe(rows, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    btc = np.array([r["btc_return"] for r in rows]) * 100
    shp = np.array([r["rule_sharpe"] for r in rows])
    # correlation
    if len(btc) > 1:
        corr = float(np.corrcoef(btc, shp)[0, 1])
    else:
        corr = float("nan")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(btc, shp, s=120, c="navy", edgecolors="white", linewidths=1.5, zorder=3)
    for r in rows:
        ax.annotate(f"f{r['fold']}", xy=(r["btc_return"]*100, r["rule_sharpe"]),
                     xytext=(8, 8), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("BTC buy-and-hold return (%)")
    ax.set_ylabel("A2 + rule-based Sharpe (annualized)")
    ax.set_title(f"Per-fold: BTC return vs A2 Sharpe   (corr = {corr:+.3f})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close()
    return corr


# ── Part A — diagnostics from cached baseline ────────────────────────────────

def part_a():
    src = CACHE / "results" / "btc_groupC2_walkforward_verify_baseline.json"
    if not src.exists():
        # fall back to the original walkforward output
        src = CACHE / "results" / "btc_groupC2_walkforward_fix240.json"
    print(f"  Loading cached walkforward: {src.name}")
    data = json.loads(src.read_text())
    rows = data["rows"]
    n = len(rows)

    print(f"\n{'='*100}\n  PART A — per-fold direction analysis  ({n} folds)\n{'='*100}\n")
    print(f"  {'fold':<5} {'date range':<26}  {'BTC ret':>8}  {'A2 Sharpe':>10}  "
          f"{'long PnL':>9}  {'short PnL':>10}  {'long n':>6}  {'short n':>7}  "
          f"{'long ratio':>10}")
    print("  " + "-" * 100)
    for r in rows:
        period = f"{r['date_a']}→{r['date_b'][5:]}"
        n_l = r.get("rule_n_long", 0)
        n_s = r.get("rule_n_short", 0)
        ratio = n_l / max(1, n_l + n_s)
        print(f"  fold{r['fold']:<2} {period:<26}  "
              f"{r['btc_return']*100:>+7.2f}%  "
              f"{r['rule_sharpe']:>+10.3f}  "
              f"{r.get('rule_long_pnl', 0)*100:>+8.2f}%  "
              f"{r.get('rule_short_pnl', 0)*100:>+9.2f}%  "
              f"{n_l:>6}  {n_s:>7}  {ratio*100:>9.1f}%")

    # Aggregate
    btc = np.array([r["btc_return"]    for r in rows])
    shp = np.array([r["rule_sharpe"]   for r in rows])
    long_pnl  = np.array([r.get("rule_long_pnl",  0) for r in rows])
    short_pnl = np.array([r.get("rule_short_pnl", 0) for r in rows])
    n_long    = np.array([r.get("rule_n_long",  0) for r in rows])
    n_short   = np.array([r.get("rule_n_short", 0) for r in rows])

    print(f"\n  Aggregate stats:")
    print(f"    BTC return       mean = {btc.mean()*100:+.2f}%   range [{btc.min()*100:+.2f}, {btc.max()*100:+.2f}]")
    print(f"    A2 fold Sharpe   mean = {shp.mean():+.3f}    range [{shp.min():+.2f}, {shp.max():+.2f}]")
    print(f"    Long PnL/fold    mean = {long_pnl.mean()*100:+.3f}%   total {long_pnl.sum()*100:+.2f}%")
    print(f"    Short PnL/fold   mean = {short_pnl.mean()*100:+.3f}%   total {short_pnl.sum()*100:+.2f}%")
    print(f"    Long trade ratio mean = {(n_long.sum()/(n_long.sum()+n_short.sum()))*100:.1f}%")
    print(f"    short_pnl / long_pnl = {short_pnl.sum() / max(1e-9, long_pnl.sum()):.2f}")

    # Correlations
    if len(btc) > 1:
        corr_sharpe = float(np.corrcoef(btc, shp)[0, 1])
        corr_long   = float(np.corrcoef(btc, long_pnl)[0, 1])
        corr_short  = float(np.corrcoef(btc, short_pnl)[0, 1])
        print(f"\n  Correlations with BTC fold return:")
        print(f"    A2 Sharpe       : {corr_sharpe:+.3f}   (positive → A2 does better when BTC goes up)")
        print(f"    Long PnL        : {corr_long:+.3f}   (positive → longs benefit from BTC up)")
        print(f"    Short PnL       : {corr_short:+.3f}   (negative expected → shorts benefit from BTC down)")

    out_png = CACHE / "plots" / "test5_btc_vs_sharpe.png"
    corr = _scatter_btc_vs_sharpe(rows, out_png)
    print(f"\n  → scatter plot saved: {out_png.name}")

    # Direction-symmetry verdict
    print(f"\n  ── DIRECTION-SYMMETRY VERDICT ──")
    if corr_sharpe > 0.7:
        print(f"  ✗ Strong long-bias dependence (corr {corr_sharpe:+.2f})")
    elif corr_sharpe > 0.3:
        print(f"  ◐ Moderate long-bias (corr {corr_sharpe:+.2f})")
    elif corr_sharpe > -0.3:
        print(f"  ✓ Direction-symmetric on aggregate (corr {corr_sharpe:+.2f})")
    else:
        print(f"  ✓ A2 even slightly counter-correlated with BTC moves (corr {corr_sharpe:+.2f}) — short alpha is real")

    if short_pnl.sum() > long_pnl.sum() * 0.7:
        print(f"  ✓ Short PnL ({short_pnl.sum()*100:+.1f}%) is at least 70% of long PnL "
              f"({long_pnl.sum()*100:+.1f}%) → balanced direction performance")

    return rows


# ── Part B — synthetic price-inversion stress test ───────────────────────────

def part_b():
    print(f"\n{'='*100}\n  PART B — synthetic price-inversion stress test\n{'='*100}\n")

    print(f"  Loading A2 entry policy + full RL period ...")
    entry_net = _load_a2_policy("btc")
    arr = _load_full_rl_period("btc")

    # Save original prices, build inverted version
    prices_orig = arr["price"].copy()
    period_mean = float(prices_orig.mean())
    prices_inv  = (2.0 * period_mean - prices_orig).astype(np.float32)
    print(f"  Period mean price: {period_mean:.2f}")
    print(f"  Original range  : [{prices_orig.min():.2f}, {prices_orig.max():.2f}]")
    print(f"  Inverted range  : [{prices_inv.min():.2f}, {prices_inv.max():.2f}]")
    print(f"  (Note: state vectors are NOT inverted — A2's decisions are unchanged;\n"
          f"   only the simulated trade outcomes use the inverted prices.)")

    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_med = float(vol["atr_train_median"])
    tp_full, sl_full, trail_full, tab_full, be_full, ts_full = _build_exit_arrays(
        prices_inv, arr["atr"], atr_med)

    folds = _fold_boundaries()
    rows = []
    for i, (a_pq, b_pq) in enumerate(folds):
        a = a_pq - RL_START_REL
        b = b_pq - RL_START_REL

        sub_state = arr["state"][a:b]
        sub_valid = arr["valid_actions"][a:b]
        sub_sigs  = arr["signals"][a:b]
        sub_pinv  = prices_inv[a:b]
        sub_porig = prices_orig[a:b]
        tp_f, sl_f, tr_f, tab_f, be_f, ts_f = (
            tp_full[a:b], sl_full[a:b], trail_full[a:b],
            tab_full[a:b], be_full[a:b], ts_full[a:b])

        date_a = datetime.fromtimestamp(int(arr["ts"][a])).strftime("%Y-%m-%d")
        date_b = datetime.fromtimestamp(int(arr["ts"][b-1])).strftime("%Y-%m-%d")

        # Original (control)
        res_orig = evaluate_a2_rule_based(
            entry_net, sub_state, sub_valid, sub_sigs, sub_porig,
            *_build_exit_arrays(sub_porig, arr["atr"][a:b], atr_med),
            fee=0.0)
        # Inverted
        res_inv = evaluate_a2_rule_based(
            entry_net, sub_state, sub_valid, sub_sigs, sub_pinv,
            tp_f, sl_f, tr_f, tab_f, be_f, ts_f, fee=0.0)

        # Long/short split for inverted
        inv_pnls = np.array(res_inv.get("trade_pnls", []), dtype=np.float64)
        inv_dirs = np.array(res_inv.get("trade_dirs",  []), dtype=np.int64)
        inv_long  = float(inv_pnls[inv_dirs ==  1].sum()) if len(inv_pnls) else 0.0
        inv_short = float(inv_pnls[inv_dirs == -1].sum()) if len(inv_pnls) else 0.0

        btc_orig = float(sub_porig[-1] / sub_porig[0] - 1.0)
        btc_inv  = float(sub_pinv[-1]  / sub_pinv[0]  - 1.0)

        print(f"\n  fold {i+1}  {date_a} → {date_b}")
        print(f"    BTC orig: {btc_orig*100:>+7.2f}%   inverted: {btc_inv*100:>+7.2f}%")
        print(f"    orig:     Sharpe {res_orig['sharpe']:>+7.3f}  eq {res_orig['equity_final']:.3f}  "
              f"trades {res_orig['n_trades']:>4}  win {res_orig['win_rate']*100:>4.1f}%")
        print(f"    inverted: Sharpe {res_inv['sharpe']:>+7.3f}  eq {res_inv['equity_final']:.3f}  "
              f"trades {res_inv['n_trades']:>4}  win {res_inv['win_rate']*100:>4.1f}%   "
              f"longPnL {inv_long*100:>+6.2f}%  shortPnL {inv_short*100:>+6.2f}%")
        print(f"    Δ Sharpe (inv − orig): {res_inv['sharpe']-res_orig['sharpe']:+.3f}")

        rows.append(dict(
            fold=i+1, date_a=date_a, date_b=date_b,
            btc_orig=btc_orig, btc_inv=btc_inv,
            orig_sharpe=res_orig["sharpe"], orig_eq=res_orig["equity_final"],
            orig_n=res_orig["n_trades"], orig_win=res_orig["win_rate"],
            inv_sharpe=res_inv["sharpe"], inv_eq=res_inv["equity_final"],
            inv_n=res_inv["n_trades"], inv_win=res_inv["win_rate"],
            inv_long_pnl=inv_long, inv_short_pnl=inv_short,
            delta_sharpe=res_inv["sharpe"] - res_orig["sharpe"],
        ))

    print(f"\n\n{'='*100}\n  PART B SUMMARY — original vs inverted prices\n{'='*100}")
    print(f"\n  {'fold':<5} {'BTC orig':>9} {'BTC inv':>9}   {'orig Sharpe':>11} {'inv Sharpe':>10}   {'Δ':>7}   {'inv long':>9} {'inv short':>10}")
    print("  " + "-" * 95)
    for r in rows:
        print(f"  fold{r['fold']:<2} {r['btc_orig']*100:>+8.2f}% {r['btc_inv']*100:>+8.2f}%   "
              f"{r['orig_sharpe']:>+11.3f} {r['inv_sharpe']:>+10.3f}   "
              f"{r['delta_sharpe']:>+7.3f}   "
              f"{r['inv_long_pnl']*100:>+8.2f}% {r['inv_short_pnl']*100:>+9.2f}%")

    orig_sharpes = np.array([r["orig_sharpe"] for r in rows])
    inv_sharpes  = np.array([r["inv_sharpe"]  for r in rows])
    print(f"\n  Aggregate:")
    print(f"    Original prices  : mean Sharpe {orig_sharpes.mean():+.3f}  median {np.median(orig_sharpes):+.3f}  "
          f"({(orig_sharpes>0).sum()}/6 folds positive)")
    print(f"    Inverted prices  : mean Sharpe {inv_sharpes.mean():+.3f}  median {np.median(inv_sharpes):+.3f}  "
          f"({(inv_sharpes>0).sum()}/6 folds positive)")
    print(f"    Δ                : {inv_sharpes.mean()-orig_sharpes.mean():+.3f}")

    print(f"\n  ── INVERSION VERDICT ──")
    if inv_sharpes.mean() > 5.0:
        print(f"  ✓ Strong: A2 still positive on inverted data → policy is direction-symmetric")
    elif inv_sharpes.mean() > 0:
        print(f"  ◐ Moderate: A2 still positive on inverted data, but lower")
    elif inv_sharpes.mean() > -5.0:
        print(f"  ⚠  A2 underperforms on inverted data — moderate long-bias dependence")
    else:
        print(f"  ✗ A2 collapses on inverted data — strong long-bias dependence")

    out = CACHE / "results" / "test5_inversion_results.json"
    out.write_text(json.dumps(dict(rows=rows,
        summary=dict(
            orig_mean=float(orig_sharpes.mean()),
            inv_mean=float(inv_sharpes.mean()),
            delta_mean=float(inv_sharpes.mean() - orig_sharpes.mean()),
        )), indent=2, default=str))
    print(f"\n  → {out.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--invert-prices", action="store_true",
                     help="run Part B (synthetic price-inversion stress test)")
    args = ap.parse_args()

    part_a()
    if args.invert_prices:
        part_b()
