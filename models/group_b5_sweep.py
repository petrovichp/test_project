"""
Group B5 — fixed-window exit DQN sweep across (window N × per-strategy).

For each strategy k = 0..8 and each window N ∈ {60, 120, 240}, train one
FixedExitDQN at fee=0. Compare best val Sharpe to the always-HOLD-to-N
baseline (no rule-based exits during training).

Run:
  python3 -m models.group_b5_sweep                # all 27 cells
  python3 -m models.group_b5_sweep --only-n 120   # only N=120 (9 cells)
"""

import sys, time, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache"

STRAT_KEYS = ["S1_VolDir", "S2_Funding", "S3_BBRevert", "S4_MACDTrend",
               "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow",
               "S10_Squeeze", "S12_VWAPVol"]
WINDOWS = [60, 120, 240]


def run_cell(N: int, k: int, ticker: str = "btc", seed: int = 42,
              fee: float = 0.0) -> dict:
    tag = f"B5_fix{N}_fee0_S{k}"
    print(f"\n{'#'*78}\n#  CELL {tag}: N={N}  strat={STRAT_KEYS[k]}  fee={fee:.4f}\n{'#'*78}")

    cmd = [
        "python3", "-m", "models.exit_dqn_fixed", ticker,
        "--tag", tag, "--seed", str(seed), "--fee", str(fee),
        "--strat-filter", str(k), "--window-n", str(N),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  ✗ cell {tag} FAILED")
        print(res.stdout[-2000:]); print(res.stderr[-2000:])
        return dict(tag=tag, N=N, k=k, status="failed", elapsed=elapsed)

    for line in res.stdout.splitlines():
        if any(s in line for s in ["TRAINING SUMMARY", "best val Sharpe",
                                      "rule-only baseline", "Δ vs baseline",
                                      "Sharpe ", "Early stop"]):
            print("    " + line)

    hist_path = CACHE / f"{ticker}_exit_dqn_fixed_history_{tag}.json"
    if not hist_path.exists():
        return dict(tag=tag, N=N, k=k, status="no_history", elapsed=elapsed)
    h = json.loads(hist_path.read_text())
    history = h.get("history", [])
    if not history:
        return dict(tag=tag, N=N, k=k, status="empty_history", elapsed=elapsed)

    best_event = next((e for e in history if e["step"] == h["best_step"]), None)
    if best_event is None:
        best_event = max(history, key=lambda e: e["val_sharpe"])
    base = h["baseline"]
    return dict(
        tag=tag, N=N, k=k, status="ok", elapsed=elapsed,
        strat=STRAT_KEYS[k],
        best_step       = h["best_step"],
        best_val_sharpe = h["best_val_sharpe"],
        best_val_trades = best_event["val_trades"],
        best_val_winrate= best_event["val_winrate"],
        best_val_equity = best_event["val_equity"],
        best_val_max_dd = best_event["val_max_dd"],
        rl_exit_pct     = best_event["rl_exit_pct"],
        exit_breakdown  = best_event["exit_breakdown"],
        baseline_sharpe = base["sharpe"],
        baseline_trades = base["n_trades"],
        baseline_winrate= base["win_rate"],
        baseline_equity = base["equity_final"],
        delta_sharpe    = h["best_val_sharpe"] - base["sharpe"],
    )


def run(ticker: str = "btc", seed: int = 42, only_n: int = None,
         only_strat: int = None):
    print(f"\n{'='*78}\n  GROUP B5 — FIXED-WINDOW EXIT DQN SWEEP  ({ticker.upper()})\n{'='*78}")
    cells = [(N, k) for N in WINDOWS for k in range(len(STRAT_KEYS))]
    if only_n is not None:
        cells = [(N, k) for (N, k) in cells if N == only_n]
    if only_strat is not None:
        cells = [(N, k) for (N, k) in cells if k == only_strat]
    print(f"  Total cells: {len(cells)}")

    rows = []
    t_total = time.perf_counter()
    for N, k in cells:
        rows.append(run_cell(N, k, ticker, seed))
    total_elapsed = time.perf_counter() - t_total

    out_json = CACHE / f"{ticker}_exit_dqn_fixed_groupB5_summary.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))

    # ── result table grouped by window ──────────────────────────────────────
    print(f"\n\n{'='*100}\n  GROUP B5 — RESULT TABLE\n{'='*100}")
    for N in sorted(set(r["N"] for r in rows if r["status"] == "ok")):
        sub = [r for r in rows if r["N"] == N and r["status"] == "ok"]
        if not sub: continue
        print(f"\n  Window N = {N} bars  ({N/60:.1f} hours)")
        print(f"  {'tag':<22}  {'strat':<14}  {'baseline':>10}  {'B5':>10}  "
              f"{'Δ':>8}  {'trd':>5}  {'win%':>5}  {'eq':>6}  {'RLexit%':>8}")
        print("  " + "─" * 102)
        deltas = []
        for r in sorted(sub, key=lambda r: r["k"]):
            print(f"  {r['tag']:<22}  {r['strat']:<14}  "
                  f"{r['baseline_sharpe']:>+10.3f}  "
                  f"{r['best_val_sharpe']:>+10.3f}  "
                  f"{r['delta_sharpe']:>+8.3f}  "
                  f"{r['best_val_trades']:>5,}  "
                  f"{r['best_val_winrate']*100:>4.1f}%  "
                  f"{r['best_val_equity']:>6.3f}  "
                  f"{r['rl_exit_pct']:>7.1f}%")
            deltas.append(r["delta_sharpe"])
        n_pos = sum(1 for d in deltas if d > 0)
        print(f"  ── {n_pos}/{len(sub)} positive Δ, mean Δ {np.mean(deltas):+.3f}, "
              f"max Δ {max(deltas):+.3f}")

    # ── overall best ────────────────────────────────────────────────────────
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["delta_sharpe"])
        print(f"\n\n  Overall best Δ vs baseline: {best['tag']}  ({best['strat']}, N={best['N']})  "
              f"Δ={best['delta_sharpe']:+.3f}  ({best['baseline_sharpe']:+.3f} → {best['best_val_sharpe']:+.3f})")

        best_abs = max(ok, key=lambda r: r["best_val_sharpe"])
        print(f"  Overall best abs Sharpe   : {best_abs['tag']}  Sharpe {best_abs['best_val_sharpe']:+.3f}")

    print(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--only-n", type=int, default=None,
                     help="run only cells with this window size (60/120/240)")
    ap.add_argument("--only-strat", type=int, default=None,
                     help="run only cells with this strategy index (0..8)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.ticker, seed=args.seed, only_n=args.only_n, only_strat=args.only_strat)
