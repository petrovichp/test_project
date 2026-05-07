"""
Group B — exit-timing DQN sweep across (fee × per-strategy).

Cells:
  B1: global exit DQN, taker fee  0.0008
  B2: global exit DQN, maker fee  0.0004
  B3: global exit DQN, fee-free   0.0
  B4: per-strategy exit DQN, maker fee — runs one DQN per entry strategy
        (B4_S1, B4_S2, ..., B4_S12) and aggregates.

Each cell trains from scratch on cached state arrays. Cell artefacts:
  cache/btc_exit_dqn_policy_{cell_id}.pt
  cache/btc_exit_dqn_history_{cell_id}.json
  cache/btc_exit_dqn_groupB_summary.json (this script)

Run: python3 -m models.group_b_sweep [ticker]
"""

import sys, time, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache"

# (cell_id, fee, strat_filter, description)
CELLS = [
    ("B1",     0.0008, -1, "global exit DQN @ taker fee"),
    ("B2",     0.0004, -1, "global exit DQN @ maker fee"),
    ("B3",     0.0000, -1, "global exit DQN @ fee-free"),
    ("B4_S0",  0.0004,  0, "per-strategy S1_VolDir   @ maker fee"),
    ("B4_S1",  0.0004,  1, "per-strategy S2_Funding  @ maker fee"),
    ("B4_S2",  0.0004,  2, "per-strategy S3_BBRevert @ maker fee"),
    ("B4_S3",  0.0004,  3, "per-strategy S4_MACDTrend @ maker fee"),
    ("B4_S4",  0.0004,  4, "per-strategy S6_TwoSignal @ maker fee"),
    ("B4_S5",  0.0004,  5, "per-strategy S7_OIDiverg  @ maker fee"),
    ("B4_S6",  0.0004,  6, "per-strategy S8_TakerFlow @ maker fee"),
    ("B4_S7",  0.0004,  7, "per-strategy S10_Squeeze  @ maker fee"),
    ("B4_S8",  0.0004,  8, "per-strategy S12_VWAPVol  @ maker fee"),
]

STRAT_KEYS = ["S1_VolDir", "S2_Funding", "S3_BBRevert", "S4_MACDTrend",
               "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow",
               "S10_Squeeze", "S12_VWAPVol"]


def run_cell(cell_id: str, fee: float, strat_filter: int,
              ticker: str = "btc", seed: int = 42) -> dict:
    print(f"\n{'#'*78}\n#  CELL {cell_id}: fee={fee:.4f}  strat_filter="
          f"{STRAT_KEYS[strat_filter] if strat_filter >= 0 else 'ALL'}\n{'#'*78}")

    cmd = [
        "python3", "-m", "models.exit_dqn",
        ticker,
        "--tag",  cell_id,
        "--seed", str(seed),
        "--fee",  str(fee),
        "--strat-filter", str(strat_filter),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  ✗ cell {cell_id} FAILED")
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        return dict(cell_id=cell_id, fee=fee, strat_filter=strat_filter,
                     status="failed", elapsed=elapsed)

    # echo summary block
    for line in res.stdout.splitlines():
        if any(s in line for s in ["TRAINING SUMMARY", "best val Sharpe",
                                      "rule-only baseline", "Δ vs rule-only",
                                      "rule-only:", "[step", "Early stop"]):
            print("    " + line)

    hist_path = CACHE / f"{ticker}_exit_dqn_history_{cell_id}.json"
    if not hist_path.exists():
        return dict(cell_id=cell_id, fee=fee, strat_filter=strat_filter,
                     status="no_history", elapsed=elapsed)
    h = json.loads(hist_path.read_text())
    history = h.get("history", [])
    if not history:
        return dict(cell_id=cell_id, fee=fee, strat_filter=strat_filter,
                     status="empty_history", elapsed=elapsed)

    best_step      = h["best_step"]
    best_val_sharpe = h["best_val_sharpe"]
    baseline       = h["baseline"]
    best_event = next((e for e in history if e["step"] == best_step), None)
    if best_event is None:
        best_event = max(history, key=lambda e: e["val_sharpe"])

    return dict(
        cell_id          = cell_id,
        fee              = fee,
        strat_filter     = strat_filter,
        strat_name       = STRAT_KEYS[strat_filter] if strat_filter >= 0 else "ALL",
        status           = "ok",
        elapsed          = elapsed,
        best_step        = best_step,
        best_val_sharpe  = best_val_sharpe,
        best_val_trades  = best_event["val_trades"],
        best_val_winrate = best_event["val_winrate"],
        best_val_equity  = best_event["val_equity"],
        best_val_max_dd  = best_event["val_max_dd"],
        rl_exit_pct      = best_event["rl_exit_pct"],
        exit_breakdown   = best_event["exit_breakdown"],
        baseline_sharpe  = baseline["sharpe"],
        baseline_trades  = baseline["n_trades"],
        baseline_winrate = baseline["win_rate"],
        baseline_equity  = baseline["equity_final"],
        delta_sharpe     = best_val_sharpe - baseline["sharpe"],
        n_steps_trained  = h["history"][-1]["step"],
    )


def run(ticker: str = "btc", seed: int = 42, only_b4: bool = False,
         skip_b4: bool = False):
    print(f"\n{'='*78}\n  GROUP B — EXIT-TIMING DQN SWEEP  ({ticker.upper()})\n{'='*78}")
    cells = CELLS
    if only_b4:
        cells = [c for c in cells if c[0].startswith("B4")]
    elif skip_b4:
        cells = [c for c in cells if not c[0].startswith("B4")]
    for cid, fee, sf, desc in cells:
        sn = STRAT_KEYS[sf] if sf >= 0 else "ALL"
        print(f"  {cid:<7}  fee={fee:.4f}  strat={sn:<14}  → {desc}")

    rows = []
    t_total = time.perf_counter()
    for cid, fee, sf, _desc in cells:
        rows.append(run_cell(cid, fee, sf, ticker, seed))
    total_elapsed = time.perf_counter() - t_total

    out_json = CACHE / f"{ticker}_exit_dqn_groupB_summary.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str))

    # ── result table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*100}\n  GROUP B — RESULT TABLE\n{'='*100}")
    print(f"\n  {'cell':<7} {'fee':>7}  {'strat':<14}  {'baseline':>9}  {'RL exit':>9}  "
          f"{'ΔSharpe':>8}  {'trades':>7}  {'win%':>6}  {'eq':>6}  {'RLexit%':>8}")
    print("  " + "─" * 100)
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['cell_id']:<7} {r['fee']:>7.4f}  FAILED ({r['status']})")
            continue
        print(f"  {r['cell_id']:<7} {r['fee']:>7.4f}  {r['strat_name']:<14}  "
              f"{r['baseline_sharpe']:>+9.3f}  "
              f"{r['best_val_sharpe']:>+9.3f}  "
              f"{r['delta_sharpe']:>+8.3f}  "
              f"{r['best_val_trades']:>7,}  "
              f"{r['best_val_winrate']*100:>5.1f}%  "
              f"{r['best_val_equity']:>6.3f}  "
              f"{r['rl_exit_pct']:>7.1f}%")

    # ── highlight ──────────────────────────────────────────────────────────
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["delta_sharpe"])
        print(f"\n  Best Δ vs rule-only: {best['cell_id']} ({best['strat_name']})  "
              f"Δ={best['delta_sharpe']:+.3f}  "
              f"({best['baseline_sharpe']:+.3f} → {best['best_val_sharpe']:+.3f})")

        # decision rule per next_steps.md
        positive = [r for r in ok if r["delta_sharpe"] > 0]
        n_pos = len(positive)
        print(f"\n  Cells with positive Δ: {n_pos}/{len(ok)}")
        if n_pos == 0:
            print(f"  → Rule-based exits are already optimal. Drop Group B, focus elsewhere.")
        elif best["delta_sharpe"] > 4.0:
            print(f"  → ≥4 Sharpe lift achieved at fee={best['fee']:.4f}. "
                  f"Real value-add. Proceed to Group C (entry+exit stack).")
        else:
            print(f"  → Modest lift (best Δ {best['delta_sharpe']:+.3f}). "
                  f"Below the +4-Sharpe gate from next_steps.md. Document and move on.")

    print(f"\n\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--only-b4", action="store_true")
    ap.add_argument("--skip-b4", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.ticker, seed=args.seed, only_b4=args.only_b4, skip_b4=args.skip_b4)
