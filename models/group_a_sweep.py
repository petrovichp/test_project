"""
Group A — entry-gating DQN sweep across (fee × trade_penalty).

Runs 7 cells back-to-back, each a fresh DQN training run with different
(fee, trade_penalty) parameters. Aggregates best val Sharpe, action
distribution, and trade count per cell into a comparison table.

Each cell trains from scratch on the cached state arrays — no upstream
model retraining. Cell artefacts go to:
  cache/btc_dqn_policy_{cell_id}.pt
  cache/btc_dqn_train_history_{cell_id}.json
  cache/btc_dqn_groupA_summary.parquet  (this script)
  cache/btc_dqn_groupA_summary.json

Run: python3 -m models.group_a_sweep [ticker]
"""

import sys, time, json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "cache"


CELLS = [
    # (cell_id, fee, trade_penalty, description)
    ("A0", 0.0008, 0.000, "baseline: taker fee, no penalty (replicates Phase 3)"),
    ("A1", 0.0000, 0.000, "fee-free, no penalty"),
    ("A2", 0.0000, 0.001, "fee-free + mild selectivity (penalty 0.1%)"),
    ("A3", 0.0000, 0.003, "fee-free + heavy selectivity (penalty 0.3%)"),
    ("A4", 0.0004, 0.000, "maker fee, no penalty"),
    ("A5", 0.0004, 0.001, "maker fee + mild selectivity"),
    ("A6", 0.0008, 0.001, "taker fee + selectivity (orig conditions + penalty)"),
]


def run_cell(cell_id: str, fee: float, penalty: float, ticker: str = "btc",
              seed: int = 42) -> dict:
    """Spawn a fresh dqn_selector training run, parse history JSON, return summary."""
    print(f"\n{'#'*78}\n#  CELL {cell_id}: fee={fee:.4f}  penalty={penalty:.4f}\n{'#'*78}")
    cmd = [
        "python3", "-m", "models.dqn_selector",
        ticker,
        "--mode", "all",
        "--tag",  cell_id,
        "--seed", str(seed),
        "--fee",  str(fee),
        "--trade-penalty", str(penalty),
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"  ✗ cell {cell_id} FAILED")
        print(res.stdout[-2000:])
        print(res.stderr[-2000:])
        return dict(cell_id=cell_id, fee=fee, trade_penalty=penalty,
                     status="failed", elapsed=elapsed)

    # Print key lines from stdout (last block summary)
    lines = res.stdout.splitlines()
    summary_started = False
    for line in lines:
        if "TRAINING SUMMARY" in line:
            summary_started = True
        if summary_started or "[step" in line or "Early stop" in line:
            print("    " + line)

    hist_path = CACHE / f"{ticker}_dqn_train_history_{cell_id}.json"
    if not hist_path.exists():
        print(f"  ✗ history file missing: {hist_path}")
        return dict(cell_id=cell_id, fee=fee, trade_penalty=penalty,
                     status="no_history", elapsed=elapsed)

    h = json.loads(hist_path.read_text())
    history = h.get("history", [])
    if not history:
        return dict(cell_id=cell_id, fee=fee, trade_penalty=penalty,
                     status="empty_history", elapsed=elapsed)

    best_step      = h["best_step"]
    best_val_sharpe = h["best_val_sharpe"]

    # Find the validation event matching best_step (closest-not-greater)
    best_val_event = None
    for ev in history:
        if ev["step"] == best_step:
            best_val_event = ev; break
    if best_val_event is None:
        best_val_event = max(history, key=lambda e: e["val_sharpe"])

    return dict(
        cell_id          = cell_id,
        fee              = fee,
        trade_penalty    = penalty,
        status           = "ok",
        elapsed          = elapsed,
        best_step        = best_step,
        best_val_sharpe  = best_val_sharpe,
        best_val_trades  = best_val_event["val_trades"],
        best_val_winrate = best_val_event["val_winrate"],
        best_val_equity  = best_val_event["val_equity"],
        best_val_max_dd  = best_val_event["val_max_dd"],
        actions          = best_val_event["actions"],
        n_steps_trained  = h["history"][-1]["step"],
    )


def run(ticker: str = "btc", seed: int = 42):
    print(f"\n{'='*78}\n  GROUP A — DQN SWEEP  ({ticker.upper()})  {len(CELLS)} cells\n{'='*78}")
    for cell_id, fee, pen, desc in CELLS:
        print(f"  {cell_id}  fee={fee:.4f}  penalty={pen:.4f}  → {desc}")

    rows = []
    t_total = time.perf_counter()
    for cell_id, fee, pen, _desc in CELLS:
        rows.append(run_cell(cell_id, fee, pen, ticker, seed))
    total_elapsed = time.perf_counter() - t_total

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / f"{ticker}_dqn_groupA_summary.parquet", index=False)
    (CACHE / f"{ticker}_dqn_groupA_summary.json").write_text(
        json.dumps(rows, indent=2, default=str))

    # ── summary table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  GROUP A — RESULT TABLE\n{'='*78}")
    print(f"\n  {'cell':<5} {'fee':>7} {'penalty':>8}  {'val Sharpe':>10}  "
          f"{'val trades':>11}  {'val win%':>9}  {'val eq':>7}  {'val DD%':>7}  {'desc':<14}")
    print("  " + "─" * 100)
    for r, (cid, fee, pen, desc) in zip(rows, CELLS):
        if r["status"] != "ok":
            print(f"  {cid:<5} {fee:>7.4f} {pen:>8.4f}  FAILED ({r['status']})")
            continue
        wr = r["best_val_winrate"] * 100
        eq = r["best_val_equity"]
        dd = r["best_val_max_dd"] * 100
        print(f"  {cid:<5} {fee:>7.4f} {pen:>8.4f}  "
              f"{r['best_val_sharpe']:>+10.3f}  "
              f"{r['best_val_trades']:>11,}  "
              f"{wr:>8.1f}%  "
              f"{eq:>7.3f}  "
              f"{dd:>+6.1f}%  {desc:<14}")

    # action distribution per cell
    print(f"\n\n  Action distribution at best step (% of validation steps):")
    action_names = ["NO_TRADE", "S1", "S2", "S3", "S4", "S6", "S7", "S8", "S10", "S12"]
    print(f"  {'cell':<5}  " + " ".join(f"{n:>7}" for n in action_names))
    print("  " + "─" * (5 + 8 * len(action_names) + 2))
    for r in rows:
        if r["status"] != "ok": continue
        a = r["actions"]
        total = sum(a) if sum(a) > 0 else 1
        line = f"  {r['cell_id']:<5}  " + " ".join(f"{a[i]/total*100:>6.1f}%" for i in range(10))
        print(line)

    print(f"\n\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    return df


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
