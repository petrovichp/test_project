"""
Vote × lag joint filter — combine R1 (vote_count → quality) with the
signal-entry-latency finding (lag=0 = fresh signal).

Hypothesis: trades with vote=5 AND lag=0 are the highest-conviction setups
(unanimous policies AND fresh signal). Test whether this joint filter gives
cleaner separation than either alone.

Output: bucketed table of (vote_count × lag_bin) mean PnL, win rate, count.
       Tests a vote×lag sizing rule against the AGGRESSIVE quadratic baseline.

Run: python3 -m models.vote_lag_joint
"""
import json, time, math, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.audit_vote5_dd import run_walkforward

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
TAKER_FEE = 0.00045
RL_START = 100_000
RL_END   = 383_174
N_FOLDS  = 6
FOLD_SIZE = (RL_END - RL_START) // N_FOLDS
N_BARS_PER_YEAR = 525_960


def load_v8_nets():
    out = []
    for s in SEEDS:
        n = DuelingDQN(52, 12, 256)
        n.load_state_dict(torch.load(
            CACHE / "policies" / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt", map_location="cpu"))
        n.eval(); out.append(n)
    return out


def load_full():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def reconstruct_sharpe(trades, pnl_field):
    by_fold = defaultdict(list)
    for tr in trades:
        by_fold[tr["fold"]].append(tr)
    per_fold = []
    for fold_id in range(1, N_FOLDS + 1):
        tr_list = sorted(by_fold.get(fold_id, []), key=lambda t: t["t_close"])
        n_bars = FOLD_SIZE if fold_id < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * FOLD_SIZE
        eq = np.full(n_bars, 1.0, dtype=np.float64)
        cur = 1.0
        for tr in tr_list:
            t_close = int(tr["t_close"])
            if 0 <= t_close < n_bars:
                cur *= (1.0 + tr[pnl_field])
                eq[t_close:] = cur
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sh = float(rets.mean() / rets.std() * math.sqrt(N_BARS_PER_YEAR)) if rets.std() > 1e-12 else 0.0
        per_fold.append(dict(fold=fold_id, sharpe=sh, n_trades=len(tr_list)))
    wf = statistics.mean(r["sharpe"] for r in per_fold)
    return wf, per_fold


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()
    full_signals = full["signals"]
    n_total = full_signals.shape[0]
    nets = load_v8_nets()

    print(f"\n{'='*120}\n  Vote × lag joint filter analysis (VOTE5_v8_H256_DD K=5, fee=4.5bp)\n{'='*120}\n")

    print(f"  Collecting trades + computing lag per trade ...")
    rows, trades = run_walkforward(nets, full, atr_median, fee=TAKER_FEE, with_reason=True)

    # Compute lag for each trade
    for tr in trades:
        decision_bar_global = (tr["fold"] - 1) * FOLD_SIZE + tr["t_open"] - 1
        if not (0 <= decision_bar_global < n_total):
            tr["lag"] = -1
            continue
        k = tr["strat_idx"]; d = tr["direction"]
        t_back = decision_bar_global
        while t_back > 0 and int(full_signals[t_back - 1, k]) == d:
            t_back -= 1
        tr["lag"] = decision_bar_global - t_back

    print(f"    {len(trades)} trades, lag computed\n")

    # ── 2-D bucket table: vote × lag ──
    print(f"{'='*120}\n  2-D bucket table — count, mean PnL%, win rate (fee=4.5bp)\n{'='*120}\n")

    lag_buckets = [(0, 0, "lag=0"), (1, 3, "lag 1-3"), (4, 9999, "lag 4+")]
    vote_buckets = [3, 4, 5]

    print(f"  COUNTS:")
    print(f"  {'votes':<8} | " + " | ".join(f"{lb[2]:^10}" for lb in lag_buckets) + " | total")
    for v in vote_buckets:
        row = []; tot = 0
        for lo, hi, lbl in lag_buckets:
            sub = [tr for tr in trades if tr["votes_count"] == v and lo <= tr["lag"] <= hi]
            row.append(len(sub)); tot += len(sub)
        print(f"  v={v:<6} | " + " | ".join(f"{c:>10,}" for c in row) + f" | {tot:,}")

    print(f"\n  MEAN PnL% per bucket:")
    print(f"  {'votes':<8} | " + " | ".join(f"{lb[2]:^10}" for lb in lag_buckets))
    for v in vote_buckets:
        row = []
        for lo, hi, lbl in lag_buckets:
            sub = [tr["pnl"] for tr in trades if tr["votes_count"] == v and lo <= tr["lag"] <= hi]
            row.append(np.mean(sub)*100 if sub else 0.0)
        print(f"  v={v:<6} | " + " | ".join(f"{m:>+10.4f}" for m in row))

    print(f"\n  WIN RATE %:")
    print(f"  {'votes':<8} | " + " | ".join(f"{lb[2]:^10}" for lb in lag_buckets))
    for v in vote_buckets:
        row = []
        for lo, hi, lbl in lag_buckets:
            sub = [tr["pnl"] for tr in trades if tr["votes_count"] == v and lo <= tr["lag"] <= hi]
            if sub:
                wr = (np.array(sub) > 0).mean() * 100
            else:
                wr = 0.0
            row.append(wr)
        print(f"  v={v:<6} | " + " | ".join(f"{wr:>9.1f}%" for wr in row))

    # ── Test joint filter sizing schemes ──
    print(f"\n{'='*120}\n  Sizing rules combining vote × lag\n{'='*120}\n")

    def apply_sizing(trades, fn, field):
        for tr in trades:
            tr[field] = tr["pnl"] * fn(tr["votes_count"], tr["lag"])

    schemes = [
        ("FIXED baseline",
         lambda v, l: 1.0),
        ("AGGRESSIVE quadratic (R1 winner)",
         lambda v, l: ((v - 2) / 3) ** 2),
        ("Vote-only LINEAR (v-2)/3",
         lambda v, l: (v - 2) / 3),
        ("v=5 AND lag=0 → 1.5×, v=5 lag>0 → 1.0×, v=4 → 0.5×, else 0.2×",
         lambda v, l: 1.5 if (v == 5 and l == 0) else (1.0 if v == 5 else (0.5 if v == 4 else 0.2))),
        ("v=5 → 1.0×, v=4 lag≤1 → 0.7×, v=4 lag>1 → 0.4×, v=3 → 0.15×",
         lambda v, l: 1.0 if v == 5 else (0.7 if (v == 4 and l <= 1) else (0.4 if v == 4 else 0.15))),
        ("Multiplicative: vote_quadratic × lag_bonus",
         lambda v, l: ((v - 2) / 3) ** 2 * (1.2 if l == 0 else (1.0 if l <= 3 else 0.7))),
        ("Conservative: only enter v=5 lag=0 (1.0×)",
         lambda v, l: 1.0 if (v == 5 and l == 0) else 0.0),
    ]

    print(f"  {'scheme':<70} {'WF':>8} {'pos':>5}")
    for name, fn in schemes:
        apply_sizing(trades, fn, "pnl_sized")
        wf, pf = reconstruct_sharpe(trades, "pnl_sized")
        pos = sum(1 for r in pf if r["sharpe"] > 0)
        per_fold = " ".join(f"{r['sharpe']:>+6.2f}" for r in pf)
        print(f"  {name:<70} {wf:>+8.3f}   {pos}/6  [{per_fold}]")

    # ── Save output ──
    out = CACHE / "results" / "vote_lag_joint.json"
    out.write_text(json.dumps(dict(
        n_trades=len(trades),
        elapsed=time.perf_counter() - t0,
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
