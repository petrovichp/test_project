"""
Mixed-fee evaluation — entries always taker (4.5bp), TP exits maker (2bp),
other exits taker (4.5bp).

Hypothesis: TP-hit trades can use limit-orders at TP (placed ahead of time;
price comes to us = maker fill). SL/TSL/BE/TIME exits are urgent → taker.

Output: WF / val / test Sharpe per policy under (uniform 9bp) vs (mixed
TP=6.5bp/else=9bp), per-strategy TP-hit rates.

Run: python3 -m models.mixed_fee_eval
"""
import json, math, statistics, time
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict

from models.dqn_network import DuelingDQN
from models.audit_vote5_dd import run_walkforward, run_fold
from models.dqn_rollout import _build_exit_arrays
from models.voting_ensemble import _VotePolicy

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
EXIT_NAMES = ["TP", "SL", "TSL", "BE", "TIME", "EOD"]
STRAT_KEYS = ["S1_VolDir","S2_Funding","S3_BBExt","S4_MACD","S6_TwoSignal",
              "S7_OIDiverg","S8_TakerSus","S10_Squeeze","S12_VWAPVol",
              "S11_Basis","S13_OBDiv"]

TAKER_BPS = 4.5
MAKER_BPS = 2.0
TAKER_FEE = TAKER_BPS / 1e4
MAKER_FEE = MAKER_BPS / 1e4
N_BARS_PER_YEAR = 525_960  # 1-min annualization

RL_START = 100_000
RL_END   = 383_174
N_FOLDS  = 6


def load_net_dueling(tag, seed):
    n = DuelingDQN(52, 12, 256)
    n.load_state_dict(torch.load(
        CACHE / "policies" / f"btc_dqn_policy_{tag}_seed{seed}.pt", map_location="cpu"))
    n.eval(); return n


def load_full(suffix):
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_{suffix}.npz")
        for key in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def adjust_pnl(raw_pnl_at_fee0, exit_reason, entry_fee, taker_fee):
    """Convert raw pnl (computed at fee=0) to mixed-fee pnl.
    entry: always taker.  exit: maker if TP (reason==0) else taker.
    """
    exit_fee = MAKER_FEE if exit_reason == 0 else taker_fee
    return raw_pnl_at_fee0 - entry_fee - exit_fee


def reconstruct_sharpe(trades, fold_size_bars, pnl_field="pnl"):
    """Build per-fold equity arrays. NOTE: t_close in trade dict is fold-relative
    (run_fold operates on a sliced state, so its 't' loop indexes start at 0).
    """
    by_fold = defaultdict(list)
    for tr in trades:
        by_fold[tr["fold"]].append(tr)

    per_fold = []
    for fold_id in range(1, N_FOLDS + 1):
        tr_list = sorted(by_fold.get(fold_id, []), key=lambda t: t["t_close"])
        # Last fold gets the modulo extra bars
        n_bars = fold_size_bars if fold_id < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * fold_size_bars
        eq = np.full(n_bars, 1.0, dtype=np.float64)
        cur = 1.0
        for tr in tr_list:
            t_close = int(tr["t_close"])     # fold-relative
            if t_close < 0 or t_close >= n_bars:
                continue
            cur *= (1.0 + tr[pnl_field])
            eq[t_close:] = cur
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        if rets.std() > 1e-12:
            sh = float(rets.mean() / rets.std() * math.sqrt(N_BARS_PER_YEAR))
        else:
            sh = 0.0
        per_fold.append(dict(fold=fold_id, sharpe=sh, equity=float(cur), n_trades=len(tr_list)))
    wf = statistics.mean(r["sharpe"] for r in per_fold)
    return wf, per_fold


def evaluate_policy(label, nets, full, atr_median, fold_size_bars):
    """Run at fee=TAKER_FEE to get the realistic trade sequence + exit reasons.
    The simulator's pnl already subtracts 2*TAKER_FEE. For TP-exit trades, we
    add back (TAKER_FEE - MAKER_FEE) to reflect cheaper maker exit."""
    print(f"\n  Collecting trades at fee={TAKER_BPS}bp uniform for {label} ...")
    rows, trades = run_walkforward(nets, full, atr_median, fee=TAKER_FEE, with_reason=True)
    print(f"    {len(trades)} trades collected; reasons:",
          dict(Counter(EXIT_NAMES[t["exit_reason"]] for t in trades)))

    # pnl from simulator already has 2*TAKER_FEE subtracted (entry+exit taker).
    # For TP exits, refund (TAKER_FEE - MAKER_FEE) since exit was a maker.
    maker_savings = TAKER_FEE - MAKER_FEE
    for tr in trades:
        tr["pnl_uniform_9bp"] = tr["pnl"]   # already at 9bp uniform
        if tr["exit_reason"] == 0:          # TP
            tr["pnl_mixed"] = tr["pnl"] + maker_savings
        else:
            tr["pnl_mixed"] = tr["pnl"]

    # Reconstruct equity curves under each fee model
    wf_uniform, pf_uniform = reconstruct_sharpe(trades, fold_size_bars, "pnl_uniform_9bp")
    wf_mixed,   pf_mixed   = reconstruct_sharpe(trades, fold_size_bars, "pnl_mixed")

    # Per-strategy breakdown
    by_strat = defaultdict(lambda: dict(n=0, n_tp=0, sum_pnl_raw=0.0,
                                          sum_pnl_uniform=0.0, sum_pnl_mixed=0.0))
    for tr in trades:
        k = STRAT_KEYS[tr["strat_idx"]]
        d = by_strat[k]
        d["n"] += 1
        if tr["exit_reason"] == 0:  # TP
            d["n_tp"] += 1
        d["sum_pnl_raw"]     += tr["pnl"]
        d["sum_pnl_uniform"] += tr["pnl_uniform_9bp"]
        d["sum_pnl_mixed"]   += tr["pnl_mixed"]

    return dict(label=label, n_trades=len(trades),
                wf_uniform=wf_uniform, wf_mixed=wf_mixed,
                per_fold_uniform=pf_uniform, per_fold_mixed=pf_mixed,
                by_strategy=dict(by_strat),
                trade_count_by_reason=dict(Counter(EXIT_NAMES[t["exit_reason"]] for t in trades)))


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    full = load_full("v8_s11s13")
    fold_size = (RL_END - RL_START) // N_FOLDS  # = 47195

    print(f"\n{'='*120}")
    print(f"  Mixed-fee evaluation — entry taker {TAKER_BPS}bp, TP exit maker {MAKER_BPS}bp, other exits taker {TAKER_BPS}bp")
    print(f"  Baseline comparison: uniform taker {TAKER_BPS}bp on both sides (= {2*TAKER_BPS}bp round-trip)")
    print(f"{'='*120}")

    results = []

    # ── DISTILL_v8_seed42 (single net, fee=0 deployable) ──
    print(f"\n--- DISTILL_v8_seed42 (single net) ---")
    dnet = load_net_dueling("DISTILL_v8", 42)
    r1 = evaluate_policy("DISTILL_v8_seed42", [dnet], full, atr_median, fold_size)
    results.append(r1)

    # ── VOTE5_v8_H256_DD (K=5 ensemble, fee=4.5bp deployable) ──
    print(f"\n--- VOTE5_v8_H256_DD K=5 plurality ---")
    vnets = [load_net_dueling("VOTE5_v8_H256_DD", s) for s in SEEDS]
    r2 = evaluate_policy("VOTE5_v8_H256_DD K=5", vnets, full, atr_median, fold_size)
    results.append(r2)

    # ── Report ──
    print(f"\n{'='*120}\n  AGGREGATE RESULTS\n{'='*120}\n")
    print(f"  {'policy':<28} {'n_trades':>9} {'TP_rate':>9} "
          f"{'WF uniform 9bp':>15} {'WF mixed':>10} {'Δ':>7}")
    for r in results:
        n_tp = sum(1 for stx in r["by_strategy"].values() for _ in [stx] if True) * 0  # placeholder
        # actually compute TP rate from trade_count_by_reason
        tp_count = r["trade_count_by_reason"].get("TP", 0)
        tp_rate = tp_count / r["n_trades"] * 100 if r["n_trades"] else 0.0
        delta = r["wf_mixed"] - r["wf_uniform"]
        print(f"  {r['label']:<28} {r['n_trades']:>9,} {tp_rate:>8.1f}% "
              f"{r['wf_uniform']:>+15.3f} {r['wf_mixed']:>+9.3f} {delta:>+7.3f}")

    print(f"\n{'='*120}\n  PER-FOLD COMPARISON\n{'='*120}\n")
    for r in results:
        print(f"\n  {r['label']}")
        print(f"  {'fold':<5} {'uniform 9bp':>13} {'mixed':>10} {'Δ':>7} {'trades':>7}")
        for pu, pm in zip(r["per_fold_uniform"], r["per_fold_mixed"]):
            d = pm["sharpe"] - pu["sharpe"]
            print(f"  f{pu['fold']:<4} {pu['sharpe']:>+13.3f} {pm['sharpe']:>+9.3f} {d:>+7.3f} {pu['n_trades']:>7}")

    print(f"\n{'='*120}\n  EXIT-REASON BREAKDOWN\n{'='*120}\n")
    for r in results:
        total = r["n_trades"]
        print(f"\n  {r['label']}  ({total} trades)")
        for k, v in sorted(r["trade_count_by_reason"].items()):
            print(f"    {k:<6} {v:>6}  ({v/total*100:>5.1f}%)")

    print(f"\n{'='*120}\n  PER-STRATEGY TP-HIT RATE + FEE IMPACT\n{'='*120}\n")
    for r in results:
        print(f"\n  {r['label']}")
        print(f"  {'strategy':<14} {'n':>5} {'TP_rate':>9} {'sum_raw%':>10} {'sum_uniform%':>14} {'sum_mixed%':>12} {'Δ%':>7}")
        for sk, s in sorted(r["by_strategy"].items(), key=lambda kv: -kv[1]["n"]):
            if s["n"] == 0: continue
            tp_rate = s["n_tp"] / s["n"] * 100
            delta_pct = (s["sum_pnl_mixed"] - s["sum_pnl_uniform"]) * 100
            print(f"  {sk:<14} {s['n']:>5} {tp_rate:>8.1f}% "
                  f"{s['sum_pnl_raw']*100:>+10.2f} {s['sum_pnl_uniform']*100:>+14.2f} "
                  f"{s['sum_pnl_mixed']*100:>+12.2f} {delta_pct:>+7.2f}")

    out = CACHE / "results" / "mixed_fee_eval.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
