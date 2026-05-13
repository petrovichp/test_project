"""
Signal → entry latency analysis.

When a strategy signal first fires (becomes non-zero), how many bars pass
before the policy actually enters via that strategy?

Two metrics:
  1. ENTRY_LAG: for each entered trade, bars between first fire of the
     current signal burst and the policy's decision bar.
     0 = policy acts immediately on first-bar signal fire
     >0 = policy waits for the signal to persist before entering
  2. BURST_LIFETIME: how long each signal "burst" lasts (consecutive
     bars with same direction) — gives the time window the policy has
     to act before the signal disappears
  3. MISSED_BURSTS: signal bursts that started but the policy never
     entered during them (NO_TRADE chosen instead)

Run on VOTE5_v8_H256_DD K=5 (the deployable taker policy).

Run: python3 -m models.signal_entry_latency
"""
import json, time, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.audit_vote5_dd import run_walkforward

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
STRAT_KEYS = ["S1_VolDir","S2_Funding","S3_BBExt","S4_MACD","S6_TwoSignal",
              "S7_OIDiverg","S8_TakerSus","S10_Squeeze","S12_VWAPVol",
              "S11_Basis","S13_OBDiv"]
TAKER_FEE = 0.00045
RL_START = 100_000
RL_END   = 383_174
N_FOLDS  = 6
FOLD_SIZE = (RL_END - RL_START) // N_FOLDS


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


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()
    nets = load_v8_nets()
    full_signals = full["signals"]   # shape (N_bars_total, 11)
    n_total = full_signals.shape[0]

    print(f"\n{'='*120}\n  Signal → entry latency analysis (VOTE5_v8_H256_DD K=5)\n{'='*120}\n")
    print(f"  Collecting trades at fee={TAKER_FEE*1e4:.1f}bp ...")
    rows, trades = run_walkforward(nets, full, atr_median, fee=TAKER_FEE, with_reason=True)
    print(f"    {len(trades)} trades collected\n")

    # ─────────────────────────────────────────────────────────────────────
    # PART 1: Entry lag per trade
    # ─────────────────────────────────────────────────────────────────────
    print(f"{'='*120}\n  PART 1 — ENTRY LAG (bars from signal first-fire to policy entry decision)\n{'='*120}\n")

    latencies = []
    latencies_by_strat = defaultdict(list)
    for tr in trades:
        # decision bar = t_open - 1 (policy decides at decision_bar, entry at t_open)
        fold = tr["fold"]
        decision_bar_rel = tr["t_open"] - 1
        # Convert fold-relative to global signals index
        decision_bar_global = (fold - 1) * FOLD_SIZE + decision_bar_rel
        if not (0 <= decision_bar_global < n_total):
            continue
        k = tr["strat_idx"]
        d = tr["direction"]
        # Walk backward: find first bar t' where signals[t', k] == d, contiguous to decision_bar
        t_back = decision_bar_global
        while t_back > 0 and int(full_signals[t_back - 1, k]) == d:
            t_back -= 1
        latency = decision_bar_global - t_back
        latencies.append(latency)
        latencies_by_strat[k].append(latency)

    latencies = np.array(latencies)
    print(f"  Aggregate ENTRY_LAG distribution ({len(latencies)} trades):")
    print(f"  {'metric':<10} {'bars':>6}")
    print(f"  {'mean':<10} {latencies.mean():>6.2f}")
    print(f"  {'median':<10} {np.median(latencies):>6.1f}")
    for p in (10, 25, 75, 90, 95, 99):
        print(f"  p{p:<9} {np.percentile(latencies, p):>6.1f}")
    print(f"  {'max':<10} {latencies.max():>6}")

    # Histogram by buckets
    buckets = [(0,0),(1,1),(2,3),(4,7),(8,15),(16,31),(32,63),(64,127),(128,255),(256,9999)]
    counts = [int(((latencies >= lo) & (latencies <= hi)).sum()) for lo, hi in buckets]
    print(f"\n  ENTRY_LAG bucketed histogram:")
    print(f"  {'bucket':<14} {'count':>6} {'pct':>6}")
    n = len(latencies)
    for (lo, hi), c in zip(buckets, counts):
        lbl = f"= {lo}" if lo == hi else (f"{lo}-{hi}" if hi < 9999 else f">= {lo}")
        print(f"  lat {lbl:<10} {c:>6} {c/n*100:>5.1f}%")

    print(f"\n  Per-strategy ENTRY_LAG (mean / median):")
    print(f"  {'strategy':<14} {'n':>5} {'mean':>7} {'median':>8} {'p90':>6} {'max':>6}")
    for k_idx in sorted(latencies_by_strat.keys(), key=lambda k: -len(latencies_by_strat[k])):
        L = np.array(latencies_by_strat[k_idx])
        print(f"  {STRAT_KEYS[k_idx]:<14} {len(L):>5} {L.mean():>7.2f} "
              f"{np.median(L):>8.1f} {np.percentile(L, 90):>6.1f} {L.max():>6}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 2: Burst lifetime — how long do signal bursts typically last
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}\n  PART 2 — BURST LIFETIME (how long each signal stays non-zero)\n{'='*120}\n")

    bursts_by_strat = defaultdict(list)
    for k in range(full_signals.shape[1]):
        col = full_signals[:, k].astype(np.int64)
        # find runs of same non-zero value
        i = 0
        while i < len(col):
            v = col[i]
            if v == 0:
                i += 1; continue
            j = i
            while j < len(col) and int(col[j]) == int(v):
                j += 1
            bursts_by_strat[k].append(j - i)
            i = j

    print(f"  Per-strategy burst lifetime (in bars):")
    print(f"  {'strategy':<14} {'n_bursts':>9} {'mean_bars':>10} {'median':>8} {'p90':>6} {'max':>6}")
    for k_idx in range(11):
        bursts = bursts_by_strat.get(k_idx, [])
        if not bursts: continue
        B = np.array(bursts)
        print(f"  {STRAT_KEYS[k_idx]:<14} {len(B):>9,} {B.mean():>10.2f} "
              f"{np.median(B):>8.1f} {np.percentile(B, 90):>6.1f} {B.max():>6}")

    # ─────────────────────────────────────────────────────────────────────
    # PART 3: Missed bursts — signal bursts where the policy DIDN'T enter
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}\n  PART 3 — MISSED BURSTS (signal fired but policy chose NO_TRADE)\n{'='*120}\n")
    print(f"  This shows how many opportunities the policy passes on.\n")

    # For each strategy, count bursts that overlapped with an actual entry
    # vs bursts that didn't lead to entry
    entered_burst_starts = defaultdict(set)
    for tr in trades:
        k = tr["strat_idx"]
        d = tr["direction"]
        decision_bar_rel = tr["t_open"] - 1
        decision_bar_global = (tr["fold"] - 1) * FOLD_SIZE + decision_bar_rel
        if not (0 <= decision_bar_global < n_total):
            continue
        # find the burst this trade was inside of
        t_back = decision_bar_global
        while t_back > 0 and int(full_signals[t_back - 1, k]) == d:
            t_back -= 1
        entered_burst_starts[k].add(t_back)

    print(f"  Per-strategy burst-hit / burst-miss rate (full_signals is already the RL period):")
    print(f"  {'strategy':<14} {'tot_bursts':>11} {'hit':>5} {'miss':>6} {'hit%':>7}")
    for k_idx in range(11):
        rl_bursts_starts = []
        col = full_signals[:, k_idx].astype(np.int64)   # full_signals is already RL period
        i = 0
        while i < len(col):
            v = col[i]
            if v == 0:
                i += 1; continue
            j = i
            while j < len(col) and int(col[j]) == int(v):
                j += 1
            rl_bursts_starts.append(i)
            i = j
        tot = len(rl_bursts_starts)
        if tot == 0:
            continue
        hit = sum(1 for s in rl_bursts_starts if s in entered_burst_starts.get(k_idx, set()))
        miss = tot - hit
        print(f"  {STRAT_KEYS[k_idx]:<14} {tot:>11,} {hit:>5,} {miss:>6,} {hit/tot*100:>6.2f}%")

    # Output JSON
    out = CACHE / "results" / "signal_entry_latency.json"
    out.write_text(json.dumps(dict(
        n_trades=len(trades),
        entry_lag_stats=dict(
            mean=float(latencies.mean()), median=float(np.median(latencies)),
            p25=float(np.percentile(latencies, 25)), p75=float(np.percentile(latencies, 75)),
            p90=float(np.percentile(latencies, 90)), p95=float(np.percentile(latencies, 95)),
            max=int(latencies.max())),
        per_strategy_lag={STRAT_KEYS[k]: dict(n=len(v), mean=float(np.mean(v)),
                                                 median=float(np.median(v)))
                           for k, v in latencies_by_strat.items()},
        burst_lifetime={STRAT_KEYS[k]: dict(n=len(v), mean=float(np.mean(v)),
                                              median=float(np.median(v)))
                         for k, v in bursts_by_strat.items()},
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
