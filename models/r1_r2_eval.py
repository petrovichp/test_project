"""
R1 + R2 — vote-strength position sizing + trade clustering / capacity.

R1: VOTE5_v8 K=5 produces vote consensus per trade (votes_count = 3/4/5).
    Test whether weighting position size by consensus improves Sharpe.

R2: Trade timing characterization for live deployment readiness — inter-trade
    intervals, peak concurrent positions, burst rates.

Run: python3 -m models.r1_r2_eval
"""
import json, math, statistics, time
from collections import Counter, defaultdict
from datetime import datetime
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
N_BARS_PER_YEAR = 525_960
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


def reconstruct_sharpe(trades, fold_size_bars, pnl_field):
    """Per-fold Sharpe from trade log (t_close is fold-relative in run_fold)."""
    by_fold = defaultdict(list)
    for tr in trades:
        by_fold[tr["fold"]].append(tr)
    per_fold = []
    for fold_id in range(1, N_FOLDS + 1):
        tr_list = sorted(by_fold.get(fold_id, []), key=lambda t: t["t_close"])
        n_bars = fold_size_bars if fold_id < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * fold_size_bars
        eq = np.full(n_bars, 1.0, dtype=np.float64)
        cur = 1.0
        for tr in tr_list:
            t_close = int(tr["t_close"])
            if 0 <= t_close < n_bars:
                cur *= (1.0 + tr[pnl_field])
                eq[t_close:] = cur
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sh = float(rets.mean() / rets.std() * math.sqrt(N_BARS_PER_YEAR)) if rets.std() > 1e-12 else 0.0
        per_fold.append(dict(fold=fold_id, sharpe=sh, equity=float(cur), n_trades=len(tr_list)))
    wf = statistics.mean(r["sharpe"] for r in per_fold)
    return wf, per_fold


def apply_sizing(trades, size_fn, field_out):
    """Scale each trade's pnl by size_fn(votes_count). Stores into trade[field_out]."""
    for tr in trades:
        s = size_fn(tr["votes_count"])
        tr[field_out] = tr["pnl"] * s


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()
    nets = load_v8_nets()

    print(f"\n{'='*120}\n  R1 + R2 — vote-strength sizing + trade timing on VOTE5_v8_H256_DD K=5\n{'='*120}\n")

    print(f"  Collecting trades at fee={TAKER_FEE*1e4:.1f}bp ...")
    rows, trades = run_walkforward(nets, full, atr_median, fee=TAKER_FEE, with_reason=True)
    n = len(trades)
    print(f"    {n} trades collected\n")

    # ─────────────────────────────────────────────────────────────────────
    # R1 — Vote-strength position sizing
    # ─────────────────────────────────────────────────────────────────────
    print(f"{'='*120}\n  R1 — Vote-strength position sizing\n{'='*120}\n")

    # First: characterize per-vote-count PnL distribution
    by_vote = defaultdict(list)
    for tr in trades:
        by_vote[tr["votes_count"]].append(tr["pnl"])

    print(f"  Per-vote-count PnL distribution (raw, no sizing):")
    print(f"  {'votes':<7} {'n':>6} {'mean_pnl%':>10} {'std_pnl%':>10} {'sum_pnl%':>10} {'win_rate%':>10}")
    for v in sorted(by_vote.keys()):
        arr = np.array(by_vote[v])
        win_rate = (arr > 0).mean() * 100
        print(f"  {v:<7} {len(arr):>6} {arr.mean()*100:>+10.4f} {arr.std()*100:>10.4f} "
              f"{arr.sum()*100:>+10.3f} {win_rate:>9.1f}%")

    # Define sizing schemes
    schemes = [
        ("FIXED (baseline)",         lambda v: 1.0),
        ("LINEAR (v-2)/3",           lambda v: (v - 2) / 3),     # 3→0.33, 4→0.67, 5→1.0
        ("AGGRESSIVE ((v-2)/3)^2",   lambda v: ((v - 2) / 3) ** 2),  # 3→0.11, 4→0.44, 5→1.0
        ("BINARY v>=4",              lambda v: 1.0 if v >= 4 else 0.0),
        ("BINARY v=5 only",          lambda v: 1.0 if v == 5 else 0.0),
        ("STEP 3=0.5, 4=1, 5=1.5",   lambda v: {3: 0.5, 4: 1.0, 5: 1.5}.get(v, 1.0)),
    ]

    print(f"\n  Sharpe under each sizing scheme:")
    print(f"  {'scheme':<28} {'WF':>8} {'f1':>7} {'f2':>7} {'f3':>7} {'f4':>7} {'f5':>7} {'f6':>7} {'pos':>4}")
    r1_results = []
    for name, fn in schemes:
        apply_sizing(trades, fn, "pnl_sized")
        wf, pf = reconstruct_sharpe(trades, FOLD_SIZE, "pnl_sized")
        pos = sum(1 for r in pf if r["sharpe"] > 0)
        per_fold_arr = [r["sharpe"] for r in pf]
        print(f"  {name:<28} {wf:>+8.3f} " + " ".join(f"{v:>+7.2f}" for v in per_fold_arr) + f"  {pos}/6")
        r1_results.append(dict(scheme=name, wf=wf, per_fold=per_fold_arr, folds_positive=pos))

    # ─────────────────────────────────────────────────────────────────────
    # R2 — Trade clustering / capacity analysis
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*120}\n  R2 — Trade clustering / capacity analysis\n{'='*120}\n")

    # Sort trades by open-time globally (need to convert fold-relative t_open to global)
    # Each fold starts at global bar (fold-1)*FOLD_SIZE within the RL window
    # The RL window starts at RL_START bars after dataset start. But for clustering
    # we just need RELATIVE timing — within each fold, use t_open.
    # For across-fold metrics, treat each fold separately first.

    # Per-fold inter-trade intervals (gap from prior trade close to next trade open)
    print(f"  Per-fold trade timing (intervals between consecutive trades):")
    print(f"  {'fold':<5} {'n_trades':>9} {'mean_gap':>10} {'median_gap':>11} {'p10':>6} {'p90':>6} {'max':>6}")
    by_fold_t = defaultdict(list)
    for tr in trades:
        by_fold_t[tr["fold"]].append(tr)
    all_gaps = []
    for fold_id in range(1, N_FOLDS + 1):
        sub = sorted(by_fold_t.get(fold_id, []), key=lambda t: t["t_open"])
        if len(sub) < 2: continue
        gaps = []
        for i in range(1, len(sub)):
            gap = sub[i]["t_open"] - sub[i-1]["t_close"]
            gaps.append(gap)
        gaps = np.array(gaps)
        all_gaps.extend(gaps.tolist())
        print(f"  f{fold_id:<4} {len(sub):>9} {gaps.mean():>10.1f} {np.median(gaps):>11.0f} "
              f"{np.percentile(gaps, 10):>6.0f} {np.percentile(gaps, 90):>6.0f} {gaps.max():>6}")
    all_gaps = np.array(all_gaps)
    print(f"\n  AGGREGATE inter-trade gap stats:  mean={all_gaps.mean():.1f}  "
          f"median={np.median(all_gaps):.0f}  p10={np.percentile(all_gaps, 10):.0f}  "
          f"p90={np.percentile(all_gaps, 90):.0f}  max={all_gaps.max()}")
    print(f"  (gap is bars between prior trade close and next trade open; 0 = back-to-back)")

    # Concurrent positions — since trades are sequential (next entry waits for prior close),
    # max concurrent = 1 by simulator design. But across STRATEGIES, real deployment could
    # in principle run them in parallel. Report bars-with-trade-active vs no-trade.
    print(f"\n  Trade duration + bars-in-position rate:")
    print(f"  {'fold':<5} {'trades':>7} {'total_bars':>11} {'in_pos_bars':>12} {'in_pos%':>9} "
          f"{'mean_dur':>10} {'p95_dur':>9}")
    aggregate_in_pos = 0; aggregate_total = 0
    for fold_id in range(1, N_FOLDS + 1):
        sub = sorted(by_fold_t.get(fold_id, []), key=lambda t: t["t_open"])
        if not sub: continue
        n_bars = FOLD_SIZE if fold_id < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * FOLD_SIZE
        in_pos = sum(tr["bars_held"] for tr in sub)
        durs = np.array([tr["bars_held"] for tr in sub])
        aggregate_in_pos += in_pos; aggregate_total += n_bars
        print(f"  f{fold_id:<4} {len(sub):>7} {n_bars:>11} {in_pos:>12} "
              f"{in_pos/n_bars*100:>8.1f}% {durs.mean():>10.1f} {np.percentile(durs, 95):>9.0f}")
    print(f"\n  AGGREGATE bars-in-position rate: {aggregate_in_pos/aggregate_total*100:.1f}% "
          f"({aggregate_in_pos:,}/{aggregate_total:,} bars)")

    # Burst rate: max trades in any 60-bar window, any 240-bar window
    print(f"\n  Burst rates (max trades in N-bar windows per fold):")
    print(f"  {'fold':<5} {'trades':>7} {'max@60':>8} {'max@240':>9} {'max@1440':>10}")
    for fold_id in range(1, N_FOLDS + 1):
        sub = sorted(by_fold_t.get(fold_id, []), key=lambda t: t["t_open"])
        if not sub: continue
        opens = np.array([tr["t_open"] for tr in sub])
        n_bars = FOLD_SIZE if fold_id < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * FOLD_SIZE
        # For each bar, count trades opened in last W bars
        def max_in_window(opens, w):
            # opens is sorted ascending
            counts = np.zeros(n_bars, dtype=np.int64)
            j = 0
            for t in range(n_bars):
                # advance j past opens older than t-w
                while j < len(opens) and opens[j] < t - w:
                    j += 1
                # count opens in [t-w, t]
                k = j
                cnt = 0
                while k < len(opens) and opens[k] <= t:
                    cnt += 1; k += 1
                counts[t] = cnt
            return int(counts.max())
        print(f"  f{fold_id:<4} {len(sub):>7} {max_in_window(opens, 60):>8} "
              f"{max_in_window(opens, 240):>9} {max_in_window(opens, 1440):>10}")
    print(f"  (max@60 = peak 1-hour trade count; max@240 = peak 4-hour; max@1440 = peak 24-hour)")

    # Strategy-mix in busy bars vs quiet bars
    print(f"\n  Per-strategy share of trades:")
    by_strat = Counter(tr["strat_idx"] for tr in trades)
    for sk_idx, cnt in by_strat.most_common():
        print(f"    {STRAT_KEYS[sk_idx]:<14} {cnt:>5} ({cnt/n*100:>5.1f}%)")

    # Output JSON
    out_data = dict(
        r1_results=r1_results,
        r1_per_vote_distribution={
            int(v): dict(n=len(by_vote[v]), mean_pnl=float(np.mean(by_vote[v])),
                           std_pnl=float(np.std(by_vote[v])), sum_pnl=float(np.sum(by_vote[v])))
            for v in sorted(by_vote.keys())
        },
        r2_inter_trade_gaps=dict(
            mean=float(all_gaps.mean()), median=float(np.median(all_gaps)),
            p10=float(np.percentile(all_gaps, 10)), p90=float(np.percentile(all_gaps, 90)),
            max=int(all_gaps.max()),
        ),
        r2_in_pos_rate=aggregate_in_pos / aggregate_total,
        r2_strategy_share={STRAT_KEYS[k]: v for k, v in by_strat.most_common()},
    )
    out = CACHE / "results" / "r1_r2_eval.json"
    out.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
