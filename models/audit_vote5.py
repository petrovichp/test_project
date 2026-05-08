"""
A4 — BASELINE_VOTE5 deal-by-deal audit.

For each trade fired by the K=5 plurality ensemble across the full WF period,
record:
  - fold (1..6)
  - bar timestamp
  - strategy index (S1..S12) and key
  - direction (long / short)
  - votes count (how many of 5 nets agreed)
  - entry / exit prices
  - holding bars
  - PnL %
  - exit reason

Aggregate to understand:
  - Per-strategy contribution: count, total PnL, win rate, mean PnL
  - Long vs short attribution per fold
  - Vote-strength → trade quality (already covered in A2 but cross-checked)
  - Regime breakdown (using regime_id from state)
"""
import json, pathlib, statistics, time
from collections import Counter, defaultdict
import numpy as np
import torch

from models.dqn_network          import DQN
from models.dqn_rollout          import _build_exit_arrays, STRAT_KEYS
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.diagnostics_ab       import _simulate_one_trade_fee
from models.analyze_a2_rule      import _simulate_one_trade_fee_with_reason

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS   = [42, 7, 123, 0, 99]


def load_net(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"{ticker}_dqn_state_{split}.npz") if False else \
             np.load(CACHE / f"btc_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr",
                     "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def trace_vote5(state, valid, signals, prices, atr, regime_id, ts,
                 tp, sl, trail, tab, be, ts_bars, nets, fold_id: int):
    """Trace BASELINE_VOTE5 trades on a fold; return list of trade dicts."""
    n_bars = len(state)
    K = len(nets)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    trades = []

    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False

        # K-seed plurality vote
        with torch.no_grad():
            sb = torch.from_numpy(s_t).float().unsqueeze(0)
            vb = torch.from_numpy(valid_t).bool()
            votes = []
            for net in nets:
                q = net(sb).squeeze(0).masked_fill(~vb, -1e9)
                votes.append(int(q.argmax().item()))
        counts = Counter(votes)
        top = counts.most_common(2)
        if len(top) >= 2 and top[0][1] == top[1][1]:
            action = 0
        else:
            action = top[0][0]
        votes_count = top[0][1]

        if action == 0:
            t_next = t + 1
        else:
            k = action - 1
            direction = int(signals[t, k])
            if direction == 0:
                t_next = t + 1
            else:
                pnl, n_held, reason = _simulate_one_trade_fee_with_reason(
                    prices, t + 1, direction,
                    float(tp[t, k]), float(sl[t, k]),
                    float(trail[t, k]), float(tab[t, k]),
                    float(be[t, k]),   int(ts_bars[t, k]),
                    0, 0.0,
                )
                t_close = t + 1 + n_held
                if t_close >= n_bars: t_close = n_bars - 1
                trades.append(dict(
                    fold=fold_id,
                    t_open=int(t+1),
                    t_close=int(t_close),
                    bars_held=int(n_held),
                    strat_idx=int(k),
                    strat_key=STRAT_KEYS[k],
                    direction=int(direction),
                    votes_count=int(votes_count),
                    entry_px=float(prices[t+1]) if t+1 < n_bars else float(prices[-1]),
                    exit_px=float(prices[t_close]),
                    pnl=float(pnl),
                    exit_reason=int(reason),
                    regime=int(regime_id[t+1]) if t+1 < n_bars else int(regime_id[-1]),
                    ts_open=int(ts[t+1]) if t+1 < n_bars else int(ts[-1]),
                ))
                equity *= (1.0 + float(pnl))
                peak = max(peak, equity); last_pnl = float(pnl)
                t_next = t_close + 1
        t = t_next
    return trades


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()
    nets = [load_net("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS]

    print(f"\n{'='*120}\n  A4 — BASELINE_VOTE5 deal-by-deal audit\n{'='*120}")

    # ── trace per fold ──────────────────────────────────────────────────────
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    all_trades = []
    fold_meta = []
    for i in range(N_FOLDS):
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL; b = b_pq - RL_START_REL
        sub = {k: full[k][a:b] for k in full}
        tp, sl, tr, tab, be, ts_bars = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        trades = trace_vote5(sub["state"], sub["valid_actions"], sub["signals"],
                                sub["price"], sub["atr"], sub["regime_id"], sub["ts"],
                                tp, sl, tr, tab, be, ts_bars, nets, fold_id=i+1)
        all_trades.extend(trades)
        # equity recalc
        eq = 1.0
        for tr_ in trades:
            eq *= (1 + tr_["pnl"])
        fold_meta.append(dict(fold=i+1, n_trades=len(trades),
                                btc_return=float(sub["price"][-1]/sub["price"][0]-1),
                                final_eq=eq))
        print(f"  fold {i+1}: {len(trades):>4} trades, equity ×{eq:.3f}, BTC return {(sub['price'][-1]/sub['price'][0]-1)*100:>+6.2f}%")

    print(f"\n  Total trades across 6 folds: {len(all_trades)}")

    # ── per-strategy aggregation ────────────────────────────────────────────
    print(f"\n  Per-strategy aggregate (across all 6 folds):")
    print(f"    {'strategy':<14} {'count':>6} {'long':>5} {'short':>6} "
          f"{'mean PnL':>10} {'win %':>7} {'sum PnL':>9} {'avg votes':>10}")
    print("    " + "-"*100)
    by_strat = defaultdict(list)
    for tr in all_trades:
        by_strat[tr["strat_key"]].append(tr)
    for key in STRAT_KEYS:
        if key not in by_strat: continue
        rows = by_strat[key]
        n = len(rows)
        n_long = sum(1 for r in rows if r["direction"] == 1)
        n_short = sum(1 for r in rows if r["direction"] == -1)
        pnls = [r["pnl"] for r in rows]
        votes = [r["votes_count"] for r in rows]
        print(f"    {key:<14} {n:>6} {n_long:>5} {n_short:>6} "
              f"{statistics.mean(pnls)*100:>+9.3f}% "
              f"{(sum(1 for p in pnls if p > 0)/n)*100:>6.1f}% "
              f"{sum(pnls)*100:>+8.2f}% "
              f"{statistics.mean(votes):>10.2f}")

    # ── per-fold long/short attribution ─────────────────────────────────────
    print(f"\n  Per-fold long/short PnL attribution:")
    print(f"    {'fold':<5} {'long n':>7} {'long PnL':>10} {'long win%':>10} "
          f"{'short n':>8} {'short PnL':>11} {'short win%':>11}")
    print("    " + "-"*90)
    by_fold = defaultdict(list)
    for tr in all_trades:
        by_fold[tr["fold"]].append(tr)
    for fi in range(1, 7):
        trs = by_fold.get(fi, [])
        longs = [t for t in trs if t["direction"] == 1]
        shorts = [t for t in trs if t["direction"] == -1]
        l_pnl = sum(t["pnl"] for t in longs) * 100
        s_pnl = sum(t["pnl"] for t in shorts) * 100
        l_win = (sum(1 for t in longs if t["pnl"] > 0) / len(longs) * 100) if longs else 0
        s_win = (sum(1 for t in shorts if t["pnl"] > 0) / len(shorts) * 100) if shorts else 0
        print(f"    {fi:<5} {len(longs):>7} {l_pnl:>+9.2f}% {l_win:>9.1f}% "
              f"{len(shorts):>8} {s_pnl:>+10.2f}% {s_win:>10.1f}%")

    # ── exit reason breakdown ───────────────────────────────────────────────
    print(f"\n  Exit reason breakdown:")
    EXIT_REASONS = {0: "TP", 1: "SL", 2: "TIME", 3: "TSL", 4: "BE"}
    by_reason = Counter(tr["exit_reason"] for tr in all_trades)
    for r, n in sorted(by_reason.items()):
        name = EXIT_REASONS.get(r, f"R{r}")
        rows = [t for t in all_trades if t["exit_reason"] == r]
        m = statistics.mean(t["pnl"] for t in rows) * 100
        win = sum(1 for t in rows if t["pnl"] > 0) / len(rows) * 100
        print(f"    {name:<6} {n:>4} ({n/len(all_trades)*100:>5.1f}%)  mean PnL {m:>+7.3f}%  win {win:>5.1f}%")

    # ── regime breakdown ────────────────────────────────────────────────────
    REGIME_NAMES = ["calm", "trend_up", "trend_down", "ranging", "chop"]
    print(f"\n  Regime breakdown:")
    by_reg = defaultdict(list)
    for tr in all_trades:
        by_reg[tr["regime"]].append(tr)
    for r in sorted(by_reg):
        rows = by_reg[r]
        n = len(rows)
        m = statistics.mean(t["pnl"] for t in rows) * 100
        win = sum(1 for t in rows if t["pnl"] > 0) / n * 100
        n_long = sum(1 for t in rows if t["direction"] == 1)
        nm = REGIME_NAMES[r] if 0 <= r < len(REGIME_NAMES) else f"R{r}"
        print(f"    {nm:<12} {n:>4} ({n/len(all_trades)*100:>5.1f}%)  "
              f"long {n_long}/{n} ({n_long/n*100:>5.1f}%)  mean PnL {m:>+7.3f}%  win {win:>5.1f}%")

    # save full trade log
    out = CACHE / "audit_vote5_trades.json"
    out.write_text(json.dumps(dict(
        fold_meta=fold_meta,
        n_trades=len(all_trades),
        trades=all_trades,
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
