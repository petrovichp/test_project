"""
A2 — Trade quality decomposition by vote agreement.

For each trade fired by the K-seed plurality ensemble, record how many seeds
voted for the chosen action. Then stratify per-trade PnL by that count.

Hypothesis: higher agreement → higher per-trade Sharpe. If so, position sizing
proportional to vote count would lift aggregate Sharpe.

Runs on BASELINE_VOTE5 (K=5) and the K=10 ensemble. Reports:
  - Per-agreement-count: count, mean PnL, win rate, hypothetical Sharpe
  - Whether monotone (higher agreement = better)

Then builds a sizing policy: trade size ∝ (votes - threshold) / (K - threshold)
and re-evaluates WF/test/val.
"""
import json, pathlib, statistics, time
from collections import Counter, defaultdict
import numpy as np
import torch

from models.dqn_network          import DQN
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.diagnostics_ab       import _simulate_one_trade_fee

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS_K5  = [42, 7, 123, 0, 99]
SEEDS_K10 = [42, 7, 123, 0, 99, 1, 13, 25, 50, 77]
TAG_FOR   = {s: ("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS_K10}


def load_net(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def evaluate_with_logging(nets, state, valid, signals, prices,
                            tp, sl, trail, tab, be, ts_bars,
                            size_fn=None):
    """Run K-seed plurality, log per-trade agreement count.

    size_fn: callable (votes_count, K) -> float in [0, 1] for position sizing.
            If None, full-size (1.0) trades.

    Returns dict with sharpe, equity, n_trades, per-trade tuples (pnl, votes).
    """
    K = len(nets)
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_log = []   # list of (pnl, votes_count)
    n_trades = 0

    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False

        # K-seed votes (plurality, tie → NO_TRADE)
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
                pnl, n_held = _simulate_one_trade_fee(
                    prices, t + 1, direction,
                    float(tp[t, k]), float(sl[t, k]),
                    float(trail[t, k]), float(tab[t, k]),
                    float(be[t, k]),   int(ts_bars[t, k]),
                    0, 0.0,
                )
                # apply optional sizing
                size = 1.0 if size_fn is None else float(size_fn(votes_count, K))
                effective_pnl = pnl * size

                t_close = t + 1 + n_held
                if t_close >= n_bars: t_close = n_bars - 1
                eq_arr[t:t_close + 1] = equity
                equity *= (1.0 + effective_pnl)
                eq_arr[t_close + 1:] = equity
                if t_close == n_bars - 1: eq_arr[-1] = equity
                peak = max(peak, equity); last_pnl = effective_pnl
                trade_log.append((float(pnl), votes_count, float(size)))
                n_trades += 1
                t_next = t_close + 1
        t = t_next

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return dict(sharpe=float(sharpe), equity=float(equity),
                n_trades=int(n_trades), trade_log=trade_log)


def stratify_by_votes(trade_log, K):
    """Stats per votes_count level."""
    by_v = defaultdict(list)
    for pnl, votes, _size in trade_log:
        by_v[votes].append(pnl)
    rows = []
    for v in sorted(by_v):
        pnls = by_v[v]
        n = len(pnls)
        m = statistics.mean(pnls)
        sd = statistics.stdev(pnls) if n > 1 else 0.0
        win = sum(1 for p in pnls if p > 0) / n
        rows.append(dict(votes=v, n=n, mean_pnl=m, std=sd, win_rate=win,
                          per_trade_sharpe=(m/sd if sd > 1e-12 else 0.0)))
    return rows


def eval_walkforward_logged(nets, atr_median, full, size_fn=None):
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    rows = []
    all_trades = []
    for i in range(N_FOLDS):
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL; b = b_pq - RL_START_REL
        sub = {k: full[k][a:b] for k in full}
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        out = evaluate_with_logging(nets, sub["state"], sub["valid_actions"],
                                       sub["signals"], sub["price"],
                                       tp, sl, tr, tab, be, ts, size_fn=size_fn)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"],
                          equity=out["equity"], trades=out["n_trades"]))
        all_trades.extend(out["trade_log"])
    return rows, all_trades


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()

    print(f"\n{'='*120}\n  A2 — TRADE QUALITY BY VOTE AGREEMENT\n{'='*120}")

    for label, seeds in [("K=5 (BASELINE_VOTE5)", SEEDS_K5), ("K=10", SEEDS_K10)]:
        K = len(seeds)
        nets = [load_net(TAG_FOR[s]) for s in seeds]

        print(f"\n  ── {label} (K={K}) — full-size trades ──")
        rows, trades = eval_walkforward_logged(nets, atr_median, full, size_fn=None)
        wf_mean = statistics.mean([r["sharpe"] for r in rows])
        wf_pos  = sum(1 for r in rows if r["sharpe"] > 0)
        print(f"  WF mean Sharpe = {wf_mean:+.3f}, folds positive = {wf_pos}/6, total trades = {len(trades)}")
        print(f"\n  Per-vote-count stats (across all WF folds):")
        print(f"    {'votes':>6} {'count':>6} {'mean PnL':>10} {'std':>9} "
              f"{'win %':>7} {'per-trade Sharpe':>17}")
        strat = stratify_by_votes(trades, K)
        for r in strat:
            print(f"    {r['votes']:>6} {r['n']:>6} {r['mean_pnl']*100:>+9.3f}% {r['std']*100:>8.3f}% "
                  f"{r['win_rate']*100:>6.1f}% {r['per_trade_sharpe']:>+17.3f}")

        # Test vote-weighted sizing: size = (votes - K/2) / (K - K/2), clamped [0,1]
        # i.e. votes=K/2 → size 0, votes=K → size 1, linear in between
        def size_linear(votes, K):
            half = K / 2
            if votes <= half: return 0.0
            return (votes - half) / (K - half)

        def size_threshold_high(votes, K):
            """Only trade if votes >= 0.6*K, full size."""
            return 1.0 if votes >= 0.6 * K else 0.0

        # additional thresholds
        def size_threshold_70(votes, K): return 1.0 if votes >= 0.7*K else 0.0
        def size_threshold_80(votes, K): return 1.0 if votes >= 0.8*K else 0.0
        def size_quadratic(votes, K):
            half = K / 2
            if votes <= half: return 0.0
            return ((votes - half) / (K - half)) ** 2

        variants = [
            ("linear (size = (votes - K/2)/(K/2))", size_linear),
            (f"threshold ≥{int(0.6*K + 0.5)}/{K}", size_threshold_high),
            (f"threshold ≥{int(0.7*K + 0.5)}/{K}", size_threshold_70),
            (f"threshold ≥{int(0.8*K + 0.5)}/{K}", size_threshold_80),
            ("quadratic (steep)", size_quadratic),
        ]
        for sname, sfn in variants:
            rows_s, trades_s = eval_walkforward_logged(nets, atr_median, full, size_fn=sfn)
            wf_mean_s = statistics.mean([r["sharpe"] for r in rows_s])
            wf_pos_s  = sum(1 for r in rows_s if r["sharpe"] > 0)
            n_eff = sum(1 for p, v, sz in trades_s if sz > 0)
            print(f"\n  Sized variant — {sname}:")
            print(f"    WF mean Sharpe = {wf_mean_s:+.3f} ({wf_mean_s-wf_mean:+.3f})  "
                  f"folds + = {wf_pos_s}/6  effective trades = {n_eff}/{len(trades_s)}")
            print(f"    per-fold: " + str([round(r['sharpe'], 2) for r in rows_s]))

    print(f"\n  Total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
