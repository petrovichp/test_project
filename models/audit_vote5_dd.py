"""
VOTE5_DD trade analysis:
  Part A: deal-by-deal audit (like A4 but for Double_Dueling)
  Part B: fee sensitivity (0, 1bp, 2bp, 4bp, 6bp, 8bp, 10bp per-side)
  Part C: trade-count reduction strategies
"""
import json, pathlib, statistics, time
from collections import Counter, defaultdict
import numpy as np
import torch

from models.dqn_network          import DuelingDQN
from models.dqn_rollout          import _build_exit_arrays, STRAT_KEYS
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.diagnostics_ab       import _simulate_one_trade_fee
from models.analyze_a2_rule      import _simulate_one_trade_fee_with_reason

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS = [42, 7, 123, 0, 99]
EXIT_REASONS = {0: "TP", 1: "SL", 2: "TIME", 3: "TSL", 4: "BE", 5: "OTHER"}
REGIME_NAMES = ["calm", "trend_up", "trend_down", "ranging", "chop"]


def load_dueling(tag: str) -> DuelingDQN:
    net = DuelingDQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def vote_action(nets, s_t, valid_t):
    """K-seed plurality vote, tie → NO_TRADE. Returns (action, votes_count)."""
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
        return 0, top[0][1]
    return top[0][0], top[0][1]


def run_fold(state, valid, signals, prices, atr, regime_id, ts,
              tp, sl, trail, tab, be, ts_bars, nets, fee: float = 0.0,
              vote_threshold: int = 1,
              strat_allowlist: set = None,
              with_reason: bool = False, fold_id: int = 0):
    """Single-fold rollout. Returns equity_curve, sharpe, trades-list."""
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trades = []

    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False

        action, votes_count = vote_action(nets, s_t, valid_t)

        # filtering: vote threshold
        if action != 0 and votes_count < vote_threshold:
            action = 0
        # filtering: strategy allowlist
        if action != 0 and strat_allowlist is not None and (action - 1) not in strat_allowlist:
            action = 0

        if action == 0:
            t_next = t + 1
        else:
            k = action - 1
            direction = int(signals[t, k])
            if direction == 0:
                t_next = t + 1
            else:
                if with_reason:
                    pnl, n_held, reason = _simulate_one_trade_fee_with_reason(
                        prices, t + 1, direction,
                        float(tp[t, k]), float(sl[t, k]),
                        float(trail[t, k]), float(tab[t, k]),
                        float(be[t, k]),   int(ts_bars[t, k]),
                        0, fee,
                    )
                else:
                    pnl, n_held = _simulate_one_trade_fee(
                        prices, t + 1, direction,
                        float(tp[t, k]), float(sl[t, k]),
                        float(trail[t, k]), float(tab[t, k]),
                        float(be[t, k]),   int(ts_bars[t, k]),
                        0, fee,
                    )
                    reason = -1
                t_close = t + 1 + n_held
                if t_close >= n_bars: t_close = n_bars - 1
                eq_arr[t:t_close + 1] = equity
                equity *= (1.0 + float(pnl))
                eq_arr[t_close + 1:] = equity
                if t_close == n_bars - 1: eq_arr[-1] = equity
                peak = max(peak, equity); last_pnl = float(pnl)
                trades.append(dict(
                    fold=fold_id, t_open=int(t+1), t_close=int(t_close),
                    bars_held=int(n_held), strat_idx=int(k), strat_key=STRAT_KEYS[k],
                    direction=int(direction), votes_count=int(votes_count),
                    pnl=float(pnl), exit_reason=int(reason),
                    regime=int(regime_id[t+1]) if t+1 < n_bars else int(regime_id[-1]),
                ))
                t_next = t_close + 1
        t = t_next

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return eq_arr, float(sharpe), float(equity), trades


def run_walkforward(nets, full, atr_median, fee: float = 0.0,
                     vote_threshold: int = 1, strat_allowlist: set = None,
                     with_reason: bool = False):
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    rows = []
    all_trades = []
    for i in range(N_FOLDS):
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL; b = b_pq - RL_START_REL
        sub = {k: full[k][a:b] for k in full}
        tp, sl, tr, tab, be, ts_bars = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        eq_arr, sh, eq_final, trades = run_fold(
            sub["state"], sub["valid_actions"], sub["signals"], sub["price"],
            sub["atr"], sub["regime_id"], sub["ts"],
            tp, sl, tr, tab, be, ts_bars, nets, fee=fee,
            vote_threshold=vote_threshold, strat_allowlist=strat_allowlist,
            with_reason=with_reason, fold_id=i+1)
        rows.append(dict(fold=i+1, sharpe=sh, equity=eq_final, trades=len(trades)))
        all_trades.extend(trades)
    return rows, all_trades


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()
    nets = [load_dueling(f"VOTE5_DD_seed{s}") for s in SEEDS]

    print(f"\n{'='*120}\n  VOTE5_DD ANALYSIS\n{'='*120}")

    # ── PART A: deal-by-deal audit ─────────────────────────────────────────
    print(f"\n{'='*60}\n  PART A — deal-by-deal audit (full WF, fee=0)\n{'='*60}")
    rows, trades = run_walkforward(nets, full, atr_median, fee=0.0, with_reason=True)

    print(f"\n  Per-fold equity:")
    print(f"    {'fold':<5} {'trades':>7} {'BTC return':>11} {'WF Sharpe':>11} {'equity':>9}")
    for r, full_row in zip(rows, range(N_FOLDS)):
        # need BTC return per fold
        fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
        a = RL_START_REL + full_row * fold_size - RL_START_REL
        b = RL_START_REL + (full_row + 1) * fold_size - RL_START_REL if full_row < N_FOLDS - 1 else RL_END_REL - RL_START_REL
        prices_fold = full["price"][a:b]
        btc_ret = (prices_fold[-1] / prices_fold[0] - 1) * 100
        print(f"    {r['fold']:<5} {r['trades']:>7} {btc_ret:>+10.2f}% {r['sharpe']:>+11.3f} {r['equity']:>9.3f}")
    wf_mean = statistics.mean(r["sharpe"] for r in rows)
    print(f"\n  Total trades: {len(trades)}, WF mean Sharpe: {wf_mean:+.3f}")

    # per-strategy
    print(f"\n  Per-strategy aggregate:")
    print(f"    {'strategy':<14} {'count':>6} {'long':>5} {'short':>6} {'mean PnL':>10} "
          f"{'win %':>7} {'sum PnL':>9}")
    by_strat = defaultdict(list)
    for tr in trades: by_strat[tr["strat_key"]].append(tr)
    for key in STRAT_KEYS:
        if key not in by_strat: continue
        rows_s = by_strat[key]
        n = len(rows_s)
        n_long = sum(1 for r in rows_s if r["direction"] == 1)
        n_short = sum(1 for r in rows_s if r["direction"] == -1)
        pnls = [r["pnl"] for r in rows_s]
        print(f"    {key:<14} {n:>6} {n_long:>5} {n_short:>6} "
              f"{statistics.mean(pnls)*100:>+9.3f}% "
              f"{(sum(1 for p in pnls if p > 0)/n)*100:>6.1f}% "
              f"{sum(pnls)*100:>+8.2f}%")

    # exit reasons
    print(f"\n  Exit reason breakdown:")
    by_reason = Counter(tr["exit_reason"] for tr in trades)
    for r, n in sorted(by_reason.items()):
        name = EXIT_REASONS.get(r, f"R{r}")
        rows_r = [t for t in trades if t["exit_reason"] == r]
        m = statistics.mean(t["pnl"] for t in rows_r) * 100
        win = sum(1 for t in rows_r if t["pnl"] > 0) / len(rows_r) * 100
        bars = statistics.mean(t["bars_held"] for t in rows_r)
        print(f"    {name:<6} {n:>4} ({n/len(trades)*100:>5.1f}%)  mean PnL {m:>+7.3f}%  "
              f"win {win:>5.1f}%  avg bars {bars:>6.1f}")

    # vote count distribution
    by_v = Counter(tr["votes_count"] for tr in trades)
    print(f"\n  Vote count distribution:")
    for v, n in sorted(by_v.items()):
        rows_v = [t for t in trades if t["votes_count"] == v]
        m = statistics.mean(t["pnl"] for t in rows_v) * 100 if rows_v else 0
        print(f"    {v} votes: {n:>4} trades ({n/len(trades)*100:>5.1f}%)  mean PnL {m:>+7.3f}%")

    # avg holding period
    avg_bars = statistics.mean(t["bars_held"] for t in trades)
    avg_long_bars = statistics.mean(t["bars_held"] for t in trades if t["direction"] == 1)
    avg_short_bars = statistics.mean(t["bars_held"] for t in trades if t["direction"] == -1)
    print(f"\n  Holding period:")
    print(f"    overall: {avg_bars:.1f} bars (~{avg_bars:.0f} min)")
    print(f"    long:    {avg_long_bars:.1f} bars")
    print(f"    short:   {avg_short_bars:.1f} bars")

    # PART B: FEE SENSITIVITY ─────────────────────────────────────────────
    print(f"\n\n{'='*60}\n  PART B — fee sensitivity\n{'='*60}")
    fee_levels = [0.0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.0012, 0.0020]
    print(f"\n  {'fee (per side)':>15} {'fee bp':>7} {'WF mean':>10} {'val Sharpe':>11} "
          f"{'test Sharpe':>12} {'WF folds+':>10} {'trades':>8}")
    print("  " + "-"*100)
    fee_results = []
    for fee in fee_levels:
        rows_f, trades_f = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in rows_f)
        wf_pos = sum(1 for r in rows_f if r["sharpe"] > 0)
        # also val and test single-shot
        sp_val = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
        sp_test = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_val["price"], sp_val["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_test["price"], sp_test["atr"], atr_median)
        _, val_sh, _, _ = run_fold(sp_val["state"], sp_val["valid_actions"], sp_val["signals"],
                                       sp_val["price"], sp_val["atr"], sp_val["regime_id"], sp_val["ts"],
                                       tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee, fold_id=0)
        _, test_sh, _, _ = run_fold(sp_test["state"], sp_test["valid_actions"], sp_test["signals"],
                                        sp_test["price"], sp_test["atr"], sp_test["regime_id"], sp_test["ts"],
                                        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee, fold_id=0)
        fee_results.append(dict(fee=fee, wf_mean=wf, wf_pos=wf_pos,
                                  val_sharpe=val_sh, test_sharpe=test_sh,
                                  trades=len(trades_f)))
        print(f"  {fee:>15.4f} {fee*10000:>7.1f} {wf:>+10.3f} {val_sh:>+11.3f} "
              f"{test_sh:>+12.3f} {wf_pos:>3}/6     {len(trades_f):>8}")

    # PART C: TRADE REDUCTION STRATEGIES ──────────────────────────────────
    print(f"\n\n{'='*60}\n  PART C — trade-count reduction\n{'='*60}")
    print(f"\n  Tested at fee=0.0004 (4bp = OKX maker tier baseline assumption):")
    test_fee = 0.0004

    print(f"\n  {'strategy':<35} {'trades':>7} {'val Sh':>9} {'test Sh':>9} {'WF':>9} {'folds+':>7}")
    print("  " + "-"*90)
    rc_results = []

    def measure(name, vote_thr=1, allowlist=None):
        rows_r, trades_r = run_walkforward(nets, full, atr_median, fee=test_fee,
                                              vote_threshold=vote_thr,
                                              strat_allowlist=allowlist, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in rows_r)
        wf_pos = sum(1 for r in rows_r if r["sharpe"] > 0)
        sp_val = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
        sp_test = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_val["price"], sp_val["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_test["price"], sp_test["atr"], atr_median)
        _, val_sh, _, _ = run_fold(sp_val["state"], sp_val["valid_actions"], sp_val["signals"],
                                       sp_val["price"], sp_val["atr"], sp_val["regime_id"], sp_val["ts"],
                                       tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=test_fee,
                                       vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        _, test_sh, _, _ = run_fold(sp_test["state"], sp_test["valid_actions"], sp_test["signals"],
                                        sp_test["price"], sp_test["atr"], sp_test["regime_id"], sp_test["ts"],
                                        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=test_fee,
                                        vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        rc_results.append(dict(name=name, vote_thr=vote_thr,
                                  allowlist=list(allowlist) if allowlist else None,
                                  trades=len(trades_r), wf_mean=wf, wf_pos=wf_pos,
                                  val_sharpe=val_sh, test_sharpe=test_sh))
        print(f"  {name:<35} {len(trades_r):>7} {val_sh:>+9.2f} {test_sh:>+9.2f} {wf:>+9.3f} {wf_pos:>4}/6")
        return len(trades_r), wf, val_sh, test_sh

    measure("baseline (no filtering)")
    measure("vote ≥ 3", vote_thr=3)
    measure("vote ≥ 4", vote_thr=4)
    measure("vote ≥ 5", vote_thr=5)
    # strategy filters — by sum-PnL ranking from Part A:
    # S1, S8, S7, S10, S4 are top contributors; drop the 4 weakest (S2, S3, S6, S12)
    top5 = {0, 6, 5, 7, 3}   # S1, S8, S7, S10, S4 — indices 0,7,6,3,5? Let me use STRAT_KEYS positions
    # STRAT_KEYS = ["S1_VolDir", "S2_Funding", "S3_BBRevert", "S4_MACDTrend",
    #              "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow", "S10_Squeeze", "S12_VWAPVol"]
    # indices:        0           1            2              3              4              5            6             7             8
    measure("top-5 strategies (drop S2,S3,S6,S12)", allowlist={0, 3, 5, 6, 7})
    measure("top-3 strategies (S1, S8, S7)",       allowlist={0, 5, 6})
    # combined
    measure("top-5 + vote ≥ 3", vote_thr=3, allowlist={0, 3, 5, 6, 7})
    measure("top-5 + vote ≥ 4", vote_thr=4, allowlist={0, 3, 5, 6, 7})

    # save
    out = CACHE / "results" / "audit_vote5_dd_results.json"
    out.write_text(json.dumps(dict(
        part_a=dict(per_fold=rows, n_trades=len(trades),
                       wf_mean=wf_mean, by_strategy={k: len(v) for k, v in by_strat.items()},
                       by_exit=dict(by_reason),
                       by_votes=dict(by_v),
                       avg_bars=avg_bars),
        part_b_fee_sensitivity=fee_results,
        part_c_trade_reduction=rc_results,
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
