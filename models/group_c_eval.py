"""
Group C1 — sequential composition of A4 entry DQN + B4 per-strategy exit DQNs.

No retraining. Loads:
  cache/btc_dqn_policy_A4.pt          (entry DQN, 50→64→32→10)
  cache/btc_exit_dqn_policy_B4_S{k}.pt for k in 0..8  (exit DQNs, 28→64→32→2)

Walks DQN-val and DQN-test:
  for bar t:
    s_t = entry state (50-dim with stateful 18,19 filled)
    action = A4 greedy(s_t, valid_t)            # ∈ {NO_TRADE, S1..S12}
    if action == 0:  t += 1; continue
    k = action - 1
    direction = signals[t, k]
    if direction == 0: t += 1; continue          # mask edge case
    # simulate trade with rule exits + B4_Sk exit DQN deciding HOLD/EXIT_NOW
    pnl, n_held, exit_reason = simulate_trade_episode(
        ..., policy_fn = greedy(B4_Sk))
    advance equity, jump past trade

Compare against:
  A4 + rule-only exits (the existing A4 baseline already in btc_dqn_train_history_A4.json)
  Re-evaluated here for clean apples-to-apples on exact same code path.

Output:
  cache/btc_groupC1_summary.json — val/test Sharpe, exit breakdown, per-strat
                                     trade attribution.

Run: python3 -m models.group_c_eval [ticker] [--fee 0.0004]
"""

import sys, time, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax
from models.dqn_rollout    import _build_exit_arrays, STRAT_KEYS
from models.exit_dqn       import (ExitDQN, simulate_trade_episode,
                                     EXIT_STATE_DIM, N_ACTIONS, MAX_TRADE_BARS,
                                     _AlwaysHold, _GreedyExit)

CACHE = ROOT / "cache"


# ── load helpers ─────────────────────────────────────────────────────────────

def _load_entry_policy(ticker: str, tag: str = "A4") -> DQN:
    path = CACHE / f"{ticker}_dqn_policy_{tag}.pt"
    net  = DQN(state_dim=50, n_actions=10, hidden=64)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def _load_exit_policies(ticker: str, prefix: str = "B4_S") -> list:
    """Load 9 per-strategy exit DQNs (e.g. B4_S0..S8 or B4_fee0_S0..S8).
    Index aligns with STRAT_KEYS."""
    nets = []
    for k in range(len(STRAT_KEYS)):
        path = CACHE / f"{ticker}_exit_dqn_policy_{prefix}{k}.pt"
        if not path.exists():
            print(f"  ! missing exit policy {path.name} — using rule-only fallback for k={k}")
            nets.append(None)
            continue
        net  = ExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=64)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


# ── unified evaluator (rule-only OR per-strategy RL exits) ───────────────────

def evaluate_combined(
    entry_net: DQN, exit_nets,
    state, valid, signals_strat, prices,
    tp, sl, trail, tab, be, ts_bars,
    fee: float, use_rl_exits: bool = True,
):
    """Walk through all bars with A4 entry policy. On each trade entry, simulate
    the trade with either rule-only exits (always HOLD) or B4_Sk exit DQN
    + rule-based exits. Returns per-bar equity curve + summary stats.

    `exit_nets` may be None when use_rl_exits=False.
    """
    n_bars = len(state)
    n_actions = valid.shape[1]
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)

    trade_pnls = []
    trade_durs = []
    actions_count = np.zeros(n_actions, dtype=np.int64)
    per_strat_pnls = [[] for _ in range(9)]
    rl_exits = tp_exits = sl_exits = time_exits = eod_exits = invalid = 0

    # build greedy exit policies once (one per strategy); None entries fall back to HOLD
    hold_policy = _AlwaysHold()
    if use_rl_exits:
        exit_policies = [_GreedyExit(net) if net is not None else hold_policy
                          for net in exit_nets]

    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))

        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False

        with torch.no_grad():
            sb = torch.from_numpy(s_t).float().unsqueeze(0)
            vb = torch.from_numpy(valid_t).bool().unsqueeze(0)
            action = int(masked_argmax(entry_net, sb, vb).item())
        actions_count[action] += 1

        if action == 0:
            t += 1
            continue

        k = action - 1
        direction = int(signals_strat[t, k])
        if direction == 0:
            t += 1
            continue

        exit_pol = exit_policies[k] if use_rl_exits else hold_policy
        pnl, n_held, reason = simulate_trade_episode(
            state, prices, entry_bar=t + 1, direction=direction,
            tp_pct=float(tp[t, k]), sl_pct=float(sl[t, k]),
            trail_pct=float(trail[t, k]), tab_pct=float(tab[t, k]),
            breakeven_pct=float(be[t, k]),
            time_stop_bars=int(ts_bars[t, k]),
            fee=fee, policy_fn=exit_pol,
            transitions_out=None,
        )
        if reason == "INVALID":
            invalid += 1
            t += 1
            continue

        t_close = t + 1 + n_held
        if t_close >= n_bars: t_close = n_bars - 1
        eq_arr[t:t_close + 1] = equity
        equity *= (1.0 + pnl)
        eq_arr[t_close + 1:] = equity
        peak = max(peak, equity)
        last_pnl = pnl

        trade_pnls.append(pnl)
        trade_durs.append(n_held)
        per_strat_pnls[k].append(pnl)
        if   reason == "RL_EXIT": rl_exits   += 1
        elif reason == "TP":      tp_exits   += 1
        elif reason == "SL":      sl_exits   += 1
        elif reason == "TIME":    time_exits += 1
        elif reason == "EOD":     eod_exits  += 1
        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    pnls_arr = np.array(trade_pnls, dtype=np.float64)
    win_rate = (pnls_arr > 0).mean() if len(pnls_arr) else 0.0
    eq_dd = eq_arr / np.maximum.accumulate(eq_arr)
    max_dd = float(eq_dd.min() - 1.0) if len(eq_arr) else 0.0

    # per-strategy attribution
    per_strat = []
    for k in range(9):
        pl = np.array(per_strat_pnls[k], dtype=np.float64)
        per_strat.append(dict(
            strat=STRAT_KEYS[k],
            n_trades=int(len(pl)),
            mean_pnl_pct=float(pl.mean() * 100) if len(pl) else 0.0,
            total_pnl_pct=float(pl.sum() * 100) if len(pl) else 0.0,
            win_rate=float((pl > 0).mean()) if len(pl) else 0.0,
        ))

    return dict(
        sharpe       = float(sharpe),
        equity_final = float(equity),
        equity_peak  = float(peak),
        max_dd       = max_dd,
        n_trades     = int(len(pnls_arr)),
        win_rate     = float(win_rate),
        mean_pnl_pct = float(pnls_arr.mean() * 100) if len(pnls_arr) else 0.0,
        mean_duration= float(np.mean(trade_durs)) if trade_durs else 0.0,
        rl_exit_pct  = float(rl_exits / max(1, len(pnls_arr)) * 100),
        actions      = actions_count.tolist(),
        exit_breakdown = dict(RL_EXIT=int(rl_exits), TP=int(tp_exits),
                                SL=int(sl_exits), TIME=int(time_exits),
                                EOD=int(eod_exits), INVALID=int(invalid)),
        per_strat    = per_strat,
        eq_arr       = eq_arr,            # for plotting later
    )


def _print_summary(label: str, r: dict, baseline: dict = None):
    print(f"\n  {label}:")
    print(f"    Sharpe          : {r['sharpe']:>+8.3f}"
          + (f"   (Δ vs baseline {r['sharpe']-baseline['sharpe']:>+6.3f})" if baseline else ""))
    print(f"    Equity final    : {r['equity_final']:>8.4f}    peak {r['equity_peak']:.4f}")
    print(f"    Max DD          : {r['max_dd']*100:>+7.2f}%")
    print(f"    Trades          : {r['n_trades']:>5,}     win {r['win_rate']*100:>5.1f}%   "
          f"mean PnL {r['mean_pnl_pct']:>+6.3f}%   mean duration {r['mean_duration']:.1f} bars")
    print(f"    Exit breakdown  : {r['exit_breakdown']}")
    if r['n_trades']:
        print(f"    Per-strategy attribution:")
        for ps in r['per_strat']:
            if ps['n_trades'] == 0: continue
            print(f"      {ps['strat']:<14}  n={ps['n_trades']:>4}  "
                  f"meanPnL {ps['mean_pnl_pct']:>+6.3f}%  totPnL {ps['total_pnl_pct']:>+7.2f}%  "
                  f"win {ps['win_rate']*100:>5.1f}%")


def run(ticker: str = "btc", fee: float = 0.0004, entry_tag: str = "A4",
         exit_prefix: str = "B4_S", out_tag: str = "C1"):
    t0 = time.perf_counter()
    print(f"\n{'='*78}\n  GROUP {out_tag} — {entry_tag} ENTRY + {exit_prefix.rstrip('_S').rstrip('_')} EXITS  "
          f"({ticker.upper()})  fee={fee:.4f}\n{'='*78}")

    # ── load policies ────────────────────────────────────────────────────────
    print(f"  loading entry policy {entry_tag} ...")
    entry_net = _load_entry_policy(ticker, entry_tag)
    print(f"  loading 9 exit policies {exit_prefix}0..8 ...")
    exit_nets = _load_exit_policies(ticker, prefix=exit_prefix)

    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_med = float(vol["atr_train_median"])

    results = {}
    for split in ("val", "test"):
        sp = np.load(CACHE / f"{ticker}_dqn_state_{split}.npz")
        print(f"\n  {'─'*72}")
        print(f"  Split {split}: {sp['state'].shape[0]:,} bars")
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_med)

        # rule-only baseline (A4 entry + always-HOLD exit policy)
        t1 = time.perf_counter()
        base = evaluate_combined(
            entry_net, None, sp["state"], sp["valid_actions"],
            sp["signals"], sp["price"], tp, sl, tr, tab, be, ts,
            fee=fee, use_rl_exits=False)
        print(f"  rule-only ({split}) eval in {time.perf_counter()-t1:.1f}s")

        # combined: A4 entry + B4 per-strategy RL exits
        t1 = time.perf_counter()
        combo = evaluate_combined(
            entry_net, exit_nets, sp["state"], sp["valid_actions"],
            sp["signals"], sp["price"], tp, sl, tr, tab, be, ts,
            fee=fee, use_rl_exits=True)
        print(f"  combined  ({split}) eval in {time.perf_counter()-t1:.1f}s")

        _print_summary(f"{split.upper()}  rule-only ({entry_tag} entry + rule exits)", base)
        _print_summary(f"{split.upper()}  combined  ({entry_tag} entry + {exit_prefix.rstrip('_S').rstrip('_')} RL exits)",
                        combo, baseline=base)

        results[split] = dict(rule_only=_pack(base), combined=_pack(combo))

    # ── final summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  GROUP {out_tag} — RESULT TABLE\n{'='*78}")
    print(f"\n  {'split':<6}  {'baseline (A4+rule)':>20}  {'combined (A4+B4)':>18}  {'ΔSharpe':>10}  "
          f"{'Δeq':>9}")
    print("  " + "─" * 75)
    for split in ("val", "test"):
        r = results[split]
        d_sharpe = r["combined"]["sharpe"] - r["rule_only"]["sharpe"]
        d_eq     = r["combined"]["equity_final"] - r["rule_only"]["equity_final"]
        print(f"  {split:<6}  {r['rule_only']['sharpe']:>+11.3f} ({r['rule_only']['equity_final']:.3f})  "
              f"{r['combined']['sharpe']:>+10.3f} ({r['combined']['equity_final']:.3f})  "
              f"{d_sharpe:>+10.3f}  {d_eq*100:>+8.2f}%")

    # ── save artefacts ──────────────────────────────────────────────────────
    out = CACHE / f"{ticker}_group{out_tag}_summary.json"
    payload = dict(ticker=ticker, fee=fee, entry_tag=entry_tag,
                    exit_prefix=exit_prefix,
                    results={s: dict(rule_only=_pack(results[s]["rule_only"], drop_eq=True),
                                       combined=_pack(results[s]["combined"], drop_eq=True))
                              for s in ("val", "test")})
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  → {out.name}")
    print(f"\n  total time {time.perf_counter()-t0:.1f}s")
    return results


def _pack(d: dict, drop_eq: bool = False) -> dict:
    """Trim eq_arr from result dict for JSON serialization."""
    out = {k: v for k, v in d.items() if k != "eq_arr"} if drop_eq else dict(d)
    if "eq_arr" in out:
        out["eq_arr"] = out["eq_arr"].tolist()
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--entry-tag", default="A4", dest="entry_tag")
    ap.add_argument("--exit-prefix", default="B4_S", dest="exit_prefix",
                     help="prefix for exit-policy filenames (e.g. 'B4_S' or 'B4_fee0_S')")
    ap.add_argument("--out-tag", default="C1", dest="out_tag",
                     help="tag for output JSON, e.g. 'C1' or 'C1_fee0'")
    args = ap.parse_args()
    run(args.ticker, fee=args.fee, entry_tag=args.entry_tag,
         exit_prefix=args.exit_prefix, out_tag=args.out_tag)
