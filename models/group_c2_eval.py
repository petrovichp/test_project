"""
Group C2 — sequential composition: A2 entry + B5 fixed-window per-strategy exits.

No retraining. Loads:
  cache/btc_dqn_policy_A2.pt                              (entry DQN)
  cache/btc_exit_dqn_fixed_policy_B5_fix{N}_fee0_S{k}.pt  (per-strategy exit DQNs)

Walks DQN-val and DQN-test:
  for bar t:
    s_t = entry state (50-dim, stateful 18,19 filled)
    action = A2 greedy(s_t, valid_t)
    if action == 0: t += 1; continue
    k = action - 1
    direction = signals[t, k]
    # simulate trade with NO rule-based exits (matches B5 training conditions)
    # but with N-bar window — DQN decides exit, force-close at bar N
    pnl, n_held, exit_reason = simulate_fixed_episode(
        ..., policy_fn = greedy(B5_S{k}), N=window_N)
    advance equity, jump past trade

Compare against:
  A2 + always-HOLD-to-bar-N (the B5 baseline):  what would A2 entries do
                                                  if held to N bars no matter what?
  Ideally also: A2 + rule-based exits (the original A2 baseline, +7.30 val / +3.78 test)

Run:
  python3 -m models.group_c2_eval btc --window-n 120
"""

import sys, time, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax
from models.dqn_rollout    import STRAT_KEYS
from models.exit_dqn_fixed import (FixedExitDQN, simulate_fixed_episode,
                                     precompute_aux_arrays, EXIT_STATE_DIM,
                                     N_ACTIONS, _AlwaysHold, _GreedyFixed)

CACHE = ROOT / "cache"


def _load_entry_policy(ticker: str, tag: str = "A2") -> DQN:
    path = CACHE / f"{ticker}_dqn_policy_{tag}.pt"
    net  = DQN(state_dim=50, n_actions=10, hidden=64)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def _load_exit_policies(ticker: str, N: int, fee_tag: str = "fee0") -> list:
    nets = []
    for k in range(len(STRAT_KEYS)):
        path = CACHE / f"{ticker}_exit_dqn_fixed_policy_B5_fix{N}_{fee_tag}_S{k}.pt"
        if not path.exists():
            print(f"  ! missing exit policy {path.name} → fallback to always-HOLD")
            nets.append(None); continue
        net = FixedExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=96)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def evaluate_combined(
    entry_net: DQN, exit_nets, base_state, valid, signals_strat, prices,
    aux, regime_id, fee: float, N: int, use_rl_exits: bool = True,
):
    """A2 entry + (B5 RL or always-HOLD-to-N) exit. Walk through bars sequentially."""
    n_bars = len(base_state)
    n_actions = valid.shape[1]
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)

    trade_pnls = []; trade_durs = []
    actions_count = np.zeros(n_actions, dtype=np.int64)
    per_strat_pnls = [[] for _ in range(9)]
    rl_exits = window_exits = eod_exits = invalid = 0

    hold_pol = _AlwaysHold()
    if use_rl_exits:
        exit_pols = [_GreedyFixed(net) if net is not None else hold_pol
                       for net in exit_nets]

    t = 0
    while t < n_bars - 2:
        s_t = base_state[t].copy()
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
            t += 1; continue

        k = action - 1
        direction = int(signals_strat[t, k])
        if direction == 0:
            t += 1; continue

        pol = exit_pols[k] if use_rl_exits else hold_pol
        pnl, n_held, reason = simulate_fixed_episode(
            base_state, prices, aux, regime_id,
            entry_bar=t + 1, direction=direction, fee=fee, N=N,
            policy_fn=pol, transitions_out=None,
        )
        if reason == "INVALID":
            invalid += 1; t += 1; continue

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
        if   reason == "RL_EXIT": rl_exits     += 1
        elif reason == "WINDOW":  window_exits += 1
        elif reason == "EOD":     eod_exits    += 1
        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    pnls_arr = np.array(trade_pnls, dtype=np.float64)
    win_rate = (pnls_arr > 0).mean() if len(pnls_arr) else 0.0
    eq_dd = eq_arr / np.maximum.accumulate(eq_arr)
    max_dd = float(eq_dd.min() - 1.0) if len(eq_arr) else 0.0

    per_strat = []
    for k in range(9):
        pl = np.array(per_strat_pnls[k], dtype=np.float64)
        per_strat.append(dict(
            strat=STRAT_KEYS[k], n_trades=int(len(pl)),
            mean_pnl_pct=float(pl.mean() * 100) if len(pl) else 0.0,
            total_pnl_pct=float(pl.sum() * 100) if len(pl) else 0.0,
            win_rate=float((pl > 0).mean()) if len(pl) else 0.0))

    return dict(
        sharpe       = float(sharpe), equity_final = float(equity),
        equity_peak  = float(peak),   max_dd       = max_dd,
        n_trades     = int(len(pnls_arr)), win_rate = float(win_rate),
        mean_pnl_pct = float(pnls_arr.mean() * 100) if len(pnls_arr) else 0.0,
        mean_duration= float(np.mean(trade_durs)) if trade_durs else 0.0,
        rl_exit_pct  = float(rl_exits / max(1, len(pnls_arr)) * 100),
        actions      = actions_count.tolist(),
        exit_breakdown = dict(RL_EXIT=int(rl_exits), WINDOW=int(window_exits),
                                EOD=int(eod_exits), INVALID=int(invalid)),
        per_strat    = per_strat,
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


def run(ticker: str = "btc", N: int = 120, fee: float = 0.0,
         entry_tag: str = "A2", out_tag: str = None):
    if out_tag is None:
        out_tag = f"C2_fix{N}_fee0"
    t0 = time.perf_counter()
    print(f"\n{'='*78}\n  GROUP C2 — {entry_tag} ENTRY + B5_fix{N} EXITS  ({ticker.upper()})\n"
          f"  fee={fee:.4f}  window N={N} bars\n{'='*78}")

    print(f"  loading entry policy {entry_tag} ...")
    entry_net = _load_entry_policy(ticker, entry_tag)
    print(f"  loading 9 exit policies B5_fix{N}_fee0_S0..8 ...")
    exit_nets = _load_exit_policies(ticker, N=N)

    results = {}
    for split in ("val", "test"):
        sp = np.load(CACHE / f"{ticker}_dqn_state_{split}.npz")
        print(f"\n  {'─'*72}")
        print(f"  Split {split}: {sp['state'].shape[0]:,} bars  (window N={N})")

        aux = precompute_aux_arrays(sp["price"], sp["ts"])

        # baseline: A2 entry + always-HOLD-to-N (no rules, just hold to window)
        t1 = time.perf_counter()
        base = evaluate_combined(
            entry_net, None, sp["state"], sp["valid_actions"],
            sp["signals"], sp["price"], aux, sp["regime_id"],
            fee=fee, N=N, use_rl_exits=False)
        print(f"  always-HOLD-to-N baseline ({split}) eval in {time.perf_counter()-t1:.1f}s")

        t1 = time.perf_counter()
        combo = evaluate_combined(
            entry_net, exit_nets, sp["state"], sp["valid_actions"],
            sp["signals"], sp["price"], aux, sp["regime_id"],
            fee=fee, N=N, use_rl_exits=True)
        print(f"  combined  ({split}) eval in {time.perf_counter()-t1:.1f}s")

        _print_summary(f"{split.upper()}  always-HOLD-to-N ({entry_tag} entry, no exits)", base)
        _print_summary(f"{split.upper()}  combined ({entry_tag} entry + B5_fix{N} exits)",
                        combo, baseline=base)

        results[split] = dict(baseline=base, combined=combo)

    # ── final summary ───────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  GROUP C2 — RESULT TABLE  (window N={N})\n{'='*78}")
    print(f"\n  {'split':<6}  {'baseline (always-HOLD-to-N)':>30}  {'combined ('+entry_tag+'+B5)':>22}  {'ΔSharpe':>10}  {'Δeq':>9}")
    print("  " + "─" * 90)
    for split in ("val", "test"):
        r = results[split]
        d_sharpe = r["combined"]["sharpe"] - r["baseline"]["sharpe"]
        d_eq     = r["combined"]["equity_final"] - r["baseline"]["equity_final"]
        print(f"  {split:<6}  {r['baseline']['sharpe']:>+22.3f} ({r['baseline']['equity_final']:.3f})  "
              f"{r['combined']['sharpe']:>+13.3f} ({r['combined']['equity_final']:.3f})  "
              f"{d_sharpe:>+10.3f}  {d_eq*100:>+8.2f}%")

    # ── reference: A2 alone with rule-based exits (the production-relevant number)
    print(f"\n  Reference: A2 alone with rule-based exits (uncapped Group A simulator):")
    print(f"    val  Sharpe +7.295  eq 1.398")
    print(f"    test Sharpe +3.776  eq 1.127")

    out = CACHE / f"{ticker}_group{out_tag}_summary.json"
    payload = dict(ticker=ticker, fee=fee, entry_tag=entry_tag, N=N,
                    results={s: dict(baseline={k: v for k, v in results[s]["baseline"].items() if k != "actions"},
                                       combined={k: v for k, v in results[s]["combined"].items() if k != "actions"})
                              for s in ("val", "test")})
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  → {out.name}")
    print(f"\n  total time {time.perf_counter()-t0:.1f}s")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--window-n", type=int, default=120, dest="window_n")
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--entry-tag", default="A2", dest="entry_tag")
    args = ap.parse_args()
    run(args.ticker, N=args.window_n, fee=args.fee, entry_tag=args.entry_tag)
