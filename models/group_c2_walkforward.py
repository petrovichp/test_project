"""
Walk-forward validation of C2_fix240 across the 6 RL folds.

Evaluates three configurations per fold:
  1. A2 + rule-based exits  (production target, original Group A simulator)
  2. A2 + always-HOLD-to-240 (B5 baseline; no exit decisions)
  3. A2 + B5_fix240 RL exits (the C2_fix240 stack)

Pre-existing trained policies (no retraining): A2 entry + 9× B5_fix240_fee0_S{k}.

Folds (matches walk_forward.py):
  fold 1: bars 101,440..148,635   2025-09-19 → 2025-10-22   (in DQN-train)
  fold 2: bars 148,635..195,830   2025-10-22 → 2025-12-15   (in DQN-train)
  fold 3: bars 195,830..243,025   2025-12-15 → 2026-01-17   (in DQN-train)
  fold 4: bars 243,025..290,220   2026-01-17 → 2026-02-19   (mostly DQN-train, into val)
  fold 5: bars 290,220..337,415   2026-02-19 → 2026-03-23   (mostly DQN-val)
  fold 6: bars 337,415..384,614   2026-03-23 → 2026-04-25   (DQN-test)

Note: B5 was trained on DQN-train only. Folds 1-3 are full in-sample;
fold 4 partially in-sample; folds 5-6 are out-of-sample. We expect folds 1-3
Sharpe ≥ folds 5-6 if the policy is at all generalizing (and worse if overfit).

Run: python3 -m models.group_c2_walkforward [ticker] [--window-n 240]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax
from models.dqn_rollout    import _build_exit_arrays, STRAT_KEYS
from models.exit_dqn_fixed import (FixedExitDQN, simulate_fixed_episode,
                                     precompute_aux_arrays, EXIT_STATE_DIM,
                                     N_ACTIONS, _AlwaysHold, _GreedyFixed)
from models.dqn_selector   import evaluate_policy as evaluate_a2_rule_based

CACHE = ROOT / "cache"
WARMUP = 1440
RL_START_REL = 100_000
RL_END_REL   = 383_174
N_FOLDS      = 6


def _load_a2_policy(ticker: str = "btc") -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / f"{ticker}_dqn_policy_A2.pt",
                                     map_location="cpu"))
    net.eval()
    return net


def _load_b5_policies(ticker: str, N: int = 240) -> list:
    nets = []
    for k in range(len(STRAT_KEYS)):
        path = CACHE / f"{ticker}_exit_dqn_fixed_policy_B5_fix{N}_fee0_S{k}.pt"
        if not path.exists():
            print(f"  ! missing {path.name} → fallback to always-HOLD")
            nets.append(None); continue
        net = FixedExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=96)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def _load_full_rl_period(ticker: str = "btc"):
    """Concatenate train + val + test arrays into one contiguous RL period."""
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"{ticker}_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr",
                     "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    out = {k: np.concatenate(arrs[k], axis=0) for k in arrs}
    return out


def _fold_boundaries():
    """Returns list of (start, end) in pq_use index space (subtracted WARMUP)."""
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    bounds = []
    for i in range(N_FOLDS):
        a = RL_START_REL + i * fold_size
        b = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        bounds.append((a, b))
    return bounds


# ── evaluator: A2 + always-HOLD-to-N OR A2 + B5 RL exits (uses fixed simulator) ──

def _evaluate_a2_b5(entry_net, exit_nets, base_state, valid, signals, prices,
                     aux, regime_id, fee, N, use_rl):
    n_bars = len(base_state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []
    rl_exits = window_exits = eod_exits = 0

    hold_pol = _AlwaysHold()
    if use_rl:
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
        if action == 0:
            t += 1; continue
        k = action - 1
        direction = int(signals[t, k])
        if direction == 0:
            t += 1; continue
        pol = exit_pols[k] if use_rl else hold_pol
        pnl, n_held, reason = simulate_fixed_episode(
            base_state, prices, aux, regime_id,
            entry_bar=t + 1, direction=direction, fee=fee, N=N,
            policy_fn=pol, transitions_out=None,
        )
        if reason == "INVALID":
            t += 1; continue
        t_close = t + 1 + n_held
        if t_close >= n_bars: t_close = n_bars - 1
        eq_arr[t:t_close + 1] = equity
        equity *= (1.0 + pnl)
        eq_arr[t_close + 1:] = equity
        peak = max(peak, equity); last_pnl = pnl
        trade_pnls.append(pnl)
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
    return dict(
        sharpe=float(sharpe), equity=float(equity), max_dd=max_dd,
        n_trades=int(len(pnls_arr)), win_rate=float(win_rate),
        rl_exits=rl_exits, window_exits=window_exits, eod_exits=eod_exits,
    )


def run(ticker: str = "btc", N: int = 240, fee: float = 0.0):
    t_start = time.perf_counter()
    print(f"\n{'='*100}\n  WALK-FORWARD — A2 entry + B5_fix{N} exits across {N_FOLDS} RL folds  "
          f"(fee={fee:.4f})\n{'='*100}")

    # ── load policies & data ────────────────────────────────────────────────
    print(f"  Loading A2 entry + 9× B5_fix{N}_fee0_S0..8 ...")
    entry_net = _load_a2_policy(ticker)
    exit_nets = _load_b5_policies(ticker, N=N)

    print(f"  Loading + concatenating full RL period (train + val + test) ...")
    arr = _load_full_rl_period(ticker)
    n_full = len(arr["state"])
    print(f"  Full RL state shape: {arr['state'].shape}  (expected ~283k bars)")

    # vol median for ATR-scaled rule exits (used in A2 + rule baseline)
    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_med = float(vol["atr_train_median"])
    tp_full, sl_full, trail_full, tab_full, be_full, ts_full = _build_exit_arrays(
        arr["price"], arr["atr"], atr_med)

    # precompute aux arrays for the full period (for B5 simulator)
    aux_full = precompute_aux_arrays(arr["price"], arr["ts"])

    # fold boundaries in pq_use index space (relative to bar 0 of WARMUP-sliced array)
    # Our concatenated array starts at the WARMUP-sliced position equal to RL_START_REL.
    # So fold start `a` (in pq_use indexing) maps to (a - RL_START_REL) in our array.
    folds = _fold_boundaries()
    print(f"\n  Fold boundaries (absolute bar / index in concatenated array):")
    for i, (a, b) in enumerate(folds):
        print(f"    fold{i+1}  pq_use [{a:>7,}, {b:>7,})  → arr_idx [{a-RL_START_REL:>6,}, "
              f"{b-RL_START_REL:>6,})  ({b-a:,} bars)")

    # ── evaluate per fold ───────────────────────────────────────────────────
    rows = []
    for i, (a_pq, b_pq) in enumerate(folds):
        a = a_pq - RL_START_REL
        b = b_pq - RL_START_REL
        sub = {k: arr[k][a:b] for k in arr}

        # local exit arrays for the rule-based baseline
        tp_f, sl_f, tr_f, tab_f, be_f, ts_f = (
            tp_full[a:b], sl_full[a:b], trail_full[a:b],
            tab_full[a:b], be_full[a:b], ts_full[a:b],
        )
        aux_f = {k: (aux_full[k][a:b] if isinstance(aux_full[k], np.ndarray)
                      else aux_full[k]) for k in aux_full}
        # but the medians/iqrs are scalars, those are fine to keep

        date_a = datetime.fromtimestamp(int(sub["ts"][0])).strftime("%Y-%m-%d")
        date_b = datetime.fromtimestamp(int(sub["ts"][-1])).strftime("%Y-%m-%d")
        print(f"\n  ── fold {i+1}  bars {a_pq:,}..{b_pq:,}  {date_a} → {date_b} "
              f"({b-a:,} bars) ──")

        # 1. A2 + rule-based exits (uncapped Group A simulator)
        t1 = time.perf_counter()
        rule = evaluate_a2_rule_based(
            entry_net, sub["state"], sub["valid_actions"],
            sub["signals"], sub["price"],
            tp_f, sl_f, tr_f, tab_f, be_f, ts_f, fee=fee)
        rule_t = time.perf_counter() - t1

        # 2. A2 + always-HOLD-to-N (no exits)
        t1 = time.perf_counter()
        nohld = _evaluate_a2_b5(
            entry_net, None, sub["state"], sub["valid_actions"], sub["signals"],
            sub["price"], aux_f, sub["regime_id"], fee=fee, N=N, use_rl=False)
        nohld_t = time.perf_counter() - t1

        # 3. A2 + B5_fix240 RL exits
        t1 = time.perf_counter()
        comb = _evaluate_a2_b5(
            entry_net, exit_nets, sub["state"], sub["valid_actions"], sub["signals"],
            sub["price"], aux_f, sub["regime_id"], fee=fee, N=N, use_rl=True)
        comb_t = time.perf_counter() - t1

        print(f"    rule-based   :  Sharpe {rule['sharpe']:>+7.3f}  trades {rule['n_trades']:>4}  "
              f"win {rule['win_rate']*100:>4.1f}%  eq {rule['equity_final']:.3f}  dd {rule['max_dd']*100:>+5.2f}%  [{rule_t:.1f}s]")
        print(f"    no-exits     :  Sharpe {nohld['sharpe']:>+7.3f}  trades {nohld['n_trades']:>4}  "
              f"win {nohld['win_rate']*100:>4.1f}%  eq {nohld['equity']:.3f}  dd {nohld['max_dd']*100:>+5.2f}%  [{nohld_t:.1f}s]")
        print(f"    A2 + B5      :  Sharpe {comb['sharpe']:>+7.3f}  trades {comb['n_trades']:>4}  "
              f"win {comb['win_rate']*100:>4.1f}%  eq {comb['equity']:.3f}  dd {comb['max_dd']*100:>+5.2f}%  RL_exit {comb['rl_exits']:>3}  [{comb_t:.1f}s]")
        print(f"    Δ vs rule    :  ΔSharpe {comb['sharpe']-rule['sharpe']:>+7.3f}   "
              f"Δeq {(comb['equity']-rule['equity_final'])*100:>+5.2f}pp")

        rows.append(dict(
            fold=i+1, n_bars=int(b-a), date_a=date_a, date_b=date_b,
            in_sample=("yes" if a_pq < 281_440 else
                        "partial" if a_pq < 332_307 else "no"),
            rule_sharpe=rule['sharpe'], rule_trades=rule['n_trades'],
            rule_eq=rule['equity_final'], rule_dd=rule['max_dd'],
            rule_winrate=rule['win_rate'],
            nohld_sharpe=nohld['sharpe'], nohld_trades=nohld['n_trades'],
            nohld_eq=nohld['equity'], nohld_dd=nohld['max_dd'],
            comb_sharpe=comb['sharpe'], comb_trades=comb['n_trades'],
            comb_eq=comb['equity'], comb_dd=comb['max_dd'],
            comb_winrate=comb['win_rate'], comb_rl_exits=comb['rl_exits'],
            delta_vs_rule=comb['sharpe'] - rule['sharpe'],
            delta_vs_nohld=comb['sharpe'] - nohld['sharpe'],
        ))

    # ── summary table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*120}\n  WALK-FORWARD SUMMARY — A2 entry + B5_fix{N} exits, fee={fee:.4f}\n{'='*120}")
    print(f"\n  {'fold':<5} {'in-sample':<10} {'date range':<26}  "
          f"{'A2+rule':>15}  {'A2+B5':>15}  {'Δvs rule':>10}  {'A2+noexit':>13}  {'Δvs noexit':>11}")
    print("  " + "-" * 120)
    for r in rows:
        period = f"{r['date_a']}→{r['date_b'][5:]}"
        print(f"  fold{r['fold']:<2} {r['in_sample']:<10} {period:<26}  "
              f"{r['rule_sharpe']:>+7.3f} ({r['rule_eq']:.3f})  "
              f"{r['comb_sharpe']:>+7.3f} ({r['comb_eq']:.3f})  "
              f"{r['delta_vs_rule']:>+10.3f}  "
              f"{r['nohld_sharpe']:>+7.3f}({r['nohld_eq']:.3f})  "
              f"{r['delta_vs_nohld']:>+11.3f}")

    # ── aggregate stats ─────────────────────────────────────────────────────
    rule_sharpes = np.array([r['rule_sharpe'] for r in rows])
    comb_sharpes = np.array([r['comb_sharpe'] for r in rows])
    nohld_sharpes = np.array([r['nohld_sharpe'] for r in rows])
    deltas       = np.array([r['delta_vs_rule'] for r in rows])

    n_rule_pos = int((rule_sharpes > 0).sum())
    n_comb_pos = int((comb_sharpes > 0).sum())
    n_delta_pos = int((deltas > 0).sum())

    print(f"\n  Aggregate across {N_FOLDS} folds:")
    print(f"    A2 + rule-based  mean Sharpe = {rule_sharpes.mean():+.3f}  median = {np.median(rule_sharpes):+.3f}  "
          f"({n_rule_pos}/{N_FOLDS} folds positive)")
    print(f"    A2 + B5_fix{N}    mean Sharpe = {comb_sharpes.mean():+.3f}  median = {np.median(comb_sharpes):+.3f}  "
          f"({n_comb_pos}/{N_FOLDS} folds positive)")
    print(f"    A2 + no-exits    mean Sharpe = {nohld_sharpes.mean():+.3f}")
    print(f"    Δ (B5 − rule)    mean = {deltas.mean():+.3f}  median = {np.median(deltas):+.3f}  "
          f"({n_delta_pos}/{N_FOLDS} folds where B5 > rule)")
    print(f"    Δ std            = {deltas.std():.3f}")

    # ── decision gate ────────────────────────────────────────────────────────
    print(f"\n  ── DECISION GATE ──")
    if n_comb_pos >= 4 and deltas.mean() > 0:
        print(f"  ✓ C2_fix{N} clears walk-forward gate ({n_comb_pos}/{N_FOLDS} folds positive AND mean Δ > 0)")
        print(f"    → real signal, proceed to seed variance + Path X scoping")
    elif n_comb_pos >= 4 and deltas.mean() < 0:
        print(f"  ◐ C2_fix{N} positive in absolute terms ({n_comb_pos}/{N_FOLDS}) but UNDERPERFORMS rule-based exits on average")
        print(f"    → A2 + rule-based is the better deployment choice")
    elif n_comb_pos < 4:
        print(f"  ✗ C2_fix{N} fails walk-forward ({n_comb_pos}/{N_FOLDS} folds positive)")
        print(f"    → fall back to A2 alone with rule-based exits")

    out = CACHE / f"{ticker}_groupC2_walkforward_fix{N}.json"
    out.write_text(json.dumps(dict(
        ticker=ticker, N=N, fee=fee, n_folds=N_FOLDS, rows=rows,
        summary=dict(
            rule_mean=float(rule_sharpes.mean()), rule_median=float(np.median(rule_sharpes)),
            comb_mean=float(comb_sharpes.mean()), comb_median=float(np.median(comb_sharpes)),
            nohld_mean=float(nohld_sharpes.mean()),
            delta_mean=float(deltas.mean()), delta_std=float(deltas.std()),
            n_rule_positive=n_rule_pos, n_comb_positive=n_comb_pos,
            n_delta_positive=n_delta_pos,
        ),
    ), indent=2, default=str))
    print(f"\n  → {out.name}")
    print(f"\n  total time {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--window-n", type=int, default=240, dest="window_n")
    ap.add_argument("--fee", type=float, default=0.0)
    args = ap.parse_args()
    run(args.ticker, N=args.window_n, fee=args.fee)
