"""
Walk-forward validation of C2_fix240 across the 6 RL folds.

Evaluates three configurations per fold:
  1. A2 + rule-based exits  (production target, original Group A simulator)
  2. A2 + always-HOLD-to-240 (B5 baseline; no exit decisions)
  3. A2 + B5_fix240 RL exits (the C2_fix240 stack)

Pre-existing trained policies (no retraining): A2 entry + 9× B5_fix240_fee0_S{k}.

Optional ablation/tuning flags (used for follow-up tests after the audit):
  --ablate-actions "5"      Comma-separated action indices (1..9) to mask out
                            of A2's action space at every bar. Index → strategy:
                            1=S1_VolDir, 2=S2_Funding, 3=S3_BBRevert,
                            4=S4_MACDTrend, 5=S6_TwoSignal, 6=S7_OIDiverg,
                            7=S8_TakerFlow, 8=S10_Squeeze, 9=S12_VWAPVol
                            NO_TRADE (idx 0) is always preserved.

  --tp-scale 0.85           Multiplies base_tp_pct of the 5 trend strategies
                            (S1, S4, S6, S8, S10) by this factor at runtime.
                            Patches EXECUTION_CONFIG in-process before
                            _build_exit_arrays is called.

  --out-tag "test1_ablate"  Output JSON suffix → cache/btc_groupC2_walkforward_<tag>.json

Per-fold output also includes BTC buy-and-hold return and A2 long-only / short-only
PnL split (extracted via the new trade_dirs field returned by evaluate_policy).

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

Run:
  python3 -m models.group_c2_walkforward                              # baseline
  python3 -m models.group_c2_walkforward --ablate-actions "5"         # ablate S6
  python3 -m models.group_c2_walkforward --tp-scale 0.85              # tighter TP
  python3 -m models.group_c2_walkforward --ablate-actions "5,8" --tp-scale 0.85
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

# Trend strategies that have trail_after_breakeven=True in EXECUTION_CONFIG
# (these are the ones whose TP is reachable; mean-reversion strategies have
# tight TP already so we exclude them from --tp-scale).
TREND_STRATS_FOR_TP_SCALE = ["S1_VolDir", "S4_MACDTrend", "S6_TwoSignal",
                                "S8_TakerFlow", "S10_Squeeze"]


def _build_valid_mask_override(ablate_actions: list) -> np.ndarray:
    """Build a (10,) bool mask: True everywhere except indices in ablate_actions.
    Action 0 (NO_TRADE) is always True.

    ablate_actions: list of int indices in [1..9] to mask (S1=1, …, S12=9).
    """
    mask = np.ones(10, dtype=np.bool_)
    for idx in ablate_actions:
        if 1 <= idx <= 9:
            mask[idx] = False
    mask[0] = True   # NO_TRADE always preserved
    return mask


def _patch_exec_config_tp(scale: float):
    """Multiply base_tp_pct on the trend strategies' ComboExit by `scale`,
    in-place. Returns a dict of original values for restore."""
    from execution.config import EXECUTION_CONFIG
    saved = {}
    for key in TREND_STRATS_FOR_TP_SCALE:
        cfg = EXECUTION_CONFIG[key]
        saved[key] = cfg.exit.base_tp
        cfg.exit.base_tp = cfg.exit.base_tp * scale
    return saved


def _restore_exec_config_tp(saved: dict):
    from execution.config import EXECUTION_CONFIG
    for key, original in saved.items():
        EXECUTION_CONFIG[key].exit.base_tp = original


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


def run(ticker: str = "btc", N: int = 240, fee: float = 0.0,
         ablate_actions: list = None, tp_scale: float = 1.0,
         out_tag: str = None, run_b5_eval: bool = True):
    """
    ablate_actions : list of action indices [1..9] to mask out of A2's action space
    tp_scale       : multiply base_tp_pct of the 5 trend strategies by this
    out_tag        : output JSON filename suffix (default: 'fix{N}')
    run_b5_eval    : if False, skip the A2+B5 RL exit and no-exit baselines
                     (saves ~50% time when only the rule-based eval is needed)
    """
    t_start = time.perf_counter()
    if ablate_actions is None: ablate_actions = []
    out_tag = out_tag or f"fix{N}"
    valid_mask_override = _build_valid_mask_override(ablate_actions) if ablate_actions else None

    print(f"\n{'='*100}\n  WALK-FORWARD — A2 entry + B5_fix{N} exits across {N_FOLDS} RL folds  "
          f"(fee={fee:.4f})\n{'='*100}")
    if ablate_actions:
        from models.dqn_rollout import STRAT_KEYS as _SK
        names = [_SK[i-1] for i in ablate_actions if 1 <= i <= 9]
        print(f"  ABLATION: actions {ablate_actions} → strategies {names}")
    if abs(tp_scale - 1.0) > 1e-9:
        print(f"  TP-SCALE: trend strategies {TREND_STRATS_FOR_TP_SCALE} × {tp_scale}")
    print(f"  Output tag: {out_tag}")

    # ── load policies & data ────────────────────────────────────────────────
    print(f"  Loading A2 entry policy ...")
    entry_net = _load_a2_policy(ticker)
    if run_b5_eval:
        print(f"  Loading 9× B5_fix{N}_fee0_S0..8 ...")
        exit_nets = _load_b5_policies(ticker, N=N)
    else:
        exit_nets = None

    print(f"  Loading + concatenating full RL period (train + val + test) ...")
    arr = _load_full_rl_period(ticker)
    n_full = len(arr["state"])
    print(f"  Full RL state shape: {arr['state'].shape}  (expected ~283k bars)")

    # ── apply tp-scale patch BEFORE _build_exit_arrays ──────────────────────
    saved_tp = None
    if abs(tp_scale - 1.0) > 1e-9:
        saved_tp = _patch_exec_config_tp(tp_scale)
        print(f"  patched base_tp_pct in EXECUTION_CONFIG: "
              f"{ {k: f'{saved_tp[k]:.4f}→{saved_tp[k]*tp_scale:.4f}' for k in saved_tp} }")

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

        # BTC buy-and-hold return per fold
        btc_return = float(sub["price"][-1] / sub["price"][0] - 1.0)

        # 1. A2 + rule-based exits (uncapped Group A simulator)
        t1 = time.perf_counter()
        rule = evaluate_a2_rule_based(
            entry_net, sub["state"], sub["valid_actions"],
            sub["signals"], sub["price"],
            tp_f, sl_f, tr_f, tab_f, be_f, ts_f, fee=fee,
            valid_mask_override=valid_mask_override)
        rule_t = time.perf_counter() - t1

        # Long-only / short-only PnL split (rule-based)
        rule_pnls  = np.array(rule.get("trade_pnls", []), dtype=np.float64)
        rule_dirs  = np.array(rule.get("trade_dirs",  []), dtype=np.int64)
        rule_long_pnl  = float(rule_pnls[rule_dirs ==  1].sum()) if len(rule_pnls) else 0.0
        rule_short_pnl = float(rule_pnls[rule_dirs == -1].sum()) if len(rule_pnls) else 0.0
        n_long  = int((rule_dirs ==  1).sum()) if len(rule_dirs) else 0
        n_short = int((rule_dirs == -1).sum()) if len(rule_dirs) else 0

        if run_b5_eval:
            t1 = time.perf_counter()
            nohld = _evaluate_a2_b5(
                entry_net, None, sub["state"], sub["valid_actions"], sub["signals"],
                sub["price"], aux_f, sub["regime_id"], fee=fee, N=N, use_rl=False)
            nohld_t = time.perf_counter() - t1
            t1 = time.perf_counter()
            comb = _evaluate_a2_b5(
                entry_net, exit_nets, sub["state"], sub["valid_actions"], sub["signals"],
                sub["price"], aux_f, sub["regime_id"], fee=fee, N=N, use_rl=True)
            comb_t = time.perf_counter() - t1
        else:
            nohld = dict(sharpe=0.0, equity=1.0, max_dd=0.0, n_trades=0,
                          win_rate=0.0, rl_exits=0, window_exits=0, eod_exits=0)
            comb  = dict(sharpe=0.0, equity=1.0, max_dd=0.0, n_trades=0,
                          win_rate=0.0, rl_exits=0, window_exits=0, eod_exits=0)
            nohld_t = comb_t = 0.0

        print(f"    BTC return    :  {btc_return*100:>+6.2f}%   long={n_long:>3}  short={n_short:>3}")
        print(f"    rule-based    :  Sharpe {rule['sharpe']:>+7.3f}  trades {rule['n_trades']:>4}  "
              f"win {rule['win_rate']*100:>4.1f}%  eq {rule['equity_final']:.3f}  dd {rule['max_dd']*100:>+5.2f}%  "
              f"longPnL {rule_long_pnl*100:>+6.2f}%  shortPnL {rule_short_pnl*100:>+6.2f}%  [{rule_t:.1f}s]")
        if run_b5_eval:
            print(f"    no-exits      :  Sharpe {nohld['sharpe']:>+7.3f}  trades {nohld['n_trades']:>4}  "
                  f"win {nohld['win_rate']*100:>4.1f}%  eq {nohld['equity']:.3f}  dd {nohld['max_dd']*100:>+5.2f}%  [{nohld_t:.1f}s]")
            print(f"    A2 + B5       :  Sharpe {comb['sharpe']:>+7.3f}  trades {comb['n_trades']:>4}  "
                  f"win {comb['win_rate']*100:>4.1f}%  eq {comb['equity']:.3f}  dd {comb['max_dd']*100:>+5.2f}%  RL_exit {comb['rl_exits']:>3}  [{comb_t:.1f}s]")

        rows.append(dict(
            fold=i+1, n_bars=int(b-a), date_a=date_a, date_b=date_b,
            in_sample=("yes" if a_pq < 281_440 else
                        "partial" if a_pq < 332_307 else "no"),
            btc_return=btc_return,
            rule_sharpe=rule['sharpe'], rule_trades=rule['n_trades'],
            rule_eq=rule['equity_final'], rule_dd=rule['max_dd'],
            rule_winrate=rule['win_rate'],
            rule_n_long=n_long, rule_n_short=n_short,
            rule_long_pnl=rule_long_pnl, rule_short_pnl=rule_short_pnl,
            nohld_sharpe=nohld['sharpe'], nohld_trades=nohld['n_trades'],
            nohld_eq=nohld['equity'], nohld_dd=nohld['max_dd'],
            comb_sharpe=comb['sharpe'], comb_trades=comb['n_trades'],
            comb_eq=comb['equity'], comb_dd=comb['max_dd'],
            comb_winrate=comb['win_rate'], comb_rl_exits=comb['rl_exits'],
            delta_vs_rule=comb['sharpe'] - rule['sharpe'],
            delta_vs_nohld=comb['sharpe'] - nohld['sharpe'],
        ))

    # ── summary table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*130}\n  WALK-FORWARD SUMMARY — out_tag={out_tag}, fee={fee:.4f}\n{'='*130}")
    print(f"\n  {'fold':<5} {'in-sample':<10} {'date range':<26}  "
          f"{'BTC ret':>8}  {'A2+rule (Sharpe / eq)':>22}  "
          f"{'long PnL':>9}  {'short PnL':>10}  "
          + (f"{'A2+B5':>15}  {'Δvs rule':>10}" if run_b5_eval else ""))
    print("  " + "-" * 130)
    for r in rows:
        period = f"{r['date_a']}→{r['date_b'][5:]}"
        line = (f"  fold{r['fold']:<2} {r['in_sample']:<10} {period:<26}  "
                f"{r['btc_return']*100:>+7.2f}%  "
                f"{r['rule_sharpe']:>+7.3f} ({r['rule_eq']:.3f})    "
                f"{r['rule_long_pnl']*100:>+7.2f}%  {r['rule_short_pnl']*100:>+8.2f}%")
        if run_b5_eval:
            line += (f"   {r['comb_sharpe']:>+7.3f} ({r['comb_eq']:.3f})  "
                     f"{r['delta_vs_rule']:>+10.3f}")
        print(line)

    # ── aggregate stats ─────────────────────────────────────────────────────
    rule_sharpes = np.array([r['rule_sharpe'] for r in rows])
    comb_sharpes = np.array([r['comb_sharpe'] for r in rows])
    nohld_sharpes = np.array([r['nohld_sharpe'] for r in rows])
    deltas       = np.array([r['delta_vs_rule'] for r in rows])
    btc_returns  = np.array([r['btc_return']  for r in rows])
    long_pnls    = np.array([r['rule_long_pnl']  for r in rows])
    short_pnls   = np.array([r['rule_short_pnl'] for r in rows])

    n_rule_pos = int((rule_sharpes > 0).sum())
    n_comb_pos = int((comb_sharpes > 0).sum())
    n_delta_pos = int((deltas > 0).sum())

    print(f"\n  Aggregate across {N_FOLDS} folds:")
    print(f"    A2 + rule-based  mean Sharpe = {rule_sharpes.mean():+.3f}  median = {np.median(rule_sharpes):+.3f}  "
          f"({n_rule_pos}/{N_FOLDS} folds positive)")
    print(f"    BTC return       mean = {btc_returns.mean()*100:+.2f}%  range = [{btc_returns.min()*100:+.2f}%, "
          f"{btc_returns.max()*100:+.2f}%]")
    print(f"    Long PnL/fold    mean = {long_pnls.mean()*100:+.3f}%   total {long_pnls.sum()*100:+.2f}%")
    print(f"    Short PnL/fold   mean = {short_pnls.mean()*100:+.3f}%  total {short_pnls.sum()*100:+.2f}%")
    if run_b5_eval:
        print(f"    A2 + B5_fix{N}    mean Sharpe = {comb_sharpes.mean():+.3f}  median = {np.median(comb_sharpes):+.3f}  "
              f"({n_comb_pos}/{N_FOLDS} folds positive)")
        print(f"    A2 + no-exits    mean Sharpe = {nohld_sharpes.mean():+.3f}")
        print(f"    Δ (B5 − rule)    mean = {deltas.mean():+.3f}  median = {np.median(deltas):+.3f}")

    # restore patched config
    if saved_tp is not None:
        _restore_exec_config_tp(saved_tp)
        print(f"\n  restored EXECUTION_CONFIG.base_tp_pct to original values")

    out = CACHE / f"{ticker}_groupC2_walkforward_{out_tag}.json"
    out.write_text(json.dumps(dict(
        ticker=ticker, N=N, fee=fee, n_folds=N_FOLDS,
        ablate_actions=ablate_actions, tp_scale=tp_scale,
        rows=rows,
        summary=dict(
            rule_mean=float(rule_sharpes.mean()), rule_median=float(np.median(rule_sharpes)),
            comb_mean=float(comb_sharpes.mean()), comb_median=float(np.median(comb_sharpes)),
            nohld_mean=float(nohld_sharpes.mean()),
            delta_mean=float(deltas.mean()), delta_std=float(deltas.std()),
            btc_return_mean=float(btc_returns.mean()),
            long_pnl_total=float(long_pnls.sum()),
            short_pnl_total=float(short_pnls.sum()),
            n_rule_positive=n_rule_pos, n_comb_positive=n_comb_pos,
            n_delta_positive=n_delta_pos,
        ),
    ), indent=2, default=str))
    print(f"\n  → {out.name}")
    print(f"  total time {time.perf_counter()-t_start:.1f}s")
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--window-n", type=int, default=240, dest="window_n")
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--ablate-actions", default="", dest="ablate_actions",
                     help="comma-separated action indices in [1..9] to mask "
                          "(e.g. '5' for S6_TwoSignal, '5,8' for S6+S10)")
    ap.add_argument("--tp-scale", type=float, default=1.0, dest="tp_scale",
                     help="scale base_tp_pct of trend strategies S1/S4/S6/S8/S10 by this factor")
    ap.add_argument("--out-tag", default=None, dest="out_tag",
                     help="output JSON suffix (default: 'fix{N}')")
    ap.add_argument("--no-b5", action="store_true", dest="no_b5",
                     help="skip A2+B5 RL exit and no-exit baselines (saves time)")
    args = ap.parse_args()

    # parse ablate-actions
    ablate = []
    if args.ablate_actions.strip():
        ablate = [int(x.strip()) for x in args.ablate_actions.split(",") if x.strip()]

    run(args.ticker, N=args.window_n, fee=args.fee,
         ablate_actions=ablate, tp_scale=args.tp_scale,
         out_tag=args.out_tag, run_b5_eval=not args.no_b5)
