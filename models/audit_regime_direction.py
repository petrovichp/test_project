"""
Three diagnostics replacing the discredited Z5.1 regime_shuffle framing:

  Q1. Corrected regime perturbation — directly mutates state[:, 2:7]
       (the one-hot the model actually sees). Two variants:
        A) row-shuffle: assign each timestep a random other timestep's regime
        B) zero-out: erase regime information entirely

  Q2. Per-fold long-only vs short-only PnL attribution from walk-forward
       trade logs. Tests whether the edge is balanced or directionally biased
       on the actual dataset (not the synthetic price-inversion test).

  Q3. Regime-conditional Sharpe — bucket trades by regime_id at entry and
       compute per-regime Sharpe. Shows where the edge actually lives.

Run on VOTE5_v8_H256_DD (K=5 plurality teacher) and DISTILL_v8_seed42 (single
net deployable). Both deployable baselines from Z5.4 freeze.

Run: python3 -m models.audit_regime_direction
"""
import json, math, statistics, time
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.voting_ensemble import _VotePolicy
from models.diagnostics_ab import _simulate_one_trade_fee
from models.audit_vote5_dd import run_walkforward, run_fold
from config.cache_paths import POLICIES, STATE, PREDS, RESULTS

SEEDS = [42, 7, 123, 0, 99]
REGIME_NAMES = ["calm", "trend_up", "trend_down", "ranging", "chop"]


def load_v8_nets():
    out = []
    for s in SEEDS:
        n = DuelingDQN(52, 12, 256)
        n.load_state_dict(torch.load(
            POLICIES / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt", map_location="cpu"))
        n.eval(); out.append(n)
    return out


def load_distill_net():
    n = DuelingDQN(52, 12, 256)
    n.load_state_dict(torch.load(
        POLICIES / f"btc_dqn_policy_DISTILL_v8_seed42.pt", map_location="cpu"))
    n.eval(); return n


def _eval_split(policy_fn, sp, atr_median):
    """Replicates the run_eval loop used in z5_ood_stress with explicit state input."""
    state = sp["state"]; valid = sp["valid_actions"]; signals = sp["signals"]
    prices = sp["price"]; atr = sp["atr"]
    tp, sl, trail, tab, be, ts_bars = _build_exit_arrays(prices, atr, atr_median)
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    n_trades = 0; pol = policy_fn(); t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        v_t = valid[t].copy()
        if not v_t.any():
            v_t[0] = True; v_t[1:] = False
        a = pol(s_t, v_t)
        if a == 0:
            t += 1; continue
        k = a - 1
        d = int(signals[t, k])
        if d == 0:
            t += 1; continue
        pnl, n_held = _simulate_one_trade_fee(
            prices, t + 1, d,
            float(tp[t, k]), float(sl[t, k]), float(trail[t, k]),
            float(tab[t, k]), float(be[t, k]), int(ts_bars[t, k]),
            0, 0.0)
        t_close = min(t + 1 + n_held, n_bars - 1)
        eq_arr[t:t_close+1] = equity
        equity *= (1.0 + float(pnl))
        eq_arr[t_close+1:] = equity
        peak = max(peak, equity); last_pnl = float(pnl); n_trades += 1
        t = t_close + 1
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = rets.mean() / rets.std() * math.sqrt(525_960) if rets.std() > 1e-12 else 0.0
    return dict(sharpe=float(sharpe), equity=float(equity), n_trades=n_trades)


def perturb_regime_rowshuffle(sp, rng):
    new = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sp.items()}
    perm = rng.permutation(sp["state"].shape[0])
    new["state"] = sp["state"].copy()
    new["state"][:, 2:7] = sp["state"][perm, 2:7]
    return new


def perturb_regime_zero(sp):
    new = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sp.items()}
    new["state"] = sp["state"].copy()
    new["state"][:, 2:7] = 0.0
    return new


# ── Q1 ────────────────────────────────────────────────────────────────
def q1_regime_perturbation(sp_val, sp_test, atr_median, v8_nets, distill_net):
    print(f"\n{'='*120}\n  Q1 — regime perturbation on state[:, 2:7] (the actual model input)\n{'='*120}\n")
    print(f"  {'policy':<28} {'split':<5} {'perturb':<20} "
          f"{'Sharpe':>9} {'Δ vs base':>10} {'eq':>8} {'trades':>7}")
    rng = np.random.default_rng(20260512)
    policies = [
        ("VOTE5_v8_H256_DD",  lambda: _VotePolicy(v8_nets,    mode="plurality")),
        ("DISTILL_v8_seed42", lambda: _VotePolicy([distill_net], mode="plurality")),
    ]
    splits = [("val", sp_val), ("test", sp_test)]
    perturbs = [
        ("baseline",        lambda sp: sp),
        ("regime_rowshuf",  lambda sp: perturb_regime_rowshuffle(sp, rng)),
        ("regime_zero",     lambda sp: perturb_regime_zero(sp)),
    ]
    rows = []
    for pname, pfn in policies:
        base_sh = {}
        for sname, sp in splits:
            for pertname, pert in perturbs:
                sp_p = pert(sp)
                r = _eval_split(pfn, sp_p, atr_median)
                if pertname == "baseline":
                    base_sh[sname] = r["sharpe"]; ds = ""
                else:
                    d = r["sharpe"] - base_sh[sname]; ds = f"{d:+.3f}"
                print(f"  {pname:<28} {sname:<5} {pertname:<20} "
                      f"{r['sharpe']:>+9.3f} {ds:>10} {r['equity']:>8.3f} {r['n_trades']:>7}")
                rows.append(dict(policy=pname, split=sname, perturb=pertname,
                                 sharpe=r["sharpe"], equity=r["equity"], n_trades=r["n_trades"],
                                 delta=(r["sharpe"] - base_sh[sname]) if pertname != "baseline" else 0.0))
    return rows


# ── Q2 + Q3 ───────────────────────────────────────────────────────────
def _pnl_sharpe(pnl_arr):
    if len(pnl_arr) < 2 or float(np.std(pnl_arr)) < 1e-12:
        return 0.0
    # Per-trade Sharpe is not directly comparable to bar-level Sharpe; report mean/std/n.
    return float(np.mean(pnl_arr)) / float(np.std(pnl_arr)) * math.sqrt(len(pnl_arr))


def q2_long_short_attribution(trades, label):
    print(f"\n  {label} — long/short attribution per fold")
    print(f"  {'fold':<5} {'n_long':>6} {'n_short':>7} {'long_pnl%':>10} {'short_pnl%':>10} "
          f"{'long_Sharpe':>12} {'short_Sharpe':>13}")
    out = []
    folds = sorted(set(t["fold"] for t in trades))
    for f in folds:
        sub = [t for t in trades if t["fold"] == f]
        L = [t for t in sub if t["direction"] == 1]
        S = [t for t in sub if t["direction"] == -1]
        l_pnl = sum(t["pnl"] for t in L) * 100
        s_pnl = sum(t["pnl"] for t in S) * 100
        l_sh = _pnl_sharpe(np.array([t["pnl"] for t in L])) if L else 0.0
        s_sh = _pnl_sharpe(np.array([t["pnl"] for t in S])) if S else 0.0
        print(f"  {f:<5} {len(L):>6} {len(S):>7} "
              f"{l_pnl:>+10.2f} {s_pnl:>+10.2f} {l_sh:>+12.3f} {s_sh:>+13.3f}")
        out.append(dict(fold=int(f), n_long=len(L), n_short=len(S),
                        long_pnl_pct=l_pnl, short_pnl_pct=s_pnl,
                        long_sharpe_trade=l_sh, short_sharpe_trade=s_sh))
    n_L = sum(1 for t in trades if t["direction"] == 1)
    n_S = sum(1 for t in trades if t["direction"] == -1)
    long_pct = n_L / max(1, n_L + n_S) * 100
    L_all = np.array([t["pnl"] for t in trades if t["direction"] == 1])
    S_all = np.array([t["pnl"] for t in trades if t["direction"] == -1])
    print(f"  TOTAL n_long={n_L} ({long_pct:.1f}%)  n_short={n_S} ({100-long_pct:.1f}%)  "
          f"sum_long_pnl={float(L_all.sum())*100:+.2f}%  sum_short_pnl={float(S_all.sum())*100:+.2f}%")
    return out, dict(n_long=int(n_L), n_short=int(n_S),
                     sum_long_pnl_pct=float(L_all.sum())*100,
                     sum_short_pnl_pct=float(S_all.sum())*100,
                     long_pct=float(long_pct))


def q3_regime_conditional(trades, label):
    print(f"\n  {label} — regime-conditional attribution")
    print(f"  {'regime':<12} {'n_trades':>9} {'frac':>6} {'mean_pnl%':>10} "
          f"{'sum_pnl%':>10} {'long_share':>11}")
    out = []
    total = len(trades)
    for r in range(5):
        sub = [t for t in trades if t["regime"] == r]
        if not sub:
            print(f"  {REGIME_NAMES[r]:<12} {'0':>9} {'0.0%':>6} {'—':>10} {'—':>10} {'—':>11}")
            out.append(dict(regime=r, regime_name=REGIME_NAMES[r], n_trades=0))
            continue
        pnl = np.array([t["pnl"] for t in sub])
        nL = sum(1 for t in sub if t["direction"] == 1)
        share = len(sub) / max(1, total) * 100
        ls = nL / len(sub) * 100
        print(f"  {REGIME_NAMES[r]:<12} {len(sub):>9} {share:>5.1f}% "
              f"{float(pnl.mean())*100:>+10.3f} {float(pnl.sum())*100:>+10.2f} {ls:>10.1f}%")
        out.append(dict(regime=r, regime_name=REGIME_NAMES[r], n_trades=len(sub),
                        share_pct=float(share), mean_pnl_pct=float(pnl.mean())*100,
                        sum_pnl_pct=float(pnl.sum())*100, long_share_pct=float(ls)))
    return out


def load_full_v8():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(STATE / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def main():
    t0 = time.perf_counter()
    vol = np.load(PREDS / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    sp_v = dict(np.load(STATE / "btc_dqn_state_val_v8_s11s13.npz"))
    sp_t = dict(np.load(STATE / "btc_dqn_state_test_v8_s11s13.npz"))
    full = load_full_v8()
    v8_nets = load_v8_nets()
    distill_net = load_distill_net()

    results = {}

    # Q1
    results["q1_regime_perturbation"] = q1_regime_perturbation(
        sp_v, sp_t, atr_median, v8_nets, distill_net)

    # Q2 + Q3 require walk-forward trade logs
    print(f"\n{'='*120}\n  Walk-forward trade collection\n{'='*120}\n")
    print(f"  Collecting trades for VOTE5_v8_H256_DD ...")
    _, trades_v8 = run_walkforward(v8_nets, full, atr_median, fee=0.0, with_reason=False)
    print(f"  ... {len(trades_v8)} trades")
    print(f"  Collecting trades for DISTILL_v8_seed42 ...")
    _, trades_dist = run_walkforward([distill_net], full, atr_median, fee=0.0, with_reason=False)
    print(f"  ... {len(trades_dist)} trades")

    print(f"\n{'='*120}\n  Q2 — long/short attribution per fold\n{'='*120}")
    fold_v8, tot_v8 = q2_long_short_attribution(trades_v8, "VOTE5_v8_H256_DD")
    fold_d,  tot_d  = q2_long_short_attribution(trades_dist, "DISTILL_v8_seed42")
    results["q2_long_short"] = dict(
        vote5_v8=dict(per_fold=fold_v8, total=tot_v8),
        distill_v8=dict(per_fold=fold_d, total=tot_d),
    )

    print(f"\n{'='*120}\n  Q3 — regime-conditional attribution\n{'='*120}")
    reg_v8 = q3_regime_conditional(trades_v8, "VOTE5_v8_H256_DD")
    reg_d  = q3_regime_conditional(trades_dist, "DISTILL_v8_seed42")
    results["q3_regime_conditional"] = dict(vote5_v8=reg_v8, distill_v8=reg_d)

    out = RESULTS / "audit_regime_direction.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
