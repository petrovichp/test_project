"""
Z5.1 — Out-of-distribution stress test.

Three stress scenarios applied to the locked val + test splits:

1. **Price inversion** — reflect prices around the period mean
   `p' = 2*mean(p) - p`. Down moves become up moves. Tests symmetry
   of the entry policy (any policy with strong long-bias will collapse).

2. **Regime shuffle** — randomly permute regime_id labels. Tests whether
   the regime classifier output is actually being used by the policy
   (if randomized regime doesn't hurt performance, the policy ignores
   regime context).

3. **Feature noise** — additive Gaussian σ=0.1 (in standardized state
   units) on the 30 microstructure dims [30:50]. Tests robustness to
   noisy OB features which is realistic in production.

Reports Sharpe degradation vs the same eval at unperturbed split, for
both `VOTE5_v8_H256_DD` and `DISTILL_v8_seed42`.
"""
import json, statistics, time
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.voting_ensemble import _VotePolicy
from models.diagnostics_ab import _simulate_one_trade_fee
from config.cache_paths import POLICIES, STATE, PREDS, RESULTS

SEEDS = [42, 7, 123, 0, 99]


def load_v8_nets():
    nets = []
    for s in SEEDS:
        n = DuelingDQN(52, 12, 256)
        n.load_state_dict(torch.load(POLICIES / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt",
                                       map_location="cpu"))
        n.eval(); nets.append(n)
    return nets


def load_distill_net():
    n = DuelingDQN(52, 12, 256)
    n.load_state_dict(torch.load(POLICIES / "btc_dqn_policy_DISTILL_v8_seed42.pt",
                                   map_location="cpu"))
    n.eval(); return n


def run_eval(policy_fn, state, valid, signals, prices, atr, atr_median, ts_arr=None):
    tp, sl, trail, tab, be, ts_bars = _build_exit_arrays(prices, atr, atr_median)
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    n_trades = 0
    pol = policy_fn()
    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False
        action = pol(s_t, valid_t)
        if action == 0:
            t += 1; continue
        k = action - 1
        direction = int(signals[t, k])
        if direction == 0:
            t += 1; continue
        pnl, n_held = _simulate_one_trade_fee(
            prices, t + 1, direction,
            float(tp[t, k]), float(sl[t, k]),
            float(trail[t, k]), float(tab[t, k]),
            float(be[t, k]),   int(ts_bars[t, k]),
            0, 0.0,
        )
        t_close = t + 1 + n_held
        if t_close >= n_bars: t_close = n_bars - 1
        eq_arr[t:t_close + 1] = equity
        equity *= (1.0 + float(pnl))
        eq_arr[t_close + 1:] = equity
        if t_close == n_bars - 1: eq_arr[-1] = equity
        peak = max(peak, equity); last_pnl = float(pnl)
        n_trades += 1
        t = t_close + 1
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return dict(sharpe=float(sharpe), equity=float(equity), n_trades=n_trades)


def stress_inversion(sp):
    """Mirror prices around period mean."""
    new_sp = {k: v.copy() for k, v in sp.items() if k != 'prices_inverted'}
    p_mean = float(sp["price"].mean())
    new_sp["price"] = 2 * p_mean - sp["price"]
    return new_sp


def stress_regime_shuffle(sp, rng):
    """Random permutation of regime_id labels."""
    new_sp = {k: v.copy() for k, v in sp.items()}
    new_sp["regime_id"] = rng.permutation(sp["regime_id"])
    return new_sp


def stress_feature_noise(sp, rng, sigma=0.1):
    """Add Gaussian noise σ to dims [30:50] of state."""
    new_sp = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in sp.items()}
    noise = rng.normal(0, sigma, size=(sp["state"].shape[0], 20)).astype(np.float32)
    new_sp["state"] = sp["state"].copy()
    new_sp["state"][:, 30:50] += noise
    return new_sp


def main():
    t0 = time.perf_counter()
    vol = np.load(PREDS / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    rng = np.random.default_rng(20260512)

    sp_v = dict(np.load(STATE / "btc_dqn_state_val_v8_s11s13.npz"))
    sp_t = dict(np.load(STATE / "btc_dqn_state_test_v8_s11s13.npz"))

    v8_nets = load_v8_nets()
    distill_net = load_distill_net()

    policies = [
        ("VOTE5_v8_H256_DD",     lambda: _VotePolicy(v8_nets,    mode="plurality")),
        ("DISTILL_v8_seed42",    lambda: _VotePolicy([distill_net], mode="plurality")),
    ]
    splits = [("val", sp_v), ("test", sp_t)]
    stresses = [
        ("baseline",          lambda sp: sp),
        ("inverted",          lambda sp: stress_inversion(sp)),
        ("regime_shuffle",    lambda sp: stress_regime_shuffle(sp, rng)),
        ("feature_noise_0.1", lambda sp: stress_feature_noise(sp, rng, 0.1)),
    ]

    print(f"\n{'='*120}\n  Z5.1 — OOD stress tests\n{'='*120}\n")
    print(f"  {'policy':<28} {'split':<5} {'stress':<20} "
          f"{'Sharpe':>9} {'Δ vs base':>10} {'eq':>8} {'trades':>7}")
    results = []
    for pol_name, pol_make in policies:
        baseline_sh = {}
        for split_name, sp in splits:
            for stress_name, perturb in stresses:
                stressed = perturb(sp)
                r = run_eval(pol_make, stressed["state"], stressed["valid_actions"],
                              stressed["signals"], stressed["price"], stressed["atr"],
                              atr_median)
                if stress_name == "baseline":
                    baseline_sh[split_name] = r["sharpe"]
                    delta_str = ""
                else:
                    delta = r["sharpe"] - baseline_sh[split_name]
                    delta_str = f"{delta:+.3f}"
                print(f"  {pol_name:<28} {split_name:<5} {stress_name:<20} "
                      f"{r['sharpe']:>+9.3f} {delta_str:>10} {r['equity']:>8.3f} {r['n_trades']:>7}")
                results.append(dict(
                    policy=pol_name, split=split_name, stress=stress_name,
                    sharpe=r["sharpe"], equity=r["equity"], n_trades=r["n_trades"],
                    delta_vs_baseline=(r["sharpe"] - baseline_sh[split_name]) if stress_name != "baseline" else 0.0,
                ))

    out = RESULTS / "z5_ood_stress.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
