"""
Compare v6 (state-dim 54, includes direction probs) vs v5 baselines.

Trains: BASELINE_FULL_V6_seed{42,7,123,0,99}
Compares:
  - per-seed v5 vs v6 (val, test, WF)
  - VOTE5 plurality v5 vs v6
  - VOTE5_disjoint comparison if v6 disjoint seeds exist

Uses the same evaluation framework as voting_ensemble.py but with state_version
plumbing.
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DQN, masked_argmax
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.voting_ensemble      import (
    _VotePolicy, evaluate_with_policy,
)

CACHE = pathlib.Path("cache")
N_FOLDS = 6


def load_net(tag: str, state_dim: int) -> DQN:
    net = DQN(state_dim, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period(version: str = "v5"):
    suffix = "" if version == "v5" else f"_{version}"
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}{suffix}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_split(policy_fn, split: str, atr_median: float, version: str = "v5"):
    suffix = "" if version == "v5" else f"_{version}"
    sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}{suffix}.npz")
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    out = evaluate_with_policy(policy_fn(), sp["state"], sp["valid_actions"],
                                 sp["signals"], sp["price"], tp, sl, tr, tab, be, ts)
    return dict(sharpe=out["sharpe"], equity=out["equity_final"],
                trades=out["n_trades"])


def eval_walkforward(policy_factory, atr_median: float, full):
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    rows = []
    for i in range(N_FOLDS):
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL; b = b_pq - RL_START_REL
        sub_state   = full["state"][a:b]
        sub_valid   = full["valid_actions"][a:b]
        sub_signals = full["signals"][a:b]
        sub_prices  = full["price"][a:b]
        sub_atr     = full["atr"][a:b]
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub_prices, sub_atr, atr_median)
        out = evaluate_with_policy(policy_factory(), sub_state, sub_valid,
                                     sub_signals, sub_prices,
                                     tp, sl, tr, tab, be, ts)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"],
                          equity=out["equity_final"], trades=out["n_trades"]))
    return rows


# ── single-seed greedy policy wrapper ────────────────────────────────────────

class _SingleGreedy:
    def __init__(self, net): self.net = net
    def __call__(self, s, v):
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool().unsqueeze(0)
            return int(masked_argmax(self.net, sb, vb).item())


SEEDS = [42, 7, 123, 0, 99]
TAG_V5 = {s: ("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS}
TAG_V6 = {s: f"BASELINE_FULL_V6_seed{s}" for s in SEEDS}


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    full_v5 = load_full_rl_period("v5")
    full_v6 = load_full_rl_period("v6")

    print(f"\n{'='*120}\n  V6 vs V5 — direction-probs in state evaluation\n{'='*120}")

    # ── per-seed: v5 single-seed vs v6 single-seed ──────────────────────────
    print(f"\n  Per-seed comparison (single-seed greedy):")
    print(f"  {'seed':<5} {'v5 val':>8} {'v6 val':>8}  {'v5 test':>9} {'v6 test':>9}  "
          f"{'v5 WF':>8} {'v6 WF':>8}  {'v5 fold6':>10} {'v6 fold6':>10}")
    print("-"*120)
    per_seed = {}
    for s in SEEDS:
        net5 = load_net(TAG_V5[s], 50)
        net6 = load_net(TAG_V6[s], 54)
        v5_val  = eval_split(lambda: _SingleGreedy(net5), "val",  atr_median, "v5")
        v6_val  = eval_split(lambda: _SingleGreedy(net6), "val",  atr_median, "v6")
        v5_test = eval_split(lambda: _SingleGreedy(net5), "test", atr_median, "v5")
        v6_test = eval_split(lambda: _SingleGreedy(net6), "test", atr_median, "v6")
        v5_wf   = eval_walkforward(lambda: _SingleGreedy(net5), atr_median, full_v5)
        v6_wf   = eval_walkforward(lambda: _SingleGreedy(net6), atr_median, full_v6)
        v5_wf_mean = statistics.mean([r["sharpe"] for r in v5_wf])
        v6_wf_mean = statistics.mean([r["sharpe"] for r in v6_wf])
        per_seed[s] = dict(
            v5_val=v5_val["sharpe"], v6_val=v6_val["sharpe"],
            v5_test=v5_test["sharpe"], v6_test=v6_test["sharpe"],
            v5_wf=v5_wf_mean, v6_wf=v6_wf_mean,
            v5_wf_per_fold=[r["sharpe"] for r in v5_wf],
            v6_wf_per_fold=[r["sharpe"] for r in v6_wf],
        )
        print(f"  {s:<5} {v5_val['sharpe']:>+8.2f} {v6_val['sharpe']:>+8.2f}  "
              f"{v5_test['sharpe']:>+9.2f} {v6_test['sharpe']:>+9.2f}  "
              f"{v5_wf_mean:>+8.3f} {v6_wf_mean:>+8.3f}  "
              f"{v5_wf[5]['sharpe']:>+10.2f} {v6_wf[5]['sharpe']:>+10.2f}")

    # ── ensemble: VOTE5 plurality v5 vs v6 ──────────────────────────────────
    print(f"\n  K=5 plurality ensemble comparison:")
    nets_v5 = [load_net(TAG_V5[s], 50) for s in SEEDS]
    nets_v6 = [load_net(TAG_V6[s], 54) for s in SEEDS]

    def make_v5(): return _VotePolicy(nets_v5, mode="plurality")
    def make_v6(): return _VotePolicy(nets_v6, mode="plurality")

    v5_val  = eval_split(make_v5, "val",  atr_median, "v5")
    v6_val  = eval_split(make_v6, "val",  atr_median, "v6")
    v5_test = eval_split(make_v5, "test", atr_median, "v5")
    v6_test = eval_split(make_v6, "test", atr_median, "v6")
    v5_wf   = eval_walkforward(make_v5, atr_median, full_v5)
    v6_wf   = eval_walkforward(make_v6, atr_median, full_v6)
    v5_wf_mean = statistics.mean([r["sharpe"] for r in v5_wf])
    v6_wf_mean = statistics.mean([r["sharpe"] for r in v6_wf])
    v5_wf_pos = sum(1 for r in v5_wf if r["sharpe"] > 0)
    v6_wf_pos = sum(1 for r in v6_wf if r["sharpe"] > 0)

    print(f"\n  {'metric':<22} {'BASELINE_VOTE5 (v5)':>22} {'VOTE5_v6 (v6)':>16}  Δ v6-v5")
    print("-"*90)
    metrics = [
        ("val Sharpe",   v5_val['sharpe'],  v6_val['sharpe']),
        ("val equity",   v5_val['equity'],  v6_val['equity']),
        ("val trades",   v5_val['trades'],  v6_val['trades']),
        ("test Sharpe",  v5_test['sharpe'], v6_test['sharpe']),
        ("test equity",  v5_test['equity'], v6_test['equity']),
        ("test trades",  v5_test['trades'], v6_test['trades']),
        ("WF mean Sharpe", v5_wf_mean,      v6_wf_mean),
        ("WF folds positive", v5_wf_pos,    v6_wf_pos),
        ("WF fold 6",    v5_wf[5]['sharpe'], v6_wf[5]['sharpe']),
    ]
    for name, vv5, vv6 in metrics:
        d = vv6 - vv5
        print(f"  {name:<22} {vv5:>22.3f} {vv6:>16.3f}  {d:>+8.3f}")

    print(f"\n  Per-fold WF Sharpe:")
    for fi in range(6):
        vv5 = v5_wf[fi]['sharpe']
        vv6 = v6_wf[fi]['sharpe']
        print(f"    fold {fi+1}: v5={vv5:>+7.2f}  v6={vv6:>+7.2f}  Δ={vv6-vv5:>+7.2f}")

    # save
    out = CACHE / "results" / "eval_v6_vs_v5_results.json"
    out.write_text(json.dumps({
        "per_seed": per_seed,
        "ensemble_v5": dict(val=v5_val, test=v5_test, wf=v5_wf, wf_mean=v5_wf_mean),
        "ensemble_v6": dict(val=v6_val, test=v6_test, wf=v6_wf, wf_mean=v6_wf_mean),
    }, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
