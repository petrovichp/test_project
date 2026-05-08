"""
A1 — Capacity test: evaluate VOTE5 plurality at hidden ∈ {64, 128, 256}.

Compares (per-seed and ensemble):
  - val Sharpe + equity + trades
  - test Sharpe + equity + trades
  - WF mean Sharpe + per-fold + folds positive

Decision criterion:
  - If hidden=128 and/or hidden=256 lifts WF mean Sharpe ≥ +0.5 above
    BASELINE_VOTE5's +10.40 → capacity-bound; expand
  - If flat or negative → signal saturated at this state spec; pivot
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DQN, masked_argmax
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.voting_ensemble      import _VotePolicy, evaluate_with_policy

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS   = [42, 7, 123, 0, 99]


def load_net(tag: str, hidden: int) -> DQN:
    net = DQN(50, 10, hidden=hidden)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"btc_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_split(policy_fn, split: str, atr_median: float):
    sp = np.load(CACHE / f"btc_dqn_state_{split}.npz")
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    out = evaluate_with_policy(policy_fn(), sp["state"], sp["valid_actions"],
                                 sp["signals"], sp["price"], tp, sl, tr, tab, be, ts)
    return dict(sharpe=out["sharpe"], equity=out["equity_final"], trades=out["n_trades"])


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
        out = evaluate_with_policy(policy_factory(), sub_state, sub_valid, sub_signals, sub_prices,
                                     tp, sl, tr, tab, be, ts)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"], equity=out["equity_final"]))
    return rows


class _SingleGreedy:
    def __init__(self, net): self.net = net
    def __call__(self, s, v):
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool().unsqueeze(0)
            return int(masked_argmax(self.net, sb, vb).item())


def tag_for(seed: int, hidden: int) -> str:
    if hidden == 64:
        return "BASELINE_FULL" if seed == 42 else f"BASELINE_FULL_seed{seed}"
    return f"BASELINE_FULL_h{hidden}_seed{seed}"


def eval_pool(hidden: int, atr_median: float, full):
    """Per-seed + K=5 plurality ensemble at given hidden size."""
    nets = [load_net(tag_for(s, hidden), hidden) for s in SEEDS]

    per_seed = {}
    for i, s in enumerate(SEEDS):
        net = nets[i]
        v = eval_split(lambda: _SingleGreedy(net), "val",  atr_median)
        t = eval_split(lambda: _SingleGreedy(net), "test", atr_median)
        wf = eval_walkforward(lambda: _SingleGreedy(net), atr_median, full)
        per_seed[s] = dict(
            val_sharpe=v["sharpe"], val_eq=v["equity"],
            test_sharpe=t["sharpe"], test_eq=t["equity"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    def make_pol(): return _VotePolicy(nets, mode="plurality")
    v = eval_split(make_pol, "val",  atr_median)
    t = eval_split(make_pol, "test", atr_median)
    wf = eval_walkforward(make_pol, atr_median, full)
    ensemble = dict(
        val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
        test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
        wf_per_fold=[r["sharpe"] for r in wf],
        wf_mean=statistics.mean([r["sharpe"] for r in wf]),
        wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
    )
    return per_seed, ensemble


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()

    print(f"\n{'='*120}\n  A1 CAPACITY TEST — hidden ∈ {{64, 128, 256}}\n{'='*120}")

    results = {}
    for hidden in (64, 128, 256):
        print(f"\n  Evaluating hidden={hidden} pool ...")
        per_seed, ens = eval_pool(hidden, atr_median, full)
        results[hidden] = dict(per_seed=per_seed, ensemble=ens)

    # ── per-seed table ──────────────────────────────────────────────────────
    print(f"\n  Per-seed val/test/WF for each hidden size:")
    print(f"  {'seed':<5} " + "".join(
        f"{'h'+str(h)+' val':>8} {'h'+str(h)+' test':>9} {'h'+str(h)+' WF':>8} "
        for h in (64, 128, 256)))
    print("  " + "-"*150)
    for s in SEEDS:
        row = f"  {s:<5} "
        for h in (64, 128, 256):
            r = results[h]["per_seed"][s]
            row += f"{r['val_sharpe']:>+8.2f} {r['test_sharpe']:>+9.2f} {r['wf_mean']:>+8.3f} "
        print(row)

    # ── ensemble comparison ────────────────────────────────────────────────
    print(f"\n  K=5 plurality ensemble comparison:")
    print(f"  {'metric':<22} " + "".join(f"{'h='+str(h):>12} " for h in (64, 128, 256)) + " " * 6 + "Δh128-h64  Δh256-h64")
    print("  " + "-"*120)

    ens_metrics = [
        ("val Sharpe",   "val_sharpe"),
        ("val equity",   "val_eq"),
        ("test Sharpe",  "test_sharpe"),
        ("test equity",  "test_eq"),
        ("WF mean",      "wf_mean"),
        ("WF folds +",   "wf_pos"),
    ]
    for name, key in ens_metrics:
        v64  = results[64 ]["ensemble"][key]
        v128 = results[128]["ensemble"][key]
        v256 = results[256]["ensemble"][key]
        d128 = v128 - v64
        d256 = v256 - v64
        print(f"  {name:<22} {v64:>+12.3f} {v128:>+12.3f} {v256:>+12.3f}   {d128:>+8.3f}  {d256:>+8.3f}")

    print(f"\n  Per-fold WF (hidden=64 / 128 / 256):")
    for fi in range(N_FOLDS):
        v64  = results[64 ]["ensemble"]["wf_per_fold"][fi]
        v128 = results[128]["ensemble"]["wf_per_fold"][fi]
        v256 = results[256]["ensemble"]["wf_per_fold"][fi]
        print(f"    fold {fi+1}: {v64:>+7.2f}  {v128:>+7.2f}  {v256:>+7.2f}   "
              f"Δ128={v128-v64:>+6.2f}  Δ256={v256-v64:>+6.2f}")

    # net params at each hidden
    def n_params(h):
        return 50*h + h + h*(h//2) + (h//2) + (h//2)*10 + 10
    print(f"\n  Net params per seed:")
    for h in (64, 128, 256):
        print(f"    hidden={h:>3}: {n_params(h):>7,} params")

    # save
    out = CACHE / "eval_capacity_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
