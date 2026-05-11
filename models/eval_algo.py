"""
A3 — Algorithmic upgrade comparison: DQN / Double / Dueling / Double_Dueling.

Compares K=5 plurality ensembles for each algorithm:
  - dqn:           BASELINE_VOTE5 (seeds 42,7,123,0,99 with vanilla DQN)
  - double:        VOTE5_DOUBLE   (same seeds, Double DQN target)
  - dueling:       VOTE5_DUELING  (same seeds, Dueling network)
  - double_dueling:VOTE5_DD       (same seeds, both)

All at h=64, v5 state.
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DQN, DuelingDQN, masked_argmax
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.voting_ensemble      import _VotePolicy, evaluate_with_policy

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS   = [42, 7, 123, 0, 99]


def load_net(tag: str, dueling: bool):
    cls = DuelingDQN if dueling else DQN
    net = cls(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_split(policy_fn, split: str, atr_median: float):
    sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
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
        sub = {k: full[k][a:b] for k in full}
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        out = evaluate_with_policy(policy_factory(), sub["state"], sub["valid_actions"],
                                     sub["signals"], sub["price"], tp, sl, tr, tab, be, ts)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"], equity=out["equity_final"]))
    return rows


class _SingleGreedy:
    def __init__(self, net): self.net = net
    def __call__(self, s, v):
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool().unsqueeze(0)
            return int(masked_argmax(self.net, sb, vb).item())


ALGO_INFO = {
    # name → (tag_template, dueling_bool)
    "dqn":             ("BASELINE_FULL{seed_suffix}",     False),
    "double":          ("VOTE5_DOUBLE_seed{seed}",        False),
    "dueling":         ("VOTE5_DUELING_seed{seed}",       True),
    "double_dueling":  ("VOTE5_DD_seed{seed}",            True),
}


def tag_for(algo: str, seed: int) -> str:
    template, _ = ALGO_INFO[algo]
    if algo == "dqn":
        # seed 42 → BASELINE_FULL; others → BASELINE_FULL_seed{seed}
        return "BASELINE_FULL" if seed == 42 else f"BASELINE_FULL_seed{seed}"
    return template.format(seed=seed)


def eval_pool(algo: str, atr_median, full):
    template, dueling = ALGO_INFO[algo]
    nets = [load_net(tag_for(algo, s), dueling) for s in SEEDS]

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
    ens = dict(
        val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
        test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
        wf_per_fold=[r["sharpe"] for r in wf],
        wf_mean=statistics.mean([r["sharpe"] for r in wf]),
        wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
    )
    return per_seed, ens


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()

    print(f"\n{'='*120}\n  A3 ALGORITHM TEST — DQN / Double / Dueling / Double-Dueling\n{'='*120}")

    results = {}
    for algo in ("dqn", "double", "dueling", "double_dueling"):
        print(f"\n  Evaluating {algo} pool ...")
        per_seed, ens = eval_pool(algo, atr_median, full)
        results[algo] = dict(per_seed=per_seed, ensemble=ens)

    # ── per-seed summary ────────────────────────────────────────────────────
    print(f"\n  Per-seed train-val / WF (single-seed greedy):")
    print(f"  {'seed':<5} " + "".join(f"{'val/'+a:>14} {'WF/'+a:>11}  " for a in ("dqn","double","dueling","dd")))
    print("  " + "-"*150)
    for s in SEEDS:
        row = f"  {s:<5} "
        for a in ("dqn", "double", "dueling", "double_dueling"):
            r = results[a]["per_seed"][s]
            row += f"{r['val_sharpe']:>+14.2f} {r['wf_mean']:>+11.3f}  "
        print(row)

    # ── ensemble comparison ─────────────────────────────────────────────────
    print(f"\n  K=5 plurality ensemble comparison:")
    print(f"  {'metric':<22} " + "".join(f"{a.upper():>14} " for a in ("dqn","double","dueling","dd")) +
          "   Δdouble  Δdueling  Δdd")
    print("  " + "-"*135)
    metrics = [
        ("val Sharpe",   "val_sharpe"),
        ("val equity",   "val_eq"),
        ("test Sharpe",  "test_sharpe"),
        ("test equity",  "test_eq"),
        ("WF mean",      "wf_mean"),
        ("WF folds +",   "wf_pos"),
    ]
    base = lambda key: results["dqn"]["ensemble"][key]
    for name, key in metrics:
        v_dqn = results["dqn"]["ensemble"][key]
        v_dbl = results["double"]["ensemble"][key]
        v_due = results["dueling"]["ensemble"][key]
        v_dd  = results["double_dueling"]["ensemble"][key]
        print(f"  {name:<22} {v_dqn:>+14.3f} {v_dbl:>+14.3f} {v_due:>+14.3f} {v_dd:>+14.3f}   "
              f"{v_dbl-v_dqn:>+7.3f}  {v_due-v_dqn:>+7.3f}  {v_dd-v_dqn:>+7.3f}")

    print(f"\n  Per-fold WF (dqn / double / dueling / dd):")
    for fi in range(N_FOLDS):
        per_fold = [results[a]["ensemble"]["wf_per_fold"][fi] for a in ("dqn","double","dueling","double_dueling")]
        print(f"    fold {fi+1}: " + " ".join(f"{x:>+7.2f}" for x in per_fold))

    out = CACHE / "results" / "eval_algo_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
