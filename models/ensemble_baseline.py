"""
Ensemble baseline: K=5 BASELINE_FULL policies (seeds 42, 7, 123, 0, 99) with
Q-value averaging at decision time.

Reports:
  - K=5 ensemble metrics on DQN-val, DQN-test, WF (6 folds)
  - Per-seed metrics for context
  - Mean-of-seeds for comparison ("if we just averaged each seed's Sharpe")
  - K=3 ensemble (subset 42, 7, 123) for direct comparison to seed_variance.md

Uses the existing rule-based exit pipeline (same simulator as evaluate_policy
and group_c2_walkforward).
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network  import DQN, EnsembleDQN
from models.dqn_selector import evaluate_policy
from models.dqn_rollout         import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL

CACHE = pathlib.Path("cache")

ALL_SEEDS = [42, 7, 123, 0, 99]
TAG_FOR  = {42: "BASELINE_FULL",
             7: "BASELINE_FULL_seed7",
             123: "BASELINE_FULL_seed123",
             0: "BASELINE_FULL_seed0",
             99: "BASELINE_FULL_seed99"}

N_FOLDS = 6


def load_net(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def build_ensemble(seeds: list) -> EnsembleDQN:
    nets = [load_net(TAG_FOR[s]) for s in seeds]
    ens = EnsembleDQN(nets)
    ens.eval()
    return ens


def eval_split(net, split: str, atr_median: float):
    sp = np.load(CACHE / f"btc_dqn_state_{split}.npz")
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    out = evaluate_policy(net, sp["state"], sp["valid_actions"],
                           sp["signals"], sp["price"], tp, sl, tr, tab, be, ts,
                           valid_mask_override=None, fee=0.0)
    return dict(sharpe=out["sharpe"], equity=out["equity_final"],
                trades=out["n_trades"], win_rate=out["win_rate"], max_dd=out["max_dd"])


def load_full_rl_period(ticker: str = "btc"):
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"{ticker}_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_walkforward(net, atr_median: float, full):
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    rows = []
    for i in range(N_FOLDS):
        # fold bounds are in pq_use space (start at RL_START_REL); subtract for array indexing
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL
        b = b_pq - RL_START_REL
        sub_state   = full["state"][a:b]
        sub_valid   = full["valid_actions"][a:b]
        sub_signals = full["signals"][a:b]
        sub_prices  = full["price"][a:b]
        sub_atr     = full["atr"][a:b]
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub_prices, sub_atr, atr_median)
        out = evaluate_policy(net, sub_state, sub_valid, sub_signals, sub_prices,
                               tp, sl, tr, tab, be, ts,
                               valid_mask_override=None, fee=0.0)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"],
                          equity=out["equity_final"], trades=out["n_trades"],
                          max_dd=out["max_dd"]))
    return rows


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    full = load_full_rl_period("btc")

    results = {}
    print(f"\n{'='*100}\n  ENSEMBLE BASELINE — comparing single seeds, K=3, K=5\n{'='*100}")

    # ── individual seed results (load from existing artefacts where possible) ──
    for s in ALL_SEEDS:
        tag = TAG_FOR[s]
        if not (CACHE / f"btc_dqn_policy_{tag}.pt").exists():
            print(f"  ! missing policy for seed={s} ({tag}); skipping")
            continue
        net = load_net(tag)
        v = eval_split(net, "val",  atr_median)
        t = eval_split(net, "test", atr_median)
        wf = eval_walkforward(net, atr_median, full)
        results[f"seed_{s}"] = dict(
            val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
            test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_eq_per_fold=[r["equity"] for r in wf],
            wf_trades_per_fold=[r["trades"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    # ── K=3 ensemble (seeds 42, 7, 123 — same set as seed_variance.md) ──
    K3 = [42, 7, 123]
    if all((CACHE / f"btc_dqn_policy_{TAG_FOR[s]}.pt").exists() for s in K3):
        ens3 = build_ensemble(K3)
        v = eval_split(ens3, "val",  atr_median)
        t = eval_split(ens3, "test", atr_median)
        wf = eval_walkforward(ens3, atr_median, full)
        results["ensemble_K3"] = dict(
            seeds=K3,
            val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
            test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_eq_per_fold=[r["equity"] for r in wf],
            wf_trades_per_fold=[r["trades"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    # ── K=4 ensemble (drop the weakest val: seed=7) ──
    K4 = [42, 123, 0, 99]
    if all((CACHE / f"btc_dqn_policy_{TAG_FOR[s]}.pt").exists() for s in K4):
        ens4 = build_ensemble(K4)
        v = eval_split(ens4, "val",  atr_median)
        t = eval_split(ens4, "test", atr_median)
        wf = eval_walkforward(ens4, atr_median, full)
        results["ensemble_K4_drop7"] = dict(
            seeds=K4,
            val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
            test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_eq_per_fold=[r["equity"] for r in wf],
            wf_trades_per_fold=[r["trades"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    # ── K=2 ensemble of top-2 by val (42, 123) ──
    K2 = [42, 123]
    if all((CACHE / f"btc_dqn_policy_{TAG_FOR[s]}.pt").exists() for s in K2):
        ens2 = build_ensemble(K2)
        v = eval_split(ens2, "val",  atr_median)
        t = eval_split(ens2, "test", atr_median)
        wf = eval_walkforward(ens2, atr_median, full)
        results["ensemble_K2_top2val"] = dict(
            seeds=K2,
            val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
            test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_eq_per_fold=[r["equity"] for r in wf],
            wf_trades_per_fold=[r["trades"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    # ── K=5 ensemble ──
    K5 = ALL_SEEDS
    if all((CACHE / f"btc_dqn_policy_{TAG_FOR[s]}.pt").exists() for s in K5):
        ens5 = build_ensemble(K5)
        v = eval_split(ens5, "val",  atr_median)
        t = eval_split(ens5, "test", atr_median)
        wf = eval_walkforward(ens5, atr_median, full)
        results["ensemble_K5"] = dict(
            seeds=K5,
            val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
            test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
            wf_per_fold=[r["sharpe"] for r in wf],
            wf_eq_per_fold=[r["equity"] for r in wf],
            wf_trades_per_fold=[r["trades"] for r in wf],
            wf_mean=statistics.mean([r["sharpe"] for r in wf]),
            wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
        )

    # ── print summary table ─────────────────────────────────────────────────
    print(f"\n{'name':<20} {'val Sharpe':>11} {'test Sharpe':>12} {'val eq':>8} {'test eq':>8} "
          f"{'WF mean':>10} {'WF pos':>7}  per-fold WF Sharpe")
    print("-"*135)
    for name, r in results.items():
        per_fold = [round(s, 2) for s in r["wf_per_fold"]]
        print(f"{name:<20} {r['val_sharpe']:>+11.3f} {r['test_sharpe']:>+12.3f} "
              f"{r['val_eq']:>8.3f} {r['test_eq']:>8.3f} "
              f"{r['wf_mean']:>+10.3f} {r['wf_pos']:>3}/6  {per_fold}")

    # variance summary across single seeds
    seed_keys = [k for k in results if k.startswith("seed_")]
    if len(seed_keys) >= 2:
        print(f"\nAcross {len(seed_keys)} single seeds:")
        for label, key in [("val Sharpe","val_sharpe"), ("test Sharpe","test_sharpe"),
                            ("WF mean","wf_mean")]:
            xs = [results[k][key] for k in seed_keys]
            m = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            print(f"  {label:<14}: mean {m:+7.3f}  std {sd:.3f}  spread {max(xs)-min(xs):.3f}")

    # ensemble vs mean-of-seeds (the key comparison)
    if "ensemble_K5" in results and len(seed_keys) == 5:
        ens = results["ensemble_K5"]
        means = {key: statistics.mean([results[k][key] for k in seed_keys])
                  for key in ("val_sharpe", "test_sharpe", "wf_mean")}
        print(f"\nEnsemble K=5 vs mean-of-seeds (K=5):")
        for label, key in [("val Sharpe","val_sharpe"),
                            ("test Sharpe","test_sharpe"),
                            ("WF mean","wf_mean")]:
            print(f"  {label:<14}: ensemble {ens[key]:+7.3f}  vs mean-of-seeds {means[key]:+7.3f}  "
                  f"  Δ {ens[key]-means[key]:+7.3f}")

    # save
    out = CACHE / "ensemble_baseline_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
