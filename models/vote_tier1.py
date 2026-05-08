"""
Tier 1 voting validation:
  1) K=4 plurality (drop seed=7) of {42, 123, 0, 99} — fixes val regression?
  2) Disjoint K=5 plurality of {1, 13, 25, 50, 77} — does voting help structurally?
  3) Top-5-by-val plurality — picks best seeds by in-sample val Sharpe
  4) Top-5-by-WF plurality — picks best seeds by walk-forward mean

Also reports per-seed train-val Sharpes + walk-forward means for the new pool
(seeds 1, 13, 25, 50, 77) so we can identify the best single-seed in that pool
for the disjoint comparison.
"""
import json, pathlib, statistics, time
import numpy as np

from models.voting_ensemble import (
    _VotePolicy, evaluate_with_policy, load_net,
    eval_split, eval_walkforward, load_full_rl_period, run_variant,
    SEEDS_K10, _TAG_FOR,
)
from models.dqn_rollout    import _build_exit_arrays

CACHE = pathlib.Path("cache")


def _per_seed_eval(seed: int, atr_median, full):
    """Single-seed eval on val + test + WF for the given seed."""
    net = load_net(_TAG_FOR[seed])
    from models.dqn_network import masked_argmax
    import torch
    class _SingleGreedy:
        def __call__(self, s, v):
            with torch.no_grad():
                sb = torch.from_numpy(s).float().unsqueeze(0)
                vb = torch.from_numpy(v).bool().unsqueeze(0)
                return int(masked_argmax(net, sb, vb).item())
    pol = _SingleGreedy()
    v = eval_split(pol, "val",  atr_median)
    t = eval_split(pol, "test", atr_median)
    wf = eval_walkforward(lambda: _SingleGreedy(), atr_median, full)
    return dict(
        val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
        test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
        wf_per_fold=[r["sharpe"] for r in wf],
        wf_mean=statistics.mean([r["sharpe"] for r in wf]),
        wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
    )


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period("btc")

    print(f"\n{'='*120}")
    print("  TIER 1 VOTING VALIDATION")
    print(f"{'='*120}")

    # ── per-seed audit (all 10) ─────────────────────────────────────────────
    print("\n  Per-seed performance (all 10):")
    per_seed = {}
    for s in SEEDS_K10:
        per_seed[s] = _per_seed_eval(s, atr_median, full)

    print(f"\n  {'seed':<6} {'val':>7} {'test':>7} {'WF mean':>9} {'WF pos':>7} {'fold 6':>9}  per-fold")
    print("-"*125)
    for s, r in per_seed.items():
        print(f"  {s:<6} {r['val_sharpe']:>+7.2f} {r['test_sharpe']:>+7.2f} "
              f"{r['wf_mean']:>+9.3f} {r['wf_pos']:>3}/6   {r['wf_per_fold'][5]:>+9.2f}  "
              f"{[round(x, 2) for x in r['wf_per_fold']]}")

    # rank by val
    val_rank   = sorted(SEEDS_K10, key=lambda s: per_seed[s]["val_sharpe"], reverse=True)
    wf_rank    = sorted(SEEDS_K10, key=lambda s: per_seed[s]["wf_mean"], reverse=True)
    print(f"\n  Top-5 by val   : {val_rank[:5]}")
    print(f"  Top-5 by WF mean: {wf_rank[:5]}")
    top5_val = val_rank[:5]
    top5_wf  = wf_rank[:5]

    # ── voting variants to test ─────────────────────────────────────────────
    test_pools = [
        ("VOTE5_orig",         "plurality", None, [42, 7, 123, 0, 99]),
        ("VOTE4_drop7",        "plurality", None, [42, 123, 0, 99]),
        ("VOTE5_disjoint",     "plurality", None, [1, 13, 25, 50, 77]),
        ("VOTE5_top5_val",     "plurality", None, top5_val),
        ("VOTE5_top5_wf",      "plurality", None, top5_wf),
    ]

    results = []
    for name, mode, thr, seeds in test_pools:
        print(f"\n  Running {name}: seeds={seeds} ...")
        res = run_variant(name, mode, thr, seeds, atr_median, full,
                           track_agreement=False)
        results.append(res)

    # ── summary table ───────────────────────────────────────────────────────
    print(f"\n\n{'name':<22} {'val':>8} {'test':>8} {'val eq':>8} {'test eq':>8} "
          f"{'WF mean':>10} {'WF pos':>7} {'fold 6':>9}  per-fold + (trades)")
    print("-"*155)
    # references first
    print(f"{'BASELINE_FULL (s42)':<22} "
          f"{per_seed[42]['val_sharpe']:>+8.2f} {per_seed[42]['test_sharpe']:>+8.2f} "
          f"{per_seed[42]['val_eq']:>8.3f} {per_seed[42]['test_eq']:>8.3f} "
          f"{per_seed[42]['wf_mean']:>+10.3f} {per_seed[42]['wf_pos']:>3}/6 "
          f"{per_seed[42]['wf_per_fold'][5]:>+9.2f}  "
          f"{[round(x,2) for x in per_seed[42]['wf_per_fold']]}")

    for r in results:
        print(f"{r['name']:<22} {r['val_sharpe']:>+8.2f} {r['test_sharpe']:>+8.2f} "
              f"{r['val_eq']:>8.3f} {r['test_eq']:>8.3f} "
              f"{r['wf_mean']:>+10.3f} {r['wf_pos']:>3}/6 "
              f"{r['wf_per_fold'][5]:>+9.2f}  "
              f"{[round(x, 2) for x in r['wf_per_fold']]}  "
              f"({sum(r['wf_trades_per_fold'])} trades)")

    # ── disjoint K=5 vs its best individual ─────────────────────────────────
    disjoint_seeds = [1, 13, 25, 50, 77]
    disjoint_best = max(disjoint_seeds, key=lambda s: per_seed[s]["wf_mean"])
    disjoint_res  = next(r for r in results if r["name"] == "VOTE5_disjoint")
    print(f"\n  ── Disjoint K=5 validation ──")
    print(f"  Best single seed in pool {disjoint_seeds}: seed={disjoint_best} (WF {per_seed[disjoint_best]['wf_mean']:+.3f})")
    print(f"  K=5 plurality of pool:                       WF {disjoint_res['wf_mean']:+.3f}")
    print(f"  Δ ensemble vs best single   :                Δ  {disjoint_res['wf_mean']-per_seed[disjoint_best]['wf_mean']:+.3f}")

    # save
    out = CACHE / "vote_tier1_results.json"
    out.write_text(json.dumps(dict(per_seed=per_seed, variants=results),
                                indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
