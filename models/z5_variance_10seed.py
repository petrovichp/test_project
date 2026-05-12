"""
Z5.2 — 10-seed variance summary for VOTE5_v8_H256_DD and DISTILL_v8.

Both families already trained 10 seeds total:
- orig pool:    {42, 7, 123, 0, 99}
- disjoint pool:{1, 13, 25, 50, 77}

Walk-forward Sharpe + val + test per individual seed, plus 10-seed plurality
eval (combining both pools). The 10-seed ensemble represents the maximum
deployable diversification.
"""
import json, statistics, time
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.audit_vote5_dd import run_fold, run_walkforward
from config.cache_paths import POLICIES, STATE, PREDS, RESULTS

ORIG_SEEDS     = [42, 7, 123, 0, 99]
DISJOINT_SEEDS = [1, 13, 25, 50, 77]


def load_v8(seed: int, tag_prefix: str = "VOTE5_v8_H256_DD") -> DuelingDQN:
    n = DuelingDQN(52, 12, 256)
    # orig pool: VOTE5_v8_H256_DD_seed{N}.pt
    # disjoint pool: VOTE5_v8_DISJOINT_H256_DD_seed{N}.pt
    if seed in DISJOINT_SEEDS:
        path = POLICIES / f"btc_dqn_policy_VOTE5_v8_DISJOINT_H256_DD_seed{seed}.pt"
    else:
        path = POLICIES / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{seed}.pt"
    n.load_state_dict(torch.load(path, map_location="cpu"))
    n.eval(); return n


def load_distill(seed: int) -> DuelingDQN:
    n = DuelingDQN(52, 12, 256)
    if seed in DISJOINT_SEEDS:
        path = POLICIES / f"btc_dqn_policy_DISTILL_v8_DISJOINT_seed{seed}.pt"
    else:
        path = POLICIES / f"btc_dqn_policy_DISTILL_v8_seed{seed}.pt"
    n.load_state_dict(torch.load(path, map_location="cpu"))
    n.eval(); return n


def load_full_v8():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(STATE / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for k in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(k, []).append(sp[k])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_pack(nets, label, atr_median, full):
    rows, trades = run_walkforward(nets, full, atr_median, fee=0.0, with_reason=False)
    wf = statistics.mean(r["sharpe"] for r in rows)
    pos = sum(1 for r in rows if r["sharpe"] > 0)
    sp_v = np.load(STATE / "btc_dqn_state_val_v8_s11s13.npz")
    sp_t = np.load(STATE / "btc_dqn_state_test_v8_s11s13.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
    _, vsh, _, vtr = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=0.0, fold_id=0)
    _, tsh, _, ttr = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=0.0, fold_id=0)
    print(f"  {label:<42}  WF {wf:>+7.3f}  val {vsh:>+7.3f}  test {tsh:>+7.3f}  folds+ {pos}/6")
    return dict(label=label, wf=wf, val=vsh, test=tsh, folds_pos=pos,
                trades_wf=len(trades), trades_val=len(vtr), trades_test=len(ttr))


def variance_summary(per_seed_results, label):
    wfs   = [r["wf"]   for r in per_seed_results]
    vals  = [r["val"]  for r in per_seed_results]
    tests = [r["test"] for r in per_seed_results]
    print(f"\n  {label} variance across {len(per_seed_results)} seeds:")
    print(f"    WF:   mean {statistics.mean(wfs):+.3f}  stdev {statistics.stdev(wfs):.3f}  "
          f"min {min(wfs):+.3f}  max {max(wfs):+.3f}")
    print(f"    val:  mean {statistics.mean(vals):+.3f}  stdev {statistics.stdev(vals):.3f}  "
          f"min {min(vals):+.3f}  max {max(vals):+.3f}")
    print(f"    test: mean {statistics.mean(tests):+.3f}  stdev {statistics.stdev(tests):.3f}  "
          f"min {min(tests):+.3f}  max {max(tests):+.3f}")
    return dict(
        wf=dict(mean=statistics.mean(wfs),  stdev=statistics.stdev(wfs),  min=min(wfs),  max=max(wfs)),
        val=dict(mean=statistics.mean(vals), stdev=statistics.stdev(vals), min=min(vals), max=max(vals)),
        test=dict(mean=statistics.mean(tests),stdev=statistics.stdev(tests),min=min(tests),max=max(tests)),
    )


def main():
    t0 = time.perf_counter()
    vol = np.load(PREDS / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_v8()
    all_seeds = ORIG_SEEDS + DISJOINT_SEEDS

    print(f"\n{'='*120}\n  Z5.2 — 10-seed variance: VOTE5_v8_H256_DD\n{'='*120}\n")
    v8_per_seed = []
    for s in all_seeds:
        v8_per_seed.append(eval_pack([load_v8(s)], f"VOTE5_v8 single s={s}", atr_median, full))
    v8_var = variance_summary(v8_per_seed, "VOTE5_v8 single-seed")

    print(f"\n  Voting ensembles:")
    v8_orig5_vote = eval_pack([load_v8(s) for s in ORIG_SEEDS],
                                "VOTE5_v8 orig K=5 vote", atr_median, full)
    v8_disj5_vote = eval_pack([load_v8(s) for s in DISJOINT_SEEDS],
                                "VOTE5_v8 disjoint K=5 vote", atr_median, full)
    v8_vote10 = eval_pack([load_v8(s) for s in all_seeds],
                           "VOTE5_v8 ALL K=10 vote", atr_median, full)

    print(f"\n{'='*120}\n  Z5.2 — 10-seed variance: DISTILL_v8\n{'='*120}\n")
    distill_per_seed = []
    for s in all_seeds:
        try:
            distill_per_seed.append(eval_pack([load_distill(s)],
                                              f"DISTILL_v8 single s={s}", atr_median, full))
        except FileNotFoundError as e:
            print(f"    skip seed {s}: {e}")
    distill_var = variance_summary(distill_per_seed, "DISTILL_v8 single-seed")

    print(f"\n  Voting ensembles:")
    distill_orig5  = eval_pack([load_distill(s) for s in ORIG_SEEDS],
                                  "DISTILL_v8 orig K=5 vote", atr_median, full)
    distill_disj5  = eval_pack([load_distill(s) for s in DISJOINT_SEEDS],
                                  "DISTILL_v8 disjoint K=5 vote", atr_median, full)
    distill_vote10 = eval_pack([load_distill(s) for s in all_seeds],
                                  "DISTILL_v8 ALL K=10 vote", atr_median, full)

    out = RESULTS / "z5_variance_10seed.json"
    out.write_text(json.dumps(dict(
        v8_per_seed=v8_per_seed, v8_variance=v8_var,
        v8_orig5_vote=v8_orig5_vote, v8_disj5_vote=v8_disj5_vote, v8_vote10=v8_vote10,
        distill_per_seed=distill_per_seed, distill_variance=distill_var,
        distill_orig5_vote=distill_orig5, distill_disj5_vote=distill_disj5,
        distill_vote10=distill_vote10,
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
