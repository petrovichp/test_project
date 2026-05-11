"""
Z1.1 + Z1.3 + Z1.4 evaluation — all use 5-seed plurality ensembles.
Z1.2 (K=10 vanilla) already done in eval_z1_vote10.py.
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DQN, DuelingDQN
from models.dqn_rollout          import _build_exit_arrays
from models.audit_vote5_dd       import run_fold, run_walkforward, load_full_rl_period

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS_ORIG     = [42, 7, 123, 0, 99]
SEEDS_DISJOINT = [1, 13, 25, 50, 77]


def load_dqn(tag, hidden=64):
    net = DQN(50, 10, hidden)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def load_dueling(tag, hidden=64):
    net = DuelingDQN(50, 10, hidden)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def evaluate(name, nets, full, atr_median, fee=0.0):
    rows, trades = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
    sp_v = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
    sp_t = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
    _, vsh, _, vtr = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                  sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                  tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee, fold_id=0)
    _, tsh, _, ttr = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                   sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                   tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee, fold_id=0)
    wf = statistics.mean(r["sharpe"] for r in rows)
    wf_pos = sum(1 for r in rows if r["sharpe"] > 0)
    return dict(name=name, wf=wf, wf_pos=wf_pos, val=vsh, test=tsh,
                  per_fold=[r["sharpe"] for r in rows],
                  trades=len(trades), val_trades=len(vtr), test_trades=len(ttr))


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()

    print(f"\n{'='*100}\n  Z1 EVALUATION (Z1.1 + Z1.3 + Z1.4)\n{'='*100}")

    rows = []

    # Z1.1 — H256 + Double_Dueling
    print("\n  [Z1.1] H256 + Double_Dueling (5 seeds, orig pool)")
    nets = [load_dueling(f"VOTE5_H256_DD_seed{s}", hidden=256) for s in SEEDS_ORIG]
    rows.append(evaluate("VOTE5_H256_DD (Z1.1)", nets, full, atr_median))

    # Z1.3 — K=5 DD with DISJOINT seed pool
    print("\n  [Z1.3] DD K=5 with disjoint seeds (validates BASELINE_VOTE5_DD structurally)")
    nets = [load_dueling(f"VOTE10_DD_seed{s}", hidden=64) for s in SEEDS_DISJOINT]
    rows.append(evaluate("VOTE5_DD_DISJOINT (Z1.3)", nets, full, atr_median))

    # Z1.4 — H128 disjoint
    print("\n  [Z1.4a] H128 disjoint (5 seeds)")
    nets = [load_dqn(f"VOTE5_H128_DISJOINT_seed{s}", hidden=128) for s in SEEDS_DISJOINT]
    rows.append(evaluate("VOTE5_H128_DISJOINT (Z1.4)", nets, full, atr_median))

    # Z1.4 — H256 disjoint
    print("\n  [Z1.4b] H256 disjoint (5 seeds)")
    nets = [load_dqn(f"VOTE5_H256_DISJOINT_seed{s}", hidden=256) for s in SEEDS_DISJOINT]
    rows.append(evaluate("VOTE5_H256_DISJOINT (Z1.4)", nets, full, atr_median))

    # ── PRINT TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  Z1 RESULTS  vs  baselines (BASELINE_VOTE5 +10.40, H256 +11.86, DD +6.80)")
    print(f"{'='*100}\n")
    print(f"  {'config':<32} {'WF':>8} {'val':>7} {'test':>7} {'folds+':>7} "
          f"{'WF tr':>7} {'val tr':>8} {'test tr':>9}")
    print('  ' + '-'*100)
    for r in rows:
        print(f"  {r['name']:<32} {r['wf']:>+8.3f} {r['val']:>+7.2f} {r['test']:>+7.2f} "
              f"{r['wf_pos']:>4}/6  {r['trades']:>7} {r['val_trades']:>8} {r['test_trades']:>9}")

    print(f"\n  Per-fold WF Sharpes:")
    print(f"  {'config':<32} " + " ".join(f"f{i+1:>5}" for i in range(N_FOLDS)))
    for r in rows:
        print(f"  {r['name']:<32} " + " ".join(f"{x:>+5.2f}" for x in r['per_fold']))

    out = CACHE / "results" / "z1_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
