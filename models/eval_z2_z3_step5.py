"""
Step 5 — evaluate VOTE5_v9_H256_DD (combined v7_basis + v8_s11s13).
Comparison vs VOTE5_v8_H256_DD (current baseline, WF +12.07) and
VOTE5_H256_DD (prior baseline, WF +11.05).
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DuelingDQN
from models.dqn_rollout          import _build_exit_arrays
from models.audit_vote5_dd       import run_fold, run_walkforward

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]


def load_dueling(tag, hidden, state_dim, n_actions):
    net = DuelingDQN(state_dim, n_actions, hidden)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def load_full_rl_period_for(suffix):
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"btc_dqn_state_{split}{suffix}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def evaluate(name, nets, full, atr_median, suffix=""):
    rows, trades = run_walkforward(nets, full, atr_median, fee=0.0, with_reason=False)
    sp_v  = np.load(CACHE / f"btc_dqn_state_val{suffix}.npz")
    sp_t  = np.load(CACHE / f"btc_dqn_state_test{suffix}.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
    _, vsh, _, vtr = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                  sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                  tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=0.0, fold_id=0)
    _, tsh, _, ttr = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                   sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                   tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=0.0, fold_id=0)
    wf = statistics.mean(r["sharpe"] for r in rows)
    wf_pos = sum(1 for r in rows if r["sharpe"] > 0)
    return dict(name=name, wf=wf, wf_pos=wf_pos, val=vsh, test=tsh,
                  per_fold=[r["sharpe"] for r in rows],
                  trades=len(trades), val_trades=len(vtr), test_trades=len(ttr))


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    print(f"\n{'='*100}\n  Z2/Z3 STEP 5 — Combined v7_basis + v8 (basis state + S11/S13 actions)\n{'='*100}")

    full = load_full_rl_period_for("_v9_basis_s11s13")
    nets = [load_dueling(f"VOTE5_v9_H256_DD_seed{s}", hidden=256,
                            state_dim=57, n_actions=12) for s in SEEDS]
    r = evaluate("VOTE5_v9_H256_DD (Step 5)", nets, full, atr_median, suffix="_v9_basis_s11s13")

    print(f"\n  {'config':<37} {'WF':>8} {'val':>7} {'test':>7} {'folds+':>7} "
          f"{'WF tr':>7} {'val tr':>8} {'test tr':>9}")
    print('  ' + '-'*100)
    # references
    refs = [
        ("─ VOTE5 (vanilla h64)",      10.40, 3.53, 4.19, 6, 1122, 233, 174),
        ("─ VOTE5_H256_DD (Z1)",       11.05, 3.21, 9.01, 6, 1372, 320, 228),
        ("─ VOTE5_v7basis (Step 3)",   11.66, 6.05, 2.90, 6, 1182, 243, 212),
        ("─ VOTE5_v8 S11+S13 (Step 4)",12.07, 6.67, 4.44, 6, 1416, 300, 199),
    ]
    for nm, wf, va, te, fp, tr, vtr, ttr in refs:
        print(f"  {nm:<37} {wf:>+8.2f} {va:>+7.2f} {te:>+7.2f} {fp:>4}/6  {tr:>7} {vtr:>8} {ttr:>9}")
    print(f"  {r['name']+' ⭐':<37} {r['wf']:>+8.3f} {r['val']:>+7.2f} {r['test']:>+7.2f} "
          f"{r['wf_pos']:>4}/6  {r['trades']:>7} {r['val_trades']:>8} {r['test_trades']:>9}")

    print(f"\n  Per-fold WF Sharpes:")
    print(f"  {'config':<37} " + " ".join(f"f{i+1:>5}" for i in range(6)))
    print(f"  {'─ VOTE5_H256_DD (Z1 baseline)':<37} " + " ".join(f"{x:>+5.2f}" for x in [11.75, 16.47, 7.86, 19.12, 2.88, 8.23]))
    print(f"  {'─ VOTE5_v8 (Step 4)':<37} " + " ".join(f"{x:>+5.2f}" for x in [11.14, 19.43, 13.08, 18.01, 6.29, 4.44]))
    print(f"  {r['name']:<37} " + " ".join(f"{x:>+5.2f}" for x in r['per_fold']))

    out = CACHE / "z2_z3_step5_results.json"
    out.write_text(json.dumps(r, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
