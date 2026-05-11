"""
Evaluate Steps 2, 3, 4 ensembles vs VOTE5_H256_DD baseline.

Step 2 — v7_pa  state (price-action context, 54-dim, 10 actions)
Step 3 — v7_basis state (basis+funding, 55-dim, 10 actions)
Step 4 — v8_s11s13 state (52-dim, 12-action space S11+S13)
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DuelingDQN
from models.dqn_rollout          import _build_exit_arrays
from models.audit_vote5_dd       import run_fold, run_walkforward, load_full_rl_period

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]


def load_dueling(tag, hidden, state_dim, n_actions):
    net = DuelingDQN(state_dim, n_actions, hidden)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval(); return net


def load_full_rl_period_for(suffix=""):
    """Load full RL period for given state version suffix (e.g. '_v7_pa', '_v8_s11s13')."""
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"btc_dqn_state_{split}{suffix}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def evaluate(name, nets, full, atr_median, suffix="", fee=0.0):
    rows, trades = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
    sp_v  = np.load(CACHE / f"btc_dqn_state_val{suffix}.npz")
    sp_t  = np.load(CACHE / f"btc_dqn_state_test{suffix}.npz")
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
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    print(f"\n{'='*100}\n  Z2/Z3 STEP EVALUATION\n{'='*100}")

    rows = []

    # ── Step 2: v7_pa (price-action) ────────────────────────────────────────
    print(f"\n  [Step 2] v7_pa price-action context (54-dim state, 10 actions)")
    full = load_full_rl_period_for("_v7_pa")
    nets = [load_dueling(f"VOTE5_v7pa_H256_DD_seed{s}", hidden=256,
                            state_dim=54, n_actions=10) for s in SEEDS]
    rows.append(evaluate("VOTE5_v7pa_H256_DD (Step 2)", nets, full, atr_median, suffix="_v7_pa"))

    # ── Step 3: v7_basis (basis+funding) ────────────────────────────────────
    print(f"\n  [Step 3] v7_basis basis+funding state (55-dim, 10 actions)")
    full = load_full_rl_period_for("_v7_basis")
    nets = [load_dueling(f"VOTE5_v7basis_H256_DD_seed{s}", hidden=256,
                            state_dim=55, n_actions=10) for s in SEEDS]
    rows.append(evaluate("VOTE5_v7basis_H256_DD (Step 3)", nets, full, atr_median, suffix="_v7_basis"))

    # ── Step 4: v8_s11s13 (S11+S13 added) ───────────────────────────────────
    print(f"\n  [Step 4] v8_s11s13 (52-dim state, 12 actions with S11+S13)")
    full = load_full_rl_period_for("_v8_s11s13")
    nets = [load_dueling(f"VOTE5_v8_H256_DD_seed{s}", hidden=256,
                            state_dim=52, n_actions=12) for s in SEEDS]
    rows.append(evaluate("VOTE5_v8_H256_DD (Step 4)", nets, full, atr_median, suffix="_v8_s11s13"))

    # ── PRINT TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  RESULTS  vs  VOTE5_H256_DD baseline (WF +11.05, val +3.21, test +9.01, 6/6 folds)")
    print(f"{'='*100}\n")
    print(f"  {'config':<35} {'WF':>8} {'val':>7} {'test':>7} {'folds+':>7} "
          f"{'WF tr':>7} {'val tr':>8} {'test tr':>9}")
    print('  ' + '-'*100)
    for r in rows:
        print(f"  {r['name']:<35} {r['wf']:>+8.3f} {r['val']:>+7.2f} {r['test']:>+7.2f} "
              f"{r['wf_pos']:>4}/6  {r['trades']:>7} {r['val_trades']:>8} {r['test_trades']:>9}")
    print(f"  {'─ baseline VOTE5_H256_DD ─':<35} {'+11.050':>8} {'+3.21':>7} {'+9.01':>7} {'  6/6':>7} "
          f"{'1372':>7} {'320':>8} {'228':>9}")

    print(f"\n  Per-fold WF Sharpes:")
    print(f"  {'config':<35} " + " ".join(f"f{i+1:>5}" for i in range(6)))
    for r in rows:
        print(f"  {r['name']:<35} " + " ".join(f"{x:>+5.2f}" for x in r['per_fold']))
    print(f"  {'─ baseline VOTE5_H256_DD ─':<35} " + " ".join(f"{x:>+5.2f}" for x in [11.75, 16.47, 7.86, 19.12, 2.88, 8.23]))

    out = CACHE / "z2_z3_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
