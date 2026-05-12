"""
Z5.3 — fee-curve scan for DISTILL_v8_seed42 (the new cheap-deploy candidate)
and compare to VOTE5_v8_H256_DD baseline at the same fee levels.

Mirrors models/fee_curve_v8.py but uses a single distilled net policy and
re-evaluates the teacher ensemble at the same grid.
"""
import json, statistics, time
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.audit_vote5_dd import run_fold, run_walkforward
from config.cache_paths import POLICIES, STATE, PREDS, RESULTS

SEEDS = [42, 7, 123, 0, 99]


def load_dueling(tag, seed=None):
    net = DuelingDQN(52, 12, 256)
    fname = f"btc_dqn_policy_{tag}_seed{seed}.pt" if seed is not None else f"btc_dqn_policy_{tag}.pt"
    net.load_state_dict(torch.load(POLICIES / fname, map_location="cpu"))
    net.eval(); return net


def load_full():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(STATE / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for k in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(k, []).append(sp[k])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def fee_curve(nets, label, atr_median, full, fee_levels):
    print(f"\n  {label}")
    print(f"  {'fee bp':>7} {'WF':>9} {'val':>9} {'test':>9} {'folds+':>8}  {'trades':>7}")
    rows = []
    sp_v = np.load(STATE / "btc_dqn_state_val_v8_s11s13.npz")
    sp_t = np.load(STATE / "btc_dqn_state_test_v8_s11s13.npz")
    for fee in fee_levels:
        wf_rows, trades = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in wf_rows)
        pos = sum(1 for r in wf_rows if r["sharpe"] > 0)
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
        _, vsh, _, _ = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                  sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                  tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee, fold_id=0)
        _, tsh, _, _ = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                  sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                  tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee, fold_id=0)
        rows.append(dict(fee_bp=fee*10000, fee=fee, wf=wf, val=vsh, test=tsh,
                         folds_pos=pos, trades=len(trades), label=label))
        print(f"  {fee*10000:>7.1f}  {wf:>+8.3f} {vsh:>+8.2f} {tsh:>+8.2f}  "
              f"{pos:>3}/6     {len(trades):>7}")
    return rows


def main():
    t0 = time.perf_counter()
    vol = np.load(PREDS / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()

    fee_levels = [0.0, 0.0001, 0.0002, 0.0004, 0.00045, 0.0006, 0.0008]

    print(f"\n{'='*100}\n  Z5.3 — Fee-curve scan: DISTILL_v8_seed42 vs VOTE5_v8 teacher\n{'='*100}")

    rows = []
    # DISTILL_v8_seed42 (single net)
    distill = load_dueling("DISTILL_v8", seed=42)
    rows += fee_curve([distill], "DISTILL_v8_seed42 (single net)", atr_median, full, fee_levels)

    # DISTILL_v8 orig K=5 plurality
    distill_v5 = [load_dueling("DISTILL_v8", seed=s) for s in SEEDS]
    rows += fee_curve(distill_v5, "DISTILL_v8 orig K=5 vote", atr_median, full, fee_levels)

    # Teacher VOTE5_v8_H256_DD
    teacher = [load_dueling("VOTE5_v8_H256_DD", seed=s) for s in SEEDS]
    rows += fee_curve(teacher, "VOTE5_v8_H256_DD teacher (K=5)", atr_median, full, fee_levels)

    out = RESULTS / "z5_fee_curve_distill.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
