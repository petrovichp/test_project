"""
Z5.3 partial — fee-curve scan for VOTE5_v8_H256_DD (new primary baseline).
Mirrors audit_vote5_dd.py Part B for the v8 ensemble.
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DuelingDQN
from models.dqn_rollout          import _build_exit_arrays
from models.audit_vote5_dd       import run_fold, run_walkforward

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]


def load_dueling_v8(s):
    net = DuelingDQN(52, 12, 256)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt",
                                       map_location="cpu"))
    net.eval(); return net


def load_full_v8():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    full = load_full_v8()
    nets = [load_dueling_v8(s) for s in SEEDS]

    print(f"\n{'='*100}\n  Fee-curve scan — VOTE5_v8_H256_DD\n{'='*100}")
    fee_levels = [0.0, 0.0001, 0.0002, 0.0004, 0.00045, 0.0006, 0.0008, 0.0012]

    print(f"\n  {'fee/side':>11} {'bp':>5} {'WF':>9} {'val':>8} {'test':>8} {'folds+':>8} {'WF tr':>8}")
    print('  ' + '-'*70)

    rows = []
    for fee in fee_levels:
        wf_rows, trades_f = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
        wf  = statistics.mean(r["sharpe"] for r in wf_rows)
        pos = sum(1 for r in wf_rows if r["sharpe"] > 0)

        sp_v = np.load(CACHE / "btc_dqn_state_val_v8_s11s13.npz")
        sp_t = np.load(CACHE / "btc_dqn_state_test_v8_s11s13.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
        _, vsh, _, _ = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                    sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                    tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee, fold_id=0)
        _, tsh, _, _ = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                     sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                     tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee, fold_id=0)
        rows.append(dict(fee=fee, fee_bp=fee*10000, wf=wf, wf_pos=pos,
                            val=vsh, test=tsh, trades=len(trades_f)))
        print(f"  {fee:>11.5f} {fee*10000:>4.1f}  {wf:>+8.3f} {vsh:>+8.2f} {tsh:>+8.2f}  {pos:>3}/6  {len(trades_f):>8}")

    out = CACHE / "fee_curve_v8_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
