"""
Path C2 — walk-forward eval of DISTILL_v8 (single-net + 5-seed ensemble).

Compares:
  - 5 single-net distilled policies (one per seed)
  - DISTILL_v8 5-seed plurality vote (the deployable alternative to VOTE5_v8)
  - VOTE5_v8_H256_DD baseline (re-eval for parity check)
  - Per-fold and val/test breakdown

Run: python3 -m models.eval_distill_v8
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.audit_vote5_dd import run_fold, run_walkforward

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]


def load_net(tag: str, seed: int) -> DuelingDQN:
    net = DuelingDQN(52, 12, 256)
    net.load_state_dict(torch.load(
        CACHE / "policies" / f"btc_dqn_policy_{tag}_seed{seed}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_v8():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_pack(nets, full, atr_median, label):
    rows, trades = run_walkforward(nets, full, atr_median, fee=0.0, with_reason=False)
    wf = statistics.mean(r["sharpe"] for r in rows)
    pos = sum(1 for r in rows if r["sharpe"] > 0)
    # val + test single shot
    sp_v = np.load(CACHE / "state" / "btc_dqn_state_val_v8_s11s13.npz")
    sp_t = np.load(CACHE / "state" / "btc_dqn_state_test_v8_s11s13.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
    _, vsh, veq, vtr = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                  sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                  tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=0.0, fold_id=0)
    _, tsh, teq, ttr = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                  sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                  tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=0.0, fold_id=0)
    per_fold = [round(r["sharpe"], 3) for r in rows]
    print(f"  {label:<35}  WF {wf:>+7.3f}  val {vsh:>+7.3f}  test {tsh:>+7.3f}  "
          f"folds+ {pos}/6  trades(WF/val/test) {len(trades):>4}/{len(vtr):>4}/{len(ttr):>4}")
    return dict(label=label, wf=wf, wf_pos=pos, val=vsh, test=tsh,
                trades_wf=len(trades), trades_val=len(vtr), trades_test=len(ttr),
                per_fold=per_fold, val_eq=veq, test_eq=teq)


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_v8()

    print(f"\n{'='*120}\n  C2 DISTILL_v8 — walk-forward eval\n{'='*120}\n")

    rows = []

    # 1) each single distilled net (variance check)
    print(f"  Single-net distilled policies (each is one seed, no voting):")
    for s in SEEDS:
        rows.append(eval_pack([load_net("DISTILL_v8", s)], full, atr_median,
                              f"DISTILL_v8 single seed={s}"))

    # 2) 5-seed DISTILL_v8 plurality
    print(f"\n  5-seed plurality ensembles:")
    rows.append(eval_pack([load_net("DISTILL_v8", s) for s in SEEDS], full, atr_median,
                          "DISTILL_v8 VOTE5 (plurality)"))

    # 3) baseline parity check
    rows.append(eval_pack([load_net("VOTE5_v8_H256_DD", s) for s in SEEDS], full, atr_median,
                          "BASELINE VOTE5_v8_H256_DD"))

    out = CACHE / "results" / "distill_v8_eval.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
