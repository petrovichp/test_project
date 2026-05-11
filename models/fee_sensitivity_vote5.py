"""
Vanilla BASELINE_VOTE5 fee sensitivity + trade-count reduction.
Mirror of audit_vote5_dd.py Parts B & C but for vanilla DQN.
Compare against VOTE5_DD to see if vanilla VOTE5 is more fee-robust
(it has fewer trades: 1,122 vs 1,437 → less fee drag per √N).
"""
import json, pathlib, statistics, time
from collections import Counter
import numpy as np
import torch

from models.dqn_network          import DQN
from models.dqn_rollout          import _build_exit_arrays, STRAT_KEYS
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.audit_vote5_dd       import run_fold, run_walkforward, load_full_rl_period

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS = [42, 7, 123, 0, 99]


def load_dqn(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()
    nets = [load_dqn("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS]

    print(f"\n{'='*120}\n  BASELINE_VOTE5 (vanilla DQN) FEE & TRADE-REDUCTION\n{'='*120}")

    # PART B: fee sensitivity
    print(f"\n{'='*60}\n  Fee sensitivity\n{'='*60}")
    fee_levels = [0.0, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.0012, 0.0020]
    print(f"\n  {'fee (per side)':>15} {'fee bp':>7} {'WF mean':>10} {'val Sharpe':>11} "
          f"{'test Sharpe':>12} {'WF folds+':>10} {'trades':>8}")
    print("  " + "-"*100)
    fee_results = []
    for fee in fee_levels:
        rows_f, trades_f = run_walkforward(nets, full, atr_median, fee=fee, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in rows_f)
        wf_pos = sum(1 for r in rows_f if r["sharpe"] > 0)
        sp_val = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
        sp_test = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_val["price"], sp_val["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_test["price"], sp_test["atr"], atr_median)
        _, val_sh, _, _ = run_fold(sp_val["state"], sp_val["valid_actions"], sp_val["signals"],
                                       sp_val["price"], sp_val["atr"], sp_val["regime_id"], sp_val["ts"],
                                       tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee, fold_id=0)
        _, test_sh, _, _ = run_fold(sp_test["state"], sp_test["valid_actions"], sp_test["signals"],
                                        sp_test["price"], sp_test["atr"], sp_test["regime_id"], sp_test["ts"],
                                        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee, fold_id=0)
        fee_results.append(dict(fee=fee, wf_mean=wf, wf_pos=wf_pos,
                                  val_sharpe=val_sh, test_sharpe=test_sh,
                                  trades=len(trades_f)))
        print(f"  {fee:>15.4f} {fee*10000:>7.1f} {wf:>+10.3f} {val_sh:>+11.3f} "
              f"{test_sh:>+12.3f} {wf_pos:>3}/6     {len(trades_f):>8}")

    # PART C: trade reduction
    print(f"\n\n{'='*60}\n  Trade-count reduction at fee=4bp\n{'='*60}")
    test_fee = 0.0004
    print(f"\n  {'strategy':<35} {'trades':>7} {'val Sh':>9} {'test Sh':>9} {'WF':>9} {'folds+':>7}")
    print("  " + "-"*90)
    rc_results = []

    def measure(name, vote_thr=1, allowlist=None):
        rows_r, trades_r = run_walkforward(nets, full, atr_median, fee=test_fee,
                                              vote_threshold=vote_thr,
                                              strat_allowlist=allowlist, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in rows_r)
        wf_pos = sum(1 for r in rows_r if r["sharpe"] > 0)
        sp_val = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
        sp_test = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_val["price"], sp_val["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_test["price"], sp_test["atr"], atr_median)
        _, val_sh, _, _ = run_fold(sp_val["state"], sp_val["valid_actions"], sp_val["signals"],
                                       sp_val["price"], sp_val["atr"], sp_val["regime_id"], sp_val["ts"],
                                       tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=test_fee,
                                       vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        _, test_sh, _, _ = run_fold(sp_test["state"], sp_test["valid_actions"], sp_test["signals"],
                                        sp_test["price"], sp_test["atr"], sp_test["regime_id"], sp_test["ts"],
                                        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=test_fee,
                                        vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        rc_results.append(dict(name=name, vote_thr=vote_thr,
                                  allowlist=list(allowlist) if allowlist else None,
                                  trades=len(trades_r), wf_mean=wf, wf_pos=wf_pos,
                                  val_sharpe=val_sh, test_sharpe=test_sh))
        print(f"  {name:<35} {len(trades_r):>7} {val_sh:>+9.2f} {test_sh:>+9.2f} {wf:>+9.3f} {wf_pos:>4}/6")

    measure("baseline (no filtering)")
    measure("vote ≥ 3", vote_thr=3)
    measure("vote ≥ 4", vote_thr=4)
    measure("vote ≥ 5", vote_thr=5)
    # vanilla VOTE5 audit (A4) showed S6, S10, S7 worst — drop those + S2, S3, S12
    measure("ablate S6 (vote5 audit)",            allowlist={0, 1, 2, 3, 5, 6, 7, 8})
    measure("ablate S6+S10 (vote5 audit)",        allowlist={0, 1, 2, 3, 5, 6, 8})
    measure("ablate S6+S7+S10 (triple)",          allowlist={0, 1, 2, 3, 6, 8})
    measure("top-5 (S1,S4,S7,S8,S10)",            allowlist={0, 3, 5, 6, 7})
    measure("top-3 (S1, S7, S8)",                 allowlist={0, 5, 6})
    measure("top-5 + vote ≥ 3", vote_thr=3,       allowlist={0, 3, 5, 6, 7})

    out = CACHE / "results" / "fee_sensitivity_vote5_results.json"
    out.write_text(json.dumps(dict(
        fee_sensitivity=fee_results,
        trade_reduction=rc_results,
    ), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
