"""
Evaluate fee-aware retrained policies (FEE4_p001, FEE4_p005) as K=5 plurality
ensembles at the real OKX taker fee (0.00045 = 4.5 bp/side).

Compare against:
  - vanilla VOTE5 baseline at fee=0.00045
  - vanilla VOTE5 + top-5 + vote≥3 (current best filter) at fee=0.00045
"""
import json, pathlib, statistics, time
import numpy as np
import torch

from models.dqn_network          import DQN
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL
from models.audit_vote5_dd       import run_fold, run_walkforward, load_full_rl_period

CACHE = pathlib.Path("cache")
N_FOLDS = 6
SEEDS = [42, 7, 123, 0, 99]
REAL_FEE = 0.00045   # OKX taker per side


def load_dqn(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def eval_config(name, nets, full, atr_median, fees, vote_thr=1, allowlist=None):
    rows_summary = []
    for fee in fees:
        rows_f, trades_f = run_walkforward(nets, full, atr_median, fee=fee,
                                              vote_threshold=vote_thr,
                                              strat_allowlist=allowlist, with_reason=False)
        wf = statistics.mean(r["sharpe"] for r in rows_f)
        wf_pos = sum(1 for r in rows_f if r["sharpe"] > 0)
        sp_val = np.load(CACHE / "state" / "btc_dqn_state_val.npz")
        sp_test = np.load(CACHE / "state" / "btc_dqn_state_test.npz")
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_val["price"], sp_val["atr"], atr_median)
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_test["price"], sp_test["atr"], atr_median)
        _, val_sh, _, _ = run_fold(sp_val["state"], sp_val["valid_actions"], sp_val["signals"],
                                       sp_val["price"], sp_val["atr"], sp_val["regime_id"], sp_val["ts"],
                                       tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=fee,
                                       vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        _, test_sh, _, _ = run_fold(sp_test["state"], sp_test["valid_actions"], sp_test["signals"],
                                        sp_test["price"], sp_test["atr"], sp_test["regime_id"], sp_test["ts"],
                                        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=fee,
                                        vote_threshold=vote_thr, strat_allowlist=allowlist, fold_id=0)
        rows_summary.append(dict(name=name, fee=fee, fee_bp=fee*10000,
                                    trades=len(trades_f), wf=wf, wf_pos=wf_pos,
                                    val=val_sh, test=test_sh,
                                    per_fold=[r["sharpe"] for r in rows_f]))
        print(f"  {name:<35} fee={fee*10000:>4.1f}bp  trades={len(trades_f):>5}  "
              f"wf={wf:>+7.3f} ({wf_pos}/6)  val={val_sh:>+7.2f}  test={test_sh:>+7.2f}")
    return rows_summary


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period()

    print(f"\n{'='*120}\n  FEE-AWARE RETRAIN EVALUATION  (real taker = {REAL_FEE*10000:.2f}bp)\n{'='*120}")

    fees = [0.0, 0.0004, REAL_FEE]   # fee=0 (ceiling), train fee, real taker
    all_results = []

    # ── vanilla VOTE5 (baseline) ─────────────────────────────────────────────
    print(f"\n[1/4] Vanilla BASELINE_VOTE5 (no filter)")
    nets_van = [load_dqn("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}") for s in SEEDS]
    all_results += eval_config("vanilla VOTE5 (no filter)", nets_van, full, atr_median, fees)

    # ── vanilla VOTE5 + top-5 + vote≥3 (current best at 4bp) ─────────────────
    print(f"\n[2/4] Vanilla BASELINE_VOTE5 + top-5 (S1,S4,S7,S8,S10) + vote≥3")
    all_results += eval_config("vanilla VOTE5 + top-5 + vote≥3", nets_van, full, atr_median, fees,
                                  vote_thr=3, allowlist={0, 3, 5, 6, 7})

    # ── FEE4_p001 (fee-aware, penalty=0.001) ─────────────────────────────────
    print(f"\n[3/4] FEE4_p001 (fee=4bp, trade_penalty=0.001) — natively fee-aware")
    nets_p001 = [load_dqn(f"FEE4_p001_seed{s}") for s in SEEDS]
    all_results += eval_config("FEE4_p001 (no filter)", nets_p001, full, atr_median, fees)
    all_results += eval_config("FEE4_p001 + vote≥3", nets_p001, full, atr_median, fees, vote_thr=3)

    # ── FEE4_p005 (fee-aware + heavier penalty) ──────────────────────────────
    print(f"\n[4/4] FEE4_p005 (fee=4bp, trade_penalty=0.005) — combined pressure")
    nets_p005 = [load_dqn(f"FEE4_p005_seed{s}") for s in SEEDS]
    all_results += eval_config("FEE4_p005 (no filter)", nets_p005, full, atr_median, fees)
    all_results += eval_config("FEE4_p005 + vote≥3", nets_p005, full, atr_median, fees, vote_thr=3)

    out = CACHE / "results" / "eval_fee_aware_results.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
