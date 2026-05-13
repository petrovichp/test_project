"""
Step 2 — measure impact of masking S12 and/or S6 at inference time on
VOTE5_v8_H256_DD K=5.

Strategy index mapping (in 12-action space, action 0 = NO_TRADE,
actions 1-11 = strategies in order):
  action 1 = S1_VolDir
  action 2 = S2_Funding
  action 3 = S3_BBExt
  action 4 = S4_MACD
  action 5 = S6_TwoSignal
  action 6 = S7_OIDiverg
  action 7 = S8_TakerSus
  action 8 = S10_Squeeze
  action 9 = S12_VWAPVol
  action 10 = S11_Basis
  action 11 = S13_OBDiv

Test runtime ablations:
  - baseline (no mask)
  - ablate S12 (action 9)
  - ablate S6 (action 5)
  - ablate both S12 + S6 (actions 5, 9)

If WF Sharpe is unchanged or improves, then dropping these strategies
permanently is safe (and could be done in a future state-pack rebuild).

Run: python3 -m models.action_ablate_s12_s6
"""
import json, statistics, time, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.audit_vote5_dd import run_walkforward

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
TAKER_FEE = 0.00045
RL_START = 100_000
RL_END   = 383_174
N_FOLDS  = 6
FOLD_SIZE = (RL_END - RL_START) // N_FOLDS
N_BARS_PER_YEAR = 525_960


def load_v8_nets():
    out = []
    for s in SEEDS:
        n = DuelingDQN(52, 12, 256)
        n.load_state_dict(torch.load(
            CACHE / "policies" / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt", map_location="cpu"))
        n.eval(); out.append(n)
    return out


def load_full():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def run_with_mask(nets, full, atr_median, ablate_actions, label):
    """Run walk-forward applying a runtime mask to disable specified actions.
    Patches full['valid_actions'] in a copy."""
    full_masked = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in full.items()}
    if ablate_actions:
        for a in ablate_actions:
            full_masked["valid_actions"][:, a] = False
    rows, trades = run_walkforward(nets, full_masked, atr_median, fee=TAKER_FEE, with_reason=True)
    wf = statistics.mean(r["sharpe"] for r in rows)
    pos = sum(1 for r in rows if r["sharpe"] > 0)
    per_fold = [round(r["sharpe"], 2) for r in rows]
    n = len(trades)
    # AGGRESSIVE quadratic sizing (R1 winner) — same applied to all variants
    by_v = Counter(tr["votes_count"] for tr in trades)
    # Compute sized WF
    by_fold = defaultdict(list)
    for tr in trades:
        sized = tr["pnl"] * ((tr["votes_count"] - 2) / 3) ** 2
        tr["pnl_sized"] = sized
        by_fold[tr["fold"]].append(tr)
    pf_sized = []
    for fid in range(1, N_FOLDS + 1):
        tl = sorted(by_fold.get(fid, []), key=lambda t: t["t_close"])
        n_bars = FOLD_SIZE if fid < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * FOLD_SIZE
        eq = np.full(n_bars, 1.0); cur = 1.0
        for tr in tl:
            tc = int(tr["t_close"])
            if 0 <= tc < n_bars:
                cur *= 1.0 + tr["pnl_sized"]
                eq[tc:] = cur
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sh = float(rets.mean() / rets.std() * math.sqrt(N_BARS_PER_YEAR)) if rets.std() > 1e-12 else 0.0
        pf_sized.append(sh)
    wf_sized = statistics.mean(pf_sized)
    pos_sized = sum(1 for s in pf_sized if s > 0)
    print(f"  {label:<32} | fixed WF {wf:>+7.3f}  pos {pos}/6  trades {n:>4} | sized WF {wf_sized:>+7.3f}  pos {pos_sized}/6")
    return dict(label=label, n_trades=n, wf_fixed=wf, wf_sized=wf_sized,
                 pos_fixed=pos, pos_sized=pos_sized, per_fold=per_fold,
                 per_fold_sized=[round(s, 2) for s in pf_sized])


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()
    nets = load_v8_nets()

    print(f"\n{'='*120}\n  Action-space ablation @ fee=4.5bp (R1 AGGRESSIVE sizing applied)\n{'='*120}\n")

    # Action indices in the 12-action space (action 0 = NO_TRADE)
    # Following STRAT_KEYS order in audit code:
    # 1=S1, 2=S2, 3=S3, 4=S4, 5=S6, 6=S7, 7=S8, 8=S10, 9=S12, 10=S11, 11=S13
    S6_ACTION  = 5
    S12_ACTION = 9

    results = []
    results.append(run_with_mask(nets, full, atr_median, [], "BASELINE (no mask)"))
    results.append(run_with_mask(nets, full, atr_median, [S12_ACTION], "ablate S12 (act 9)"))
    results.append(run_with_mask(nets, full, atr_median, [S6_ACTION], "ablate S6 (act 5)"))
    results.append(run_with_mask(nets, full, atr_median, [S6_ACTION, S12_ACTION], "ablate S6 + S12"))

    out = CACHE / "results" / "action_ablate_s6_s12.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
