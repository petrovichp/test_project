"""
Seed-variance analysis for BASELINE_FULL.

Evaluates 3 policies (seeds 42, 7, 123) on:
  - DQN-val   (50,867 bars)
  - DQN-test  (52,307 bars, locked)
  - Walk-forward (6 folds across full RL period)

Outputs mean ± std across seeds for every metric.
"""
import json, pathlib, statistics
import numpy as np
import torch

from models.dqn_network  import DQN
from models.dqn_selector import evaluate_policy
from models.dqn_rollout  import _build_exit_arrays

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123]
TAGS  = {42: "BASELINE_FULL", 7: "BASELINE_FULL_seed7", 123: "BASELINE_FULL_seed123"}


def _load_policy(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def eval_split(tag: str, split: str, atr_median: float):
    sp = np.load(CACHE / f"btc_dqn_state_{split}.npz")
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    out = evaluate_policy(_load_policy(tag), sp["state"], sp["valid_actions"],
                           sp["signals"], sp["price"], tp, sl, tr, tab, be, ts,
                           valid_mask_override=None, fee=0.0)
    return dict(sharpe=out["sharpe"], equity=out["equity_final"],
                trades=out["n_trades"], win_rate=out["win_rate"], max_dd=out["max_dd"])


def load_wf(tag_suffix: str):
    """tag_suffix: '' for seed=42 (uses verify_baseline), 'seed7'/'seed123' otherwise."""
    if tag_suffix == "":
        f = CACHE / "btc_groupC2_walkforward_verify_baseline.json"
    else:
        f = CACHE / f"btc_groupC2_walkforward_{tag_suffix}.json"
    d = json.loads(f.read_text())
    return [r["rule_sharpe"] for r in d["rows"]], [r["rule_eq"] for r in d["rows"]], \
           [r["rule_trades"] for r in d["rows"]]


def stats(xs, prec=3):
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def main():
    vol = np.load(CACHE / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    rows = {}
    for seed in SEEDS:
        tag = TAGS[seed]
        suffix = "" if seed == 42 else (f"seed{seed}")
        wf_sharpes, wf_eqs, wf_trades = load_wf(suffix)
        v = eval_split(tag, "val",  atr_median)
        t = eval_split(tag, "test", atr_median)
        rows[seed] = dict(
            train_val_best=json.loads((CACHE / f"btc_dqn_train_history_{tag}.json").read_text())["best_val_sharpe"],
            val_sharpe=v["sharpe"],   val_eq=v["equity"],   val_trades=v["trades"],
            test_sharpe=t["sharpe"],  test_eq=t["equity"],  test_trades=t["trades"],
            wf_sharpes=wf_sharpes, wf_eqs=wf_eqs, wf_trades=wf_trades,
            wf_mean=statistics.mean(wf_sharpes),
            wf_median=statistics.median(wf_sharpes),
            wf_pos=sum(1 for s in wf_sharpes if s > 0),
        )

    # ── per-seed table ──────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("  SEED VARIANCE — BASELINE_FULL (3 seeds)")
    print(f"{'='*120}")
    print(f"\n{'seed':<6} {'train-val best':>15} {'val Sharpe':>12} {'test Sharpe':>12} "
          f"{'val eq':>8} {'test eq':>8} {'WF mean':>10} {'WF med':>10} {'WF pos':>7}  per-fold WF")
    print("-"*150)
    for seed in SEEDS:
        r = rows[seed]
        print(f"{seed:<6} {r['train_val_best']:>+15.3f} {r['val_sharpe']:>+12.3f} {r['test_sharpe']:>+12.3f} "
              f"{r['val_eq']:>8.3f} {r['test_eq']:>8.3f} {r['wf_mean']:>+10.3f} {r['wf_median']:>+10.3f} "
              f"{r['wf_pos']:>3}/6  {[round(s, 2) for s in r['wf_sharpes']]}")

    # ── aggregate stats ─────────────────────────────────────────────────────
    metrics = [
        ("train-val best", "train_val_best"),
        ("val Sharpe",     "val_sharpe"),
        ("test Sharpe",    "test_sharpe"),
        ("val equity",     "val_eq"),
        ("test equity",    "test_eq"),
        ("WF mean Sharpe", "wf_mean"),
        ("WF median Sharpe","wf_median"),
        ("WF folds positive","wf_pos"),
    ]
    print(f"\n{'metric':<22} {'mean':>10} {'std':>9} {'min':>9} {'max':>9} {'spread':>9}")
    print("-"*75)
    for name, key in metrics:
        xs = [rows[s][key] for s in SEEDS]
        m, sd = stats(xs)
        print(f"{name:<22} {m:>+10.3f} {sd:>9.3f} {min(xs):>+9.3f} {max(xs):>+9.3f} {max(xs)-min(xs):>+9.3f}")

    # ── per-fold variance ───────────────────────────────────────────────────
    print(f"\nPer-fold WF Sharpe variance (across seeds):")
    print(f"{'fold':<6} " + " ".join(f"{f'seed={s}':>10}" for s in SEEDS) + f"{'mean':>10}{'std':>10}{'spread':>10}")
    print("-"*80)
    for fi in range(6):
        per_seed = [rows[s]["wf_sharpes"][fi] for s in SEEDS]
        m, sd = stats(per_seed)
        print(f"{fi+1:<6} " + " ".join(f"{x:>+10.3f}" for x in per_seed) +
              f"{m:>+10.3f}{sd:>10.3f}{max(per_seed)-min(per_seed):>+10.3f}")

    # save
    out = CACHE / "seed_variance_results.json"
    out.write_text(json.dumps({str(k): v for k, v in rows.items()}, indent=2, default=str))
    print(f"\n  → {out.name}")


if __name__ == "__main__":
    main()
