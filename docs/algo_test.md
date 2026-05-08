# A3 — Algorithm Test (DQN / Double / Dueling / Double-Dueling)

> **TL;DR**: Algorithmic upgrades show a **regularization trade-off**. Vanilla DQN wins WF aggregate (+10.40) by a wide margin. **Double_Dueling (DD) wins out-of-sample**: best val (+6.12, highest ever for K=5 ensemble) and second-best test (+5.91), with 6/6 folds positive. Promoted as `BASELINE_VOTE5_DD` — the regularized alternative to `BASELINE_VOTE5`.

## Setup

Trained 15 new policies (5 seeds × 3 algos): Double DQN, Dueling DQN, Double_Dueling. Same hyperparameters as `BASELINE_VOTE5` (h=64, fee=0, trade-penalty=0.001) except for the algorithm modification.

| Algorithm | Bellman target | Network architecture |
|---|---|---|
| dqn (baseline) | max_a' Q_target(s', a') | DQN: 50→64→32→10 |
| double | Q_target(s', argmax_a' Q_online(s', a')) | DQN: 50→64→32→10 |
| dueling | max_a' Q_target(s', a') | DuelingDQN: V(s) + A(s,a) − mean(A) |
| double_dueling | Q_target(...) Double + Dueling combined | DuelingDQN |

Reproduction:
```bash
for algo in double dueling double_dueling; do
  for s in 42 7 123 0 99; do
    python3 -m models.dqn_selector btc \
        --tag VOTE5_${algo^^}_seed${s} --seed $s --algo $algo \
        --fee 0 --trade-penalty 0.001 --hidden 64
  done
done
python3 -m models.eval_algo
```

## Per-seed comparison (single-seed greedy WF Sharpe)

| seed | DQN | Double | Dueling | DD |
|---|---|---|---|---|
| 42 | **+9.03** | +5.38 | +6.53 | +6.49 |
| 7 | +5.60 | +7.97 | **+10.31** | +7.64 |
| 123 | **+9.63** | +6.15 | +6.64 | +7.45 |
| 0 | +8.26 | **+8.98** | +7.13 | +6.32 |
| 99 | +6.19 | **+7.12** | +5.72 | +5.09 |
| **mean** | **+7.74** | +7.13 | +7.27 | +6.60 |

Single-seed mean WF: DQN +7.74 best. Algorithmic variants don't lift average single-seed.

## K=5 plurality ensemble comparison

| metric | DQN (VOTE5) | Double | Dueling | **Double_Dueling** | Δdouble | Δdueling | Δdd |
|---|---|---|---|---|---|---|---|
| val Sharpe | +3.53 | −1.00 | +1.88 | **+6.12** | −4.53 | −1.66 | **+2.59** |
| val equity | 1.146 | 0.948 | 1.076 | **1.332** | −0.20 | −0.07 | +0.19 |
| test Sharpe | +4.19 | +3.76 | −1.73 | **+5.91** | −0.43 | −5.92 | +1.72 |
| test equity | 1.139 | 1.140 | 0.941 | **1.238** | +0.00 | −0.20 | +0.10 |
| **WF mean** | **+10.40** | +7.61 | +6.76 | +6.80 | −2.79 | −3.64 | −3.60 |
| WF folds + | 6/6 | 5/6 | 5/6 | 6/6 | — | — | — |

## Per-fold WF Sharpe

| Fold | DQN | Double | Dueling | DD |
|---|---|---|---|---|
| 1 | **+12.48** | +9.81 | +9.78 | +6.76 |
| 2 | **+16.24** | +10.47 | +12.44 | +13.06 |
| 3 | +9.95 | +8.16 | **+10.21** | +4.19 |
| 4 | +14.19 | **+14.73** | +10.31 | +6.78 |
| 5 | +4.35 | −0.33 | +1.32 | **+5.43** |
| 6 | +5.20 | +2.79 | −3.50 | +4.58 |

## Mechanistic interpretation

**The regularization trade-off in this domain**:
- Vanilla DQN → maximum bias toward fitting training distribution → high WF aggregate where folds 1–4 are training-similar
- Double DQN target → reduces overestimation bias → milder fit to training distribution → lower WF, no obvious OOD lift
- Dueling network → V(s) + A(s,a) decomposition → smoother value estimates → reduces overfitting → lower WF, mixed OOD
- Double + Dueling combined → strongest regularization → BEST out-of-sample (val/test), lowest WF

**This is the same trade-off as A1 capacity, in reverse**: A1 showed adding capacity boosts WF aggregate but degrades fold 6 (most-recent fold). A3 shows adding regularization degrades WF aggregate but boosts val/test.

The right operating point depends on what regime you expect to face forward. DD's higher val/test (which haven't been seen by training) is the more honest OOD measure.

## Decision: BASELINE_VOTE5_DD as the regularized baseline

Promoted as a formal baseline alongside others. Best for "regime change is likely" scenarios.

| Spec | Value |
|---|---|
| Aggregation | Plurality (most-voted; tie → NO_TRADE) |
| Constituents | 5 BASELINE_FULL policies (Double_Dueling), seeds 42/7/123/0/99 |
| Underlying nets | `cache/btc_dqn_policy_VOTE5_DD_seed{42,7,123,0,99}.pt` |
| Algorithm | Double DQN target + Dueling network |
| Hidden size | 64 (same as BASELINE_VOTE5) |
| WF mean Sharpe | +6.80 |
| WF folds positive | 6/6 |
| **DQN-val Sharpe** | **+6.12** (best of any K=5) |
| DQN-test Sharpe | +5.91 |
| Fold 6 Sharpe | +4.58 |

## Why pure Double or Dueling alone don't help

- **Double alone**: only fixes overestimation in target. Network architecture unchanged → can still overfit on action evaluation. Modest regularization, mostly hurts WF (−2.79).
- **Dueling alone**: V/A decomposition smooths but actions still chosen by raw network argmax with full bias. WF folds 1, 2, 4 hurt by smoothing without enough OOD compensation. Worst test Sharpe (−1.73) of any variant.
- **Double + Dueling combined**: the two regularizations stack productively. The Double target prevents the Dueling network's V-head from being noisy; the Dueling decomposition stabilizes the action choice that Double evaluates. Combined effect transcends individual contributions.

## Files

| File | Contents |
|---|---|
| [models/eval_algo.py](../models/eval_algo.py) | per-seed + ensemble eval across all 4 algorithms |
| [models/dqn_network.py](../models/dqn_network.py) | DuelingDQN class (lines 38-60) |
| [models/dqn_selector.py](../models/dqn_selector.py) | --algo flag (dqn/double/dueling/double_dueling) |
| `cache/btc_dqn_policy_VOTE5_DOUBLE_seed*.pt`, `_DUELING_*.pt`, `_DD_*.pt` | 15 trained policies |
| `cache/eval_algo_results.json` | aggregated metrics |

## Implications for future work

1. **Algorithmic upgrades alone don't beat BASELINE_VOTE5 on WF aggregate.** The +10.40 ceiling is robust across DQN/Double/Dueling/DD.

2. **DD provides genuine OOD lift on val/test single-shot.** This may be more meaningful than WF aggregate for forward deployment.

3. **Combining DD with capacity (h=256 + double_dueling)** is the natural next experiment. If DD's regularization counteracts h=256's overfitting (fold 6 +0.41 → ?), we might get the best of both.

4. **Consider DD as the "production-conservative" baseline** — better OOD properties for live deployment risk profiles.
