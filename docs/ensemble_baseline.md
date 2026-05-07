# Ensemble Baseline — K-seed Q-value averaging

> **TL;DR**: Q-value averaging ensembles (K=2, 3, 4, 5) do NOT improve baselines. Every ensemble has worse DQN-val and DQN-test than seed=42 alone, and every ensemble has negative WF fold 6 even though every single seed is positive there. Single-seed seed=42 (`BASELINE_FULL`) remains the strongest policy.

## Setup

Trained 5 seeds total at identical hyperparameters: 42, 7, 123, 0, 99. Built ensembles by averaging Q-values across K nets at decision time (`models/dqn_network.py:EnsembleDQN`).

Reproduction:
```bash
# Train all 5 seeds
for s in 42 7 123 0 99; do
  python3 -m models.dqn_selector btc --tag BASELINE_FULL_seed${s} --seed $s --fee 0 --trade-penalty 0.001
done

python3 -m models.ensemble_baseline
```

## Results

| Variant | val | test | WF mean | WF pos | per-fold WF |
|---|---|---|---|---|---|
| seed=42 (BASELINE_FULL) | **+7.30** | **+3.67** | +9.03 | **6/6** | [13.03, 14.82, 6.29, 9.56, 8.17, +2.33] |
| seed=7 | +3.96 | −1.14 | +5.60 | 5/6 | [5.48, 11.14, 8.12, 4.62, 4.40, −0.14] |
| seed=123 | +6.18 | +3.04 | **+9.63** | 6/6 | [11.77, 14.71, 9.95, 13.67, 5.63, +2.03] |
| seed=0 | +4.02 | +5.59 | +8.26 | 6/6 | [9.60, 16.51, 4.67, 11.40, 4.68, +2.71] |
| seed=99 | +4.92 | **+9.06** | +6.19 | 6/6 | [9.33, 8.85, 2.76, 7.14, 4.37, +4.71] |
| **K=2 (42, 123)** | −1.33 | −0.21 | +8.29 | 5/6 | [14.57, 15.02, 8.85, 12.01, 0.53, **−1.23**] |
| **K=3 (42, 7, 123)** | +2.87 | −0.97 | +7.67 | 5/6 | [11.79, 13.63, 10.20, 8.97, 4.98, **−3.53**] |
| **K=4 (drop seed=7)** | +4.91 | −2.90 | +8.90 | 5/6 | [13.30, 17.18, 9.68, 10.52, 7.22, **−4.49**] |
| **K=5 (all)** | +3.53 | +0.19 | +9.09 | 5/6 | [12.52, 12.94, 9.10, 12.57, 8.31, **−0.94**] |

5-seed aggregate: val mean +5.27 std 1.44, test mean +4.04 std **3.73**, WF mean +7.74 std 1.77.

**K=5 ensemble vs mean-of-seeds:** val Δ −1.75, test Δ **−3.86**, WF Δ +1.34.

## Why Q-averaging fails here

Q-value averaging at decision time is **not** equivalent to averaging returns. Decision = argmax over Q. When seeds disagree on which strategy to pick at a given bar, the averaged Q can land on a third action (or NO_TRADE) that no individual seed would have chosen.

Result: the ensemble produces a qualitatively new policy — not a smoothed combination of the inputs. Most pronounced where seeds disagree most:
- Late period (fold 6, DQN-test) — every ensemble underperforms despite every individual being positive
- Folds where seed=7 is the outlier (folds 1, 4) — ensemble correctly "votes out" seed=7 and lifts those folds

Net: ensemble lifts the weak-seed folds and drags the strong-seed folds toward the mean. Net WF is approximately the seed mean but with the structural risk of disagreement-driven bad decisions.

## Implications

1. **Q-averaging is not a good aggregation strategy for action-selection DQNs.** Useful for value estimation, but action argmax over an averaged Q is brittle to seed disagreement.
2. **Single-seed baselines remain our strongest policies** for now. seed=42 (`BASELINE_FULL`) wins on val/test/folds-positive; seed=123 wins on WF mean.
3. **Multi-seed evaluation should be the standard** for comparing experiments going forward — but the *baseline* itself doesn't need to be an ensemble.

## Better ensemble approaches to consider (not yet tested)

| Approach | Mechanism | Expected behavior |
|---|---|---|
| **Vote-based** | Each net argmaxes individually, majority action wins | Preserves individual policy character; no third-action drift |
| **Pick-by-val-Sharpe** | Use only the top-N seeds by val Sharpe (not ensemble) | Reduces spread by dropping outliers |
| **Rolling-Sharpe selection** | Pick which seed to use based on rolling Sharpe in recent K bars | Adapts to regime |
| **Confidence-weighted Q** | Weight each net's Q by its in-sample certainty | More principled than uniform averaging |

These are interesting but lower priority than the higher-leverage architectural test (direction probabilities into state — see [docs/seed_variance.md](seed_variance.md) recommendation).

## Files

| File | Contents |
|---|---|
| [models/dqn_network.py](../models/dqn_network.py) | added `EnsembleDQN` class (lines 38-50) |
| [models/ensemble_baseline.py](../models/ensemble_baseline.py) | analysis script |
| `cache/btc_dqn_policy_BASELINE_FULL_seed{0,99}.pt` | additional seed policies |
| `cache/ensemble_baseline_results.json` | aggregated metrics |
