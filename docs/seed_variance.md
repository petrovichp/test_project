# Seed Variance Analysis — BASELINE_FULL

> **TL;DR**: 3 seeds (42, 7, 123) trained from scratch with identical hyperparameters. Walk-forward mean Sharpe std = **±2.17**. DQN-test Sharpe std = **±2.61** with seed=7 producing a NEGATIVE test Sharpe. Most of our prior single-seed Δ-vs-baseline conclusions are within seed noise.

## Setup

- Same train command, same hyperparameters, only `--seed` differs.
- 3 seeds: 42 (original `BASELINE_FULL`), 7, 123.
- Each policy evaluated on:
  - DQN-val (50,867 bars, ~35 days)
  - DQN-test (52,307 bars, locked, ~36 days)
  - Walk-forward across 6 RL folds (full RL period, 283k bars)

Reproduction:
```bash
python3 -m models.dqn_selector btc --tag BASELINE_FULL_seed7   --seed 7   --fee 0 --trade-penalty 0.001
python3 -m models.dqn_selector btc --tag BASELINE_FULL_seed123 --seed 123 --fee 0 --trade-penalty 0.001
python3 -m models.group_c2_walkforward --policy-tag BASELINE_FULL_seed7   --no-b5 --out-tag seed7
python3 -m models.group_c2_walkforward --policy-tag BASELINE_FULL_seed123 --no-b5 --out-tag seed123
python3 -m models.seed_variance
```

## Results

### Per-seed metrics

| seed | train-val best | val Sharpe | test Sharpe | val eq | test eq | WF mean | WF median | WF pos | per-fold WF Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| 42  | +7.295 | +7.295 | +3.666 | 1.398 | 1.127 | **+9.034** | +8.865 | **6/6** | [13.03, 14.82, 6.29, 9.56, 8.17, 2.33] |
| 7   | +3.963 | +3.963 | **−1.137** | 1.188 | 0.955 | +5.603 | +5.052 | 5/6 | [5.48, 11.14, 8.12, 4.62, 4.40, −0.14] |
| 123 | +6.178 | +6.178 | +3.035 | 1.324 | 1.099 | +9.625 | +10.858 | 6/6 | [11.77, 14.71, 9.95, 13.67, 5.63, 2.03] |

### Aggregate variance

| metric | mean | std | min | max | spread |
|---|---|---|---|---|---|
| train-val best | +5.812 | 1.696 | +3.963 | +7.295 | 3.332 |
| val Sharpe | +5.812 | 1.696 | +3.963 | +7.295 | 3.332 |
| test Sharpe | +1.855 | **2.610** | −1.137 | +3.666 | 4.803 |
| val equity | 1.303 | 0.106 | 1.188 | 1.398 | 0.210 |
| test equity | 1.061 | 0.093 | 0.955 | 1.127 | 0.172 |
| WF mean Sharpe | +8.088 | **2.172** | +5.603 | +9.625 | 4.022 |
| WF median Sharpe | +8.258 | 2.950 | +5.052 | +10.858 | 5.806 |
| WF folds positive | 5.667 | 0.577 | 5 | 6 | 1 |

### Per-fold variance (across seeds)

| Fold | seed=42 | seed=7 | seed=123 | mean | std | spread |
|---|---|---|---|---|---|---|
| 1 | +13.03 | +5.48 | +11.77 | +10.09 | **4.04** | 7.55 |
| 2 | +14.82 | +11.14 | +14.71 | +13.56 | 2.09 | 3.68 |
| 3 | +6.29 | +8.12 | +9.95 | +8.12 | 1.83 | 3.66 |
| 4 | +9.56 | +4.62 | +13.67 | +9.29 | **4.53** | 9.05 |
| 5 | +8.17 | +4.40 | +5.63 | +6.06 | 1.93 | 3.77 |
| 6 | +2.33 | −0.14 | +2.03 | +1.41 | 1.35 | 2.48 |

Fold-1 and fold-4 have the highest cross-seed variance (~4.5 Sharpe std). Fold 6 (most recent, hardest regime) has the lowest variance.

## What's robust to seed

| Statement | Holds across all 3 seeds? |
|---|---|
| WF mean > 0 | yes (5.6, 9.0, 9.6) |
| WF median > 0 | yes (5.1, 8.9, 10.9) |
| ≥5/6 folds positive | yes |
| Beats BTC buy-and-hold on val | yes (eq 1.19, 1.32, 1.40 vs 1.07) |
| Beats BTC buy-and-hold on test | **no** (eq 0.96, 1.10, 1.13 vs 1.09 — seed=7 loses to BH) |
| Train-val best > +3 | yes |

**The signal is real** — every seed produces a positive WF mean. But the magnitude is highly variable.

## What's NOT robust to seed

- Single-seed test Sharpe (std 2.6 — ranges from negative to +3.7).
- Per-fold Sharpe (folds 1, 4 have std > 4).
- Single-seed deltas vs alternatives (Δ < 2 Sharpe is within noise).

## Implications for prior conclusions

Many of the audit follow-up "drop" verdicts were single-seed Δs near or below seed noise:

| Prior claim | Δ vs BASELINE_FULL | vs WF std (2.17) | New verdict |
|---|---|---|---|
| Ablate S6 (eval-only) | −0.47 | 0.22σ | within noise |
| Ablate S6 (retrain) | −1.64 | 0.76σ | within noise |
| Ablate S7 (eval-only) | −1.39 | 0.64σ | within noise |
| Ablate S7 (retrain) | −2.04 | 0.94σ | borderline |
| Ablate S10 (eval-only) | −1.44 | 0.66σ | within noise |
| Ablate S10 (retrain) | −3.17 | 1.46σ | borderline-real |
| Triple ablation (LEAN) | −2.28 | 1.05σ | borderline |
| TP × 0.85 | −0.96 | 0.44σ | within noise |
| TP × 0.70 | −0.60 | 0.28σ | within noise |
| LEAN wins DQN-test (+5.19 vs +3.67) | +1.52 (test std 2.61) | 0.58σ | within noise |

**Most "drop" decisions were not statistically supported.** The audit follow-up's headline ("all 5+3 perturbations degrade baseline") becomes "most perturbations produce Δs within seed noise; some borderline."

This does not invalidate the conclusion that we found no clear winner — but it means the *strength* of those rejections was overstated.

## Recommendations

1. **Adopt multi-seed evaluation as the new baseline standard.** Single-seed Δs < 2 Sharpe should be treated as inconclusive.

2. **Build an ensemble baseline**: average Q-values across 3+ seed-trained policies at decision time. Likely smooths fold-level variance and may beat any individual seed's WF mean.

3. **Re-baseline future experiments against the multi-seed mean ± std**, not the single-seed +9.034.

4. **Don't deploy a single-seed policy live.** Seed=7 producing −1.14 on locked test is a strong argument against any single-policy deployment.

5. **Re-test the borderline-real cases** (S10 retrain, triple-ablation) under multi-seed protocol — they may or may not actually degrade.

## Files

| File | Contents |
|---|---|
| [models/seed_variance.py](../models/seed_variance.py) | analysis script |
| `cache/btc_dqn_policy_BASELINE_FULL_seed7.pt` | seed=7 policy weights |
| `cache/btc_dqn_policy_BASELINE_FULL_seed123.pt` | seed=123 policy weights |
| `cache/btc_dqn_train_history_BASELINE_FULL_seed{7,123}.json` | training histories |
| `cache/btc_groupC2_walkforward_seed{7,123}.json` | per-seed walk-forward results |
| `cache/seed_variance_results.json` | aggregated metrics |
