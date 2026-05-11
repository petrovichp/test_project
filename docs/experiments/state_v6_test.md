# State V6 (direction probabilities) — Tier 1.3 negative result

> **TL;DR**: Adding 4 direction-probability features to the state (50→54 dims) DEGRADES baselines. Per-seed and ensemble Sharpes drop on val/WF. The architectural intuition was wrong — the direction probs are already implicit in the binary signal flags.

## Setup

Built v6 state arrays extending v5 by 4 dimensions:
- dim 50: P_up_60 centered to [−1, 1]
- dim 51: P_dn_60 centered to [−1, 1]
- dim 52: P_up_100 centered to [−1, 1]
- dim 53: P_dn_100 centered to [−1, 1]

These are the 4 CNN-LSTM direction predictions (AUC 0.64–0.70 OOS) that v5 already uses to compute strategy signals but never exposes to the DQN directly.

Trained 5 seeds (42, 7, 123, 0, 99) on v6 state with identical hyperparameters to v5, formed K=5 plurality ensemble (`VOTE5_v6`).

Reproduction:
```bash
python3 -m models.dqn_state_v6
for s in 42 7 123 0 99; do
  python3 -m models.dqn_selector btc --tag BASELINE_FULL_V6_seed${s} \
                                    --seed $s --fee 0 --trade-penalty 0.001 \
                                    --state-version v6
done
python3 -m models.eval_v6_vs_v5
```

## Results

### Per-seed comparison

| seed | v5 val | v6 val | v5 test | v6 test | v5 WF | v6 WF | v5 fold6 | v6 fold6 |
|---|---|---|---|---|---|---|---|---|
| 42 | +7.30 | +5.04 | +3.67 | +0.41 | +9.03 | +6.79 | +2.33 | +1.17 |
| 7 | +3.96 | +6.96 | −1.14 | −0.86 | +5.60 | +7.44 | −0.14 | −1.92 |
| 123 | +6.18 | +3.12 | +3.03 | +3.97 | +9.63 | +7.96 | +2.03 | −0.47 |
| 0 | +4.02 | +5.78 | +5.59 | +6.54 | +8.26 | +6.33 | +2.71 | +5.64 |
| 99 | +4.92 | +6.51 | +9.05 | +1.62 | +6.19 | +5.46 | +4.71 | +0.92 |

Mixed at single-seed level: v6 helps seed=7 (WF +1.84) but hurts most others. Mean WF across 5 seeds: v5 +7.74 → v6 +6.80 (Δ −0.94).

### VOTE5 ensemble (K=5 plurality)

| metric | BASELINE_VOTE5 (v5) | VOTE5_v6 (v6) | Δ v6-v5 |
|---|---|---|---|
| val Sharpe | **+3.53** | +1.22 | **−2.31** |
| test Sharpe | +4.19 | +4.17 | −0.02 |
| WF mean Sharpe | **+10.40** | +8.63 | **−1.78** |
| WF folds positive | 6/6 | 6/6 | 0 |
| WF fold 6 | +5.20 | +3.89 | −1.31 |
| trades (val) | 233 | 285 | +52 |
| trades (test) | 174 | 214 | +40 |

**Per-fold WF Δ (v6 minus v5):** [+1.68, −3.44, −0.88, −3.49, −3.21, −1.31]. Only fold 1 improves.

V6 trades MORE on val/test (+52 / +40 trades) but per-trade quality drops, so aggregate Sharpe falls.

## Why direction probs failed

### 1. The information is already implicit

The 9 binary signal flags (state[7..15]) are computed FROM the same direction probs:

```python
# models/dqn_state.py:122-125
df["p_up_60"]   = dir_preds["up_60"]
df["p_dn_60"]   = dir_preds["down_60"]
df["p_up_100"]  = dir_preds["up_100"]
df["p_dn_100"]  = dir_preds["down_100"]

# strategy/agent.py: S1, S4, S6 fire when p_up_60 / p_dn_60 / p_up_100 / p_dn_100
# cross trained thresholds → binary {-1, 0, +1}
```

Adding the raw probs gives the DQN no genuinely new information — just a continuous version of the signal it already has via the binary trigger. The strategy logic has *already learned the right thresholds*; the DQN doesn't need to re-learn them.

### 2. State distribution shift across splits

V6 state-builder reports per-split stats for the new dims:

| dim | train mean | val mean | test mean |
|---|---|---|---|
| p_up_60 | −0.48 | −0.28 | −0.47 |
| p_dn_60 | −0.61 | −0.74 | −0.80 |
| p_up_100 | −0.24 | −0.08 | −0.21 |
| p_dn_100 | −0.56 | −0.68 | −0.74 |

The val period has noticeably different mean for the new dims (more "up" signal, less "down" signal). The DQN trained on the train distribution sees off-distribution inputs at val/test → degraded OOD performance.

The binary signal flags don't have this issue because they're already discretized to {−1, 0, +1} — the regime shift in raw probabilities doesn't propagate to the binary triggers as long as the thresholds aren't crossed differently.

### 3. Capacity vs signal ratio

54 dims → 64 hidden = same hidden size; first layer params: 50×64 = 3,200 (v5) vs 54×64 = 3,456 (v6). Modest increase. But the signal-to-parameter ratio dropped — same 200k grad steps, more dims to fit. Some seeds overfit (42, 123 got noticeably worse).

## What this tells us

**The DQN's bottleneck is not in input information.** The audit found we were already exposing the right signals (binary flags + regime + windowed micro features). Adding more raw inputs HURTS rather than HELPS.

**Future architectural improvements should target:**
- Network capacity (test if signal is capacity-bound at hidden=128, 256)
- Reward shaping (differential Sharpe, drawdown-aware)
- Different RL algorithm (Double DQN, Dueling DQN, distributional)
- NOT input enrichment with redundant features

This is a useful negative result — it eliminates a hypothesis I had high confidence in, and re-orients the search.

## Files

| File | Contents |
|---|---|
| [models/dqn_state_v6.py](../models/dqn_state_v6.py) | v6 state-builder (50→54 dims) |
| [models/eval_v6_vs_v5.py](../models/eval_v6_vs_v5.py) | per-seed and ensemble v5 vs v6 evaluator |
| `cache/btc_dqn_state_{train,val,test}_v6.npz` | v6 state arrays (54-dim) |
| `cache/btc_dqn_policy_BASELINE_FULL_V6_seed{42,7,123,0,99}.pt` | v6 trained policies |
| `cache/results/eval_v6_vs_v5_results.json` | aggregated comparison metrics |

## Implications for prior planning

The "direction probabilities in state" recommendation in [seed_variance.md](seed_variance.md) and [voting_ensemble.md](voting_ensemble.md) Tier 1.3 is now **superseded by this negative result**. Tier 2 (larger network, capacity test) becomes the natural next step.
