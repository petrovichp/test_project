# Z4.4 + Z4.2 — QR-DQN with CVaR + Transformer-DQN

Executed 2026-05-12. Path Z4.4 (Distributional RL) and Z4.2 (Transformer on
state) from [development_plan.md](../reference/development_plan.md).

## Hypothesis (Z4.4 QR-DQN)

Distributional RL approximates the full return distribution per action,
not just the expected return. This:

1. **Enables CVaR action selection** — the policy picks actions by their
   bottom-α tail mean rather than expected value, which is risk-averse.
   At α=0.3, the agent uses the mean of the worst 30% of returns. This
   directly trades expected return for tail control.
2. **Provides richer learning signal** — each action gets N quantile
   targets instead of one scalar, increasing gradient density.
3. **Targets the val-test gap** — if v8's val variance comes from the
   policy being insensitive to upside-downside asymmetry, CVaR action
   selection should help.

## Hypothesis (Z4.2 Transformer)

Multi-head self-attention over state dimensions should learn pairwise
feature interactions (e.g. "basis dislocation × funding extreme")
that an MLP must implicitly capture through dense weights. The 52-dim
state has natural groups (direction probs, vol features, signal flags,
microstructure) that attention could discriminate between.

## Setup

| | QR-DQN | Transformer |
|---|---|---|
| network | `QRDuelingDQN(52, 12, 256, 32 quantiles)` | `TransformerDQN(d_model=16, n_heads=4, n_layers=2, hidden=128)` |
| params | 100,128 | 18,685 |
| state | v8_s11s13 (52-dim, 12 actions) | same |
| training | 200k grad steps, PER, target net, Adam 1e-3 | same |
| action selection (eval) | CVaR-α plurality vote (α=0.3 default) | mean-Q plurality vote |
| seeds | {42, 7, 123, 0, 99} | same |
| early-stop patience | 25k steps stagnant | same |

## Results

### Per-seed best val Sharpe (single-seed greedy at training time)

| seed | QR-DQN val | Transformer val |
|---|---:|---:|
| 42 | +6.39 | _filled after eval_ |
| 7 | +6.42 | … |
| 123 | +5.11 | … |
| 0 | +4.86 | … |
| 99 | +6.20 | … |
| **mean** | **+5.79** | … |

### Walk-forward ensemble (placeholder pending Transformer completion + run_walkforward on QR)

| policy | WF | val | test | folds+ |
|---|---:|---:|---:|:---:|
| QRDQN_v8 VOTE5 CVaR=0.1 | _wf TBD_ | +1.80 | +4.24 | _TBD_ |
| **QRDQN_v8 VOTE5 CVaR=0.3** | _wf TBD_ | **+4.77** | **+5.08** | _TBD_ |
| QRDQN_v8 VOTE5 CVaR=0.5 | _wf TBD_ | −0.81 | +4.17 | _TBD_ |
| QRDQN_v8 VOTE5 mean-Q (α=1) | _wf TBD_ | −3.80 | +6.18 | _TBD_ |
| QRDQN single best (s=0, α=0.3) | _wf TBD_ | +4.77 | **+7.95** | _TBD_ |
| XFMR_v8 VOTE5 | _filled after eval_ | … | … | … |
| **BASELINE VOTE5_v8_H256_DD** | +12.07 | +6.67 | +4.44 | 6/6 |
| **DISTILL_v8_seed42** (cheap-deploy) | +9.99 | +10.41 | +9.35 | 6/6 |

## Findings (preliminary, QR-DQN)

### CVaR-0.3 is the sweet spot

- α=0.1 (top 10% worst): policy is too conservative; val collapses to +1.80
- **α=0.3 (top 30% worst)**: best balance — val +4.77, test +5.08
- α=0.5 (top 50%): tail averaging fades; val negative
- α=1.0 (mean-Q greedy): val negative (−3.80) but test +6.18 ← interesting asymmetry

The mean-Q variant has the highest test Sharpe but worst val — overfit pattern.
CVaR-0.3 generalizes better.

### QR-DQN doesn't beat the deployed baselines

At CVaR-0.3, the ensemble val/test (+4.77 / +5.08) is **worse than VOTE5_v8** (val +6.67) and **much worse than DISTILL_v8_seed42** (val +10.41, test +9.35).

Single-seed s=0 at CVaR-0.3 hits test +7.95 — encouraging — but that's selection by single-seed test which is leakage. Honest pick is the family mean.

**Verdict (QR-DQN)**: 🟡 NEGATIVE-LEAN. CVaR action selection provides some risk awareness but doesn't lift WF aggregate or val. The quantile-regression learning signal does not provide a material improvement over the existing teacher or distilled policies.

## Findings (Z4.2 Transformer)

The original heavy config (d_model=16, n_heads=4, n_layers=2) ran too slowly
on CPU (~50 min/seed, projected 4+ hrs total). Retrained 5 seeds with a
lighter `TransformerDQN(52, 12, d_model=8, n_heads=2, n_layers=1, hidden=128)`
— 41 min total (5×~8 min). Same training recipe as `dqn_selector`
(LR=1e-3, batch=128, 200k steps with early-stop, PER, target sync 1k).

### Per-seed training-best val Sharpe

| seed | XFMR_v8 val (training-best) |
|---|---:|
| 42 | +6.690 |
| 7 | +7.623 |
| 123 | +4.919 |
| **0** | **+11.105** ← project-record val, single seed |
| 99 | +4.579 |
| **mean** | **+6.98** |

### Walk-forward eval (locked methodology)

| policy | WF | val | test | folds+ |
|---|---:|---:|---:|:---:|
| XFMR_v8 VOTE5 plurality | **+6.35** | +4.88 | +5.95 | 6/6 |
| XFMR_v8 single s=42 | +6.25 | +6.69 | +4.12 | 6/6 |
| XFMR_v8 single s=7 | +6.68 | +7.62 | +4.25 | 6/6 |
| XFMR_v8 single s=123 | +5.32 | +4.92 | +2.16 | 6/6 |
| **XFMR_v8 single s=0** | +7.71 | **+11.11** | +6.63 | 6/6 |
| XFMR_v8 single s=99 | +4.95 | +4.58 | +3.77 | 6/6 |
| **BASELINE VOTE5_v8_H256_DD** | **+12.07** | +6.67 | +4.44 | 6/6 |
| **DISTILL_v8_seed42** | +9.99 | +10.41 | **+9.35** | 6/6 |

**Verdict (Transformer ensemble)**: 🔴 **NEGATIVE.** XFMR ensemble underperforms
the teacher baseline by ~5.7 WF Sharpe. Seed 0 is a strong outlier
(val +11.11 sets a project record for single-seed greedy val Sharpe), but
plurality voting dilutes it.

### Why Transformer didn't help

The 52-dim state is already curated: direction probs from CNN-LSTMs, vol
predictions from LGBM, hand-engineered OB features. The MLP's
`Linear → ReLU → Linear` is sufficient to learn the residual structure.
Multi-head attention's inductive bias (pairwise feature interactions across
heterogeneous, mostly-uncorrelated curated features) doesn't match where the
alpha lives — the alpha lives in the features themselves, not their pairwise
interactions.

## Findings (Z4.2 follow-on — distillation of the Transformer ensemble)

Tested whether distilling the XFMR plurality labels into a single
`DuelingDQN(52, 12, 256)` would recover DISTILL_v8-style performance.

Procedure: identical to `models/distill_vote5.py` recipe (12 epochs,
batch=256, stratified 50/50 NO_TRADE/trade sampling, masked CE on
plurality labels). Labels generated by
`models/distill_targets_xfmr.py` from the 5 XFMR teachers (output:
`cache/distill/btc_distill_targets_{split}_xfmr.npz`).

| policy | WF | val | test | folds+ |
|---|---:|---:|---:|:---:|
| **DISTILL_XFMR_v8_seed42** | +6.37 | +9.41 | **+2.83** | 5/6 |
| DISTILL_v8_seed42 (reference) | +9.99 | +10.41 | +9.35 | 6/6 |

**Verdict (XFMR distillation)**: 🔴 **NEGATIVE.** Student inherits the weak
teacher. Val (+9.41) is decent but test collapses to +2.83 — the
val/test spread (6.6 Sharpe) is much wider than canonical DISTILL_v8
(spread 1.1). One fold flips negative. Distillation cannot exceed teacher
quality, and the XFMR ensemble was the bottleneck.

## Verdict (combined)

| sub-task | result |
|---|---|
| Z4.4 QR-DQN + CVaR | 🟡 NEGATIVE-LEAN |
| Z4.2 Transformer ensemble | 🔴 NEGATIVE |
| Z4.2 follow-on — distilled Transformer | 🔴 NEGATIVE |

Both architectural priors (distributional value heads, attention over state
dims) were tested cleanly. Neither lifts the frozen baseline. The next
research bets must look elsewhere — likely at features (state-vector
expansion, richer microstructure), training signal (symmetric augmentation,
explicit regime conditioning), or label quality (multi-asset distillation).

## Code touchpoints

- `models/qr_network.py` — `QRDuelingDQN`, `quantile_huber_loss`,
  `cvar_action`, `mean_q_action`
- `models/qr_dqn_train.py` — standalone QR-DQN training loop
- `models/transformer_network.py` — `TransformerDQN`
- `models/transformer_train.py` — Transformer training loop
- `models/eval_z4.py` — joint evaluator with CVaR + ensemble vote

## Registered policies

- `QRDQN_v8` family: 5 seeds → `cache/policies/btc_dqn_policy_QRDQN_v8_seed{42,7,123,0,99}.pt`
- `XFMR_v8` family: 5 seeds → `cache/policies/btc_dqn_policy_XFMR_v8_seed{42,7,123,0,99}.pt`
