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

## Findings (Transformer — pending)

To be filled after Z4.2 training completes and eval runs.

## Verdict

_Pending Transformer results. Update after Z5.4 freeze decision._

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
