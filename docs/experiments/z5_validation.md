# Z5 — Validation & freeze

Executed 2026-05-12. Path Z5 from [development_plan.md](../reference/development_plan.md).

This consolidates the four Z5 sub-experiments that validated `VOTE5_v8_H256_DD`
and `DISTILL_v8_seed42` as production-ready and informed the freeze decision.

## Z5.1 — Out-of-distribution stress test

Three OOD perturbations applied to val + test for both candidate policies.

| policy | split | stress | Sharpe | Δ vs base | equity | trades |
|---|---|---|---:|---:|---:|---:|
| VOTE5_v8_H256_DD | val | baseline | +6.67 | — | 1.34 | 300 |
| VOTE5_v8_H256_DD | val | inverted | +0.66 | **−6.02** | 1.02 | 299 |
| VOTE5_v8_H256_DD | val | regime_shuffle | +6.67 | +0.00 | 1.34 | 300 |
| VOTE5_v8_H256_DD | val | feature_noise σ=0.1 | +2.49 | **−4.18** | 1.10 | 285 |
| VOTE5_v8_H256_DD | test | baseline | +4.44 | — | 1.17 | 199 |
| VOTE5_v8_H256_DD | test | inverted | +0.65 | **−3.80** | 1.02 | 244 |
| VOTE5_v8_H256_DD | test | regime_shuffle | +4.44 | +0.00 | 1.17 | 199 |
| VOTE5_v8_H256_DD | test | feature_noise σ=0.1 | +4.57 | +0.13 | 1.17 | 222 |
| DISTILL_v8_seed42 | val | baseline | +10.41 | — | 1.64 | 341 |
| DISTILL_v8_seed42 | val | inverted | +4.79 | **−5.62** | 1.24 | 336 |
| DISTILL_v8_seed42 | val | regime_shuffle | +10.41 | +0.00 | 1.64 | 341 |
| DISTILL_v8_seed42 | val | feature_noise σ=0.1 | +10.59 | **+0.17** | 1.64 | 360 |
| DISTILL_v8_seed42 | test | baseline | +9.35 | — | 1.38 | 291 |
| DISTILL_v8_seed42 | test | inverted | −0.24 | **−9.59** | 0.98 | 272 |
| DISTILL_v8_seed42 | test | regime_shuffle | +9.35 | +0.00 | 1.38 | 291 |
| DISTILL_v8_seed42 | test | feature_noise σ=0.1 | +7.54 | −1.81 | 1.30 | 287 |

### Findings

1. **Long-bias asymmetry confirmed** — both policies lose 6-10 Sharpe on inverted prices. The policy is materially *not symmetric*; it bets on long edges 60-65% of the time and fails when those structural patterns reverse. Audit finding #5 from May 11 is validated. Mitigation: explicit symmetric augmentation in training (future work).
2. **Regime shuffle has zero impact** — regime_id is computed and cached but NOT in the state vector. The policy is feature-driven, not regime-aware. Reaffirms why Z4.3 curriculum learning (which gates by regime) failed.
3. **DISTILL is more feature-noise-robust on val** (+0.17 Δ vs teacher −4.18 Δ). Teacher uses microstructure features more heavily; DISTILL is content-smoothed by the plurality labels it was trained on.

## Z5.2 — 10-seed variance

Single-seed walk-forward Sharpe across orig pool {42,7,123,0,99} + disjoint {1,13,25,50,77}.

| family | WF mean | WF stdev | WF min | WF max | val mean | val stdev | test mean | test stdev |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **VOTE5_v8_H256_DD** | +8.33 | **1.67** | +5.97 | +10.41 | +6.31 | 1.89 | +4.39 | 2.32 |
| **DISTILL_v8** | +8.18 | **0.79** | +7.30 | +9.99 | +8.42 | 1.17 | +7.27 | 2.08 |

| ensemble | WF | val | test | folds+ |
|---|---:|---:|---:|---:|
| VOTE5_v8 orig K=5 | **+12.07** | +6.67 | +4.44 | 6/6 |
| VOTE5_v8 disjoint K=5 | +8.91 | −2.12 | +5.62 | 6/6 |
| VOTE5_v8 ALL K=10 | +9.34 | −0.41 | +4.11 | **5/6** |
| DISTILL_v8 orig K=5 | +7.13 | +3.45 | +3.98 | 6/6 |
| DISTILL_v8 disjoint K=5 | +7.80 | +6.10 | +5.68 | 6/6 |
| DISTILL_v8 ALL K=10 | +7.20 | **+8.70** | +3.01 | 6/6 |

### Findings

1. **DISTILL_v8 has half the WF variance of the teacher** (stdev 0.79 vs 1.67). Distillation produces more reliable single-net deployment artifacts. This is a structural property — the supervised teacher labels are deterministic given state, while RL is intrinsically noisy.
2. **Disjoint K=5 plurality voting underperforms orig K=5 for the teacher** (val collapses to −2.12). The seed-luck in orig val (+6.67) is partially exposed — A3 finding reaffirmed.
3. **K=10 voting fails for teacher** (5/6 folds, fold-6 turns negative). More voters → more ties → more NO_TRADE → Sharpe ∝ √N collapses. Same pattern as Z1.2.
4. **K=10 voting for DISTILL has highest val ever** (+8.70) but lowest test (+3.01) — overfits val.

## Z5.3 — Fee-curve robustness check

OKX taker is 4.5bp/side; maker is 0bp (free) but fill rate uncertain.

| policy | 0bp | 1bp | 2bp | 4bp | 4.5bp | 6bp | 8bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| DISTILL_v8_seed42 single | +9.99 | +6.77 | +4.60 | −0.83 | **−0.55** | −4.62 | −9.37 |
| DISTILL_v8 K=5 vote | +7.13 | +5.16 | +4.15 | −0.28 | **−2.94** | −5.49 | −8.26 |
| VOTE5_v8 K=5 (teacher) | +12.07 | +10.55 | +9.59 | +5.42 | **+4.58** | +1.70 | −3.39 |

### Findings

1. **Teacher VOTE5_v8 is dramatically more fee-robust** (breakeven ~6bp) than DISTILL_v8_seed42 (breakeven ~3bp). At realistic 4.5bp taker, teacher WF +4.58 vs DISTILL −0.55 — teacher wins by +5.13.
2. **DISTILL's edge requires maker-only execution** (fee=0). If Path X confirms maker fill rates suffice, DISTILL is the production policy.
3. **DISTILL ensemble (K=5 vote) underperforms single-net** even at zero fee, and the gap widens with fees. Single distill stays competitive longer.

## Z5.4 — Freeze decision

Based on Z5.1-Z5.3 findings:

### Frozen production baselines

| use case | policy | inference cost | metric profile |
|---|---|---|---|
| fee=0 (maker-only) BTC | `DISTILL_v8_seed42` | 1× forward | WF +9.99, test **+9.35**, family-mean test +7.27 |
| fee=4.5bp (taker) BTC | `VOTE5_v8_H256_DD` (K=5) | 5× forward | WF **+12.07**, fee=4.5bp WF +4.58 |
| cross-asset diversification | `VOTE5_v8_H256_DD_{btc,eth,sol}` | 5× per ticker | partial transfer, ETH/SOL ~60-70% of BTC |

### What the freeze does NOT preclude

The freeze provides a stable reference baseline. Future experiments compare against it. The freeze does NOT mean "no more research" — open research bets are documented in [development_plan.md §Forward plan](../reference/development_plan.md).

### Negative results retained for context

- `VOTE5_v8_CURR_H256_DD` (Z4.3 curriculum): NEGATIVE — retained for audit
- `VOTE10_DD` (K=10 vote): NEGATIVE — retained for audit
- `BASELINE_VOTE5_H128`: SEED-LUCK — dropped
- `VOTE5_v9_H256_DD`: NOT-ADDITIVE — retained as fold-6 alternative
- `QRDQN_v8` (Z4.4): NEGATIVE-LEAN — retained for audit
- `XFMR_v8` (Z4.2): PENDING — retained when training completes

## Code touchpoints

- `models/z5_ood_stress.py` — OOD stress test (price inversion, regime shuffle, feature noise)
- `models/z5_variance_10seed.py` — single-seed variance + K=10 vote eval
- `models/z5_fee_curve_distill.py` — fee curve for DISTILL vs teacher

## Outputs

- `cache/results/z5_ood_stress.json`
- `cache/results/z5_variance_10seed.json`
- `cache/results/z5_fee_curve_distill.json`
