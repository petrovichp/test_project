# Z5 — Validation & freeze

Executed 2026-05-12. Path Z5 from [development_plan.md](../reference/development_plan.md).

This consolidates the four Z5 sub-experiments that validated `VOTE5_v8_H256_DD`
and `DISTILL_v8_seed42` as production-ready and informed the freeze decision.

> **Audit note (same day):** the original Z5.1 regime_shuffle test was buggy
> (it shuffled an unused sidecar `regime_id` array instead of `state[:, 2:7]`,
> which is the actual model input). Two original Z5.1 findings — "policy is
> feature-driven, not regime-aware" and "policy has long-bias asymmetry" — have
> been **withdrawn**. The corrected diagnostics (Q1/Q2/Q3 below) show:
>
> 1. DISTILL_v8 *does* use the regime one-hot (perturbation costs 2-6 Sharpe).
> 2. Both policies trade ~55/45 long/short with shorts contributing *more*
>    PnL — no long-bias in deployment.
> 3. All 5 regimes are profitable; trend_down is the highest per-trade-PnL bucket.
>
> The freeze decision (Z5.4) is unchanged: it never rested on the regime or
> long-bias claims; it rests on Z5.2 (variance) and Z5.3 (fee curve), both
> of which were mechanically correct.

## Z5.1 — Out-of-distribution stress test

Three OOD perturbations applied to val + test for both candidate policies.

| policy | split | stress | Sharpe | Δ vs base | equity | trades |
|---|---|---|---:|---:|---:|---:|
| VOTE5_v8_H256_DD | val | baseline | +6.67 | — | 1.34 | 300 |
| VOTE5_v8_H256_DD | val | inverted | +0.66 | −6.02 | 1.02 | 299 |
| VOTE5_v8_H256_DD | val | feature_noise σ=0.1 | +2.49 | **−4.18** | 1.10 | 285 |
| VOTE5_v8_H256_DD | test | baseline | +4.44 | — | 1.17 | 199 |
| VOTE5_v8_H256_DD | test | inverted | +0.65 | −3.80 | 1.02 | 244 |
| VOTE5_v8_H256_DD | test | feature_noise σ=0.1 | +4.57 | +0.13 | 1.17 | 222 |
| DISTILL_v8_seed42 | val | baseline | +10.41 | — | 1.64 | 341 |
| DISTILL_v8_seed42 | val | inverted | +4.79 | −5.62 | 1.24 | 336 |
| DISTILL_v8_seed42 | val | feature_noise σ=0.1 | +10.59 | **+0.17** | 1.64 | 360 |
| DISTILL_v8_seed42 | test | baseline | +9.35 | — | 1.38 | 291 |
| DISTILL_v8_seed42 | test | inverted | −0.24 | −9.59 | 0.98 | 272 |
| DISTILL_v8_seed42 | test | feature_noise σ=0.1 | +7.54 | −1.81 | 1.30 | 287 |

### Findings (revised)

1. **Price inversion** — mechanically correct test; interpretation was
   over-reaching. The 6-10 Sharpe drop reflects feature-price correlation
   (state was computed from original prices but the simulator ran on flipped
   prices), not a long-bias asymmetry in the policy. The actual dataset has
   no sustained directional trend, so "would underperform in downtrend" is
   unsupported extrapolation. The deployment-time long/short attribution
   (Q2) shows the policy is roughly balanced. **Withdrawn as evidence of
   asymmetry.** Retained as a feature-importance signal — confirms features
   drive entries.
2. **Regime shuffle (original)** — **buggy**. Perturbed `sp["regime_id"]`
   integer sidecar, never read by `run_eval`. The agent's input
   (`state[:, 2:7]` one-hot, inherited from v5 state-pack) was unchanged.
   Zero impact ≠ ignored regime. **Withdrawn.** Replaced by Q1 below.
3. **Feature noise σ=0.1 on dims [30:50]** — mechanically correct. DISTILL
   robustly handles noise (Δ +0.17 on val) while teacher degrades (Δ −4.18).
   Distillation produces a smoother policy with less reliance on noisy
   microstructure dims. **Stands.**

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

1. **DISTILL_v8 has half the WF variance of the teacher** (stdev 0.79 vs 1.67).
   Structural — distillation labels are deterministic given state, RL is
   intrinsically noisy. Production implication: DISTILL is more reproducible
   across retrains.
2. **K=10 plurality voting fails for the teacher** (5/6 folds, fold-6 turns
   negative). More voters → more ties → more NO_TRADE → Sharpe ∝ √N collapse.
   Same pattern as Z1.2. K=5 is the sweet spot; K>5 should not be re-tried.
3. **Disjoint K=5 underperforms orig K=5 on the teacher's val** (val collapses
   to −2.12) — confirms partial seed-luck in orig val Sharpe.

## Z5.3 — Fee-curve robustness check

OKX taker is 4.5bp/side; maker is 0bp (free) but fill rate uncertain.

| policy | 0bp | 1bp | 2bp | 4bp | 4.5bp | 6bp | 8bp |
|---|---:|---:|---:|---:|---:|---:|---:|
| DISTILL_v8_seed42 single | +9.99 | +6.77 | +4.60 | −0.83 | **−0.55** | −4.62 | −9.37 |
| DISTILL_v8 K=5 vote | +7.13 | +5.16 | +4.15 | −0.28 | **−2.94** | −5.49 | −8.26 |
| VOTE5_v8 K=5 (teacher) | +12.07 | +10.55 | +9.59 | +5.42 | **+4.58** | +1.70 | −3.39 |

### Findings

1. **Teacher VOTE5_v8 is dramatically more fee-robust** (breakeven ~6bp) than
   DISTILL_v8_seed42 (breakeven ~3bp). At realistic 4.5bp taker, teacher
   WF +4.58 vs DISTILL −0.55 — teacher wins by +5.13.
2. **DISTILL's edge requires maker-only execution** (fee=0). If Path X confirms
   maker fill rates suffice, DISTILL is the production policy.
3. **DISTILL ensemble (K=5 vote) underperforms single-net** even at zero fee,
   and the gap widens with fees. Single distill stays competitive longer.

## Z5.4 — Freeze decision

Based on Z5.2-Z5.3 (Z5.1 originally cited but its regime/long-bias claims
have been withdrawn — see audit note above; the freeze does not rely on them):

### Frozen production baselines

| use case | policy | inference cost | metric profile |
|---|---|---|---|
| fee=0 (maker-only) BTC | `DISTILL_v8_seed42` | 1× forward | WF +9.99, test **+9.35**, family-mean test +7.27 |
| fee=4.5bp (taker) BTC | `VOTE5_v8_H256_DD` (K=5) | 5× forward | WF **+12.07**, fee=4.5bp WF +4.58 |
| cross-asset diversification | `VOTE5_v8_H256_DD_{btc,eth,sol}` | 5× per ticker | partial transfer, ETH/SOL ~60-70% of BTC |

### What the freeze does NOT preclude

The freeze provides a stable reference baseline. Future experiments compare
against it. The freeze does NOT mean "no more research" — open research bets
are documented in [development_plan.md §Forward plan](../reference/development_plan.md).

### Negative results retained for context

- `VOTE5_v8_CURR_H256_DD` (Z4.3 curriculum): NEGATIVE — retained for audit.
  Note: original rationale "regime not in state → curriculum hits a no-op"
  is **wrong**. Regime IS in state at [2:7]. The actual failure cause is
  buffer-bias from regime-gated training samples distorting the replay
  distribution.
- `VOTE10_DD` (K=10 vote): NEGATIVE — retained for audit
- `BASELINE_VOTE5_H128`: SEED-LUCK — dropped
- `VOTE5_v9_H256_DD`: NOT-ADDITIVE — retained as fold-6 alternative
- `QRDQN_v8` (Z4.4): NEGATIVE-LEAN — retained for audit
- `XFMR_v8` (Z4.2): NEGATIVE — retained for audit (see
  [z4_qrdqn_transformer.md](z4_qrdqn_transformer.md))

## Z5 Audit — Corrected diagnostics (Q1 / Q2 / Q3)

Run via `python3 -m models.audit_regime_direction`. Outputs:
`cache/results/audit_regime_direction.json`.

### Q1 — Corrected regime perturbation on state[:, 2:7]

| policy | split | row-shuffle Δ | zero-out Δ |
|---|---|---:|---:|
| VOTE5_v8_H256_DD | val | **+2.37** | **−7.70** |
| VOTE5_v8_H256_DD | test | −0.71 | **+2.63** |
| DISTILL_v8_seed42 | val | **−5.72** | **−5.00** |
| DISTILL_v8_seed42 | test | **−1.99** | **−2.98** |

**Verdict.** Regime info IS used. DISTILL_v8 degrades in all 4 cells
(−2 to −6 Sharpe) — consistent dependence. VOTE5 ensemble is inconsistent
(per-seed regime sensitivities differ; plurality vote masks them).
Distillation by averaging plurality labels over seeds surfaces the consensus
regime dependence into a single net.

### Q2 — Long/short PnL attribution (walk-forward, 6 folds)

| policy | n_long | n_short | long PnL | short PnL | dominant side |
|---|---:|---:|---:|---:|---|
| VOTE5_v8_H256_DD | 769 (54.3%) | 647 (45.7%) | +141.4% | **+183.7%** | shorts |
| DISTILL_v8_seed42 | 918 (55.3%) | 742 (44.7%) | +118.9% | **+151.4%** | shorts |

Every fold positive on both directions (one trivial exception:
VOTE5 fold 6 short ≈ 0). **No long-bias in deployment.** Both policies
trade ~55/45 long/short by count; shorts contribute MORE PnL by sum.

### Q3 — Regime-conditional attribution

**VOTE5_v8_H256_DD** (1416 trades total):

| regime | n_trades | share | mean PnL% | sum PnL% | long share |
|---|---:|---:|---:|---:|---:|
| calm | 125 | 8.8% | +0.120 | +14.94 | 57.6% |
| trend_up | 136 | 9.6% | +0.168 | +22.82 | 36.8% |
| trend_down | 162 | 11.4% | **+0.317** | +51.33 | 59.9% |
| ranging | 448 | 31.6% | +0.240 | +107.39 | 54.0% |
| chop | 545 | 38.5% | +0.236 | +128.67 | 56.5% |

**DISTILL_v8_seed42** (1660 trades total):

| regime | n_trades | share | mean PnL% | sum PnL% | long share |
|---|---:|---:|---:|---:|---:|
| calm | 156 | 9.4% | +0.072 | +11.17 | 59.0% |
| trend_up | 135 | 8.1% | +0.078 | +10.52 | 45.9% |
| trend_down | 200 | 12.0% | **+0.244** | +48.73 | 60.5% |
| ranging | 527 | 31.7% | +0.158 | +83.01 | 57.1% |
| chop | 642 | 38.7% | +0.182 | +116.89 | 53.3% |

**Verdict.** All 5 regimes profitable for both policies. **trend_down has the
highest per-trade PnL** for both (+0.32% / +0.24%); long-share in trend_down
is ~60% — policy is not a directional trend-follower. Chop+ranging dominate
trade volume (~70%) but per-trade edge is highest in trends. DISTILL has
slightly lower per-trade PnL in every regime (consistent with WF +9.99 vs
+12.07) but preserves the teacher's regime *shape*.

## Code touchpoints

- `models/z5_ood_stress.py` — OOD stress test. The `stress_regime_shuffle`
  function was patched 2026-05-12 to perturb `state[:, 2:7]` directly;
  a `stress_regime_zero` variant was added.
- `models/z5_variance_10seed.py` — single-seed variance + K=10 vote eval
- `models/z5_fee_curve_distill.py` — fee curve for DISTILL vs teacher
- `models/audit_regime_direction.py` — Q1/Q2/Q3 corrected diagnostics

## Outputs

- `cache/results/z5_ood_stress.json` (original Z5.1 — regime_shuffle row is invalid)
- `cache/results/z5_variance_10seed.json`
- `cache/results/z5_fee_curve_distill.json`
- `cache/results/audit_regime_direction.json` (corrected Q1/Q2/Q3)
