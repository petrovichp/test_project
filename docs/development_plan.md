# Development Plan — zero-fee & non-zero-fee paths

This is the live forward plan as of 2026-05-10. Supersedes [next_steps.md](next_steps.md) (kept for historical reference).

## Current state — the bars to beat

### Zero-fee headline
| baseline | WF | val | test | folds + |
|---|---:|---:|---:|:--:|
| **`BASELINE_VOTE5`** ⭐ | **+10.40** | +3.53 | +4.19 | 6/6 |
| `BASELINE_VOTE5_H256` | +11.86 | +3.32 | +1.21 | 6/6 (fold-6 +0.41) |
| `BASELINE_VOTE5_H128` | +10.22 | +0.31 | **+10.59** | 5/6 (fold-6 +10.70) |

### Non-zero-fee headline (4.5 bp/side OKX taker)
| config | WF | val | test | folds + |
|---|---:|---:|---:|:--:|
| **vanilla VOTE5 + top-5 + vote≥3** ⭐ | **+3.72** | −8.11 | +0.97 | 4/6 |
| FEE4_p005 + vote≥3 | +2.57 | −4.82 | −4.07 | 5/6 |

---

# Path Z — Zero-fee algorithm improvement (ACTIVE)

Goal: lift WF mean Sharpe meaningfully above +10.40 (vanilla VOTE5) or +11.86 (H256), while preserving 6/6 fold positivity and improving fold-6 robustness.

The zero-fee assumption corresponds to maker-only execution. Per [docs/fee_improvement_proposals.md](fee_improvement_proposals.md) #1, this is unblocked by Path X (maker-only scoping) which is engineering work, not research — separate track.

## Phase Z1 — Stack the proven winners (low-risk)

Test combinations of independently-validated improvements. None of these have been tried together.

| ID | Experiment | Why | Cost |
|---|---|---|---|
| Z1.1 | **H256 + Double_Dueling** (5 seeds) | H256 wins WF (+11.86), DD wins val (+6.12). Stack: capacity + regularization | ~30 min × 5 = 2.5 h |
| Z1.2 | **K=10 vanilla VOTE10** (10 seeds, plurality) | K=5 already wins. K=10 should reduce ensemble variance further | ~6 h training (5 new seeds) |
| Z1.3 | **K=10 Double_Dueling VOTE10_DD** | If Z1.2 helps, repeat for DD | ~6 h |
| Z1.4 | **Disjoint validation of H256 + H128** | Confirm capacity gains are structural, not seed-luck | ~6 h |

**Decision gate**: Z1 winners go to Phase Z2 as the new baseline. WF +12 with 6/6 folds positive is the target.

## Phase Z2 — Better state (medium-risk)

The state vector is currently 50 dims, BTC-only, no perp basis. Multiple signals exist that we haven't used.

| ID | Experiment | Hypothesis | Cost |
|---|---|---|---|
| Z2.1 | **Cross-asset state**: add lagged ETH + SOL features (8 dims) | Cross-asset lead-lag is well-documented; predictor asset must lag by 1+ bars per data-integrity rule | ~1 day |
| Z2.2 | **Perp basis**: spot-perp price spread, perp funding rate, OI change rate (5 dims) | Funding & basis are leading indicators of squeeze conditions; we have OI in features but not state | ~1 day |
| Z2.3 | **OB depth features**: cumulative depth at ±2%, ±5%, ±10% (6 dims) | We have 800 OB amount columns — currently only summary features in state. More liquidity context could help size + signal quality | ~1 day |
| Z2.4 | **Price action context**: realized volatility ratio, recent run-up/down magnitude (4 dims) | Helps policy condition on recent price extremes regardless of regime label | ~0.5 day |

**Decision gate**: each Z2 experiment is run independently against the Z1 winner. Keep additions that lift WF by ≥+0.5 with no fold regression > 0.5. Combine survivors in Z2.5.

| Z2.5 | **Combined state v7**: stack all surviving features into one expanded state | Test additivity of state improvements | ~6 h |

## Phase Z3 — Better signals (medium-risk)

The 9 strategies in `strategy/agent.py` are fixed. Adding new strategies expands the action space.

| ID | Strategy idea | Why | Cost |
|---|---|---|---|
| Z3.1 | **S13_FundingExtreme**: trade against extreme positive/negative funding (mean-reversion on funding) | Funding-rate mean-reversion is a documented edge; we have `fund_rate` in features | ~1 day |
| Z3.2 | **S14_DepthImbalance**: trade direction of large OB depth imbalance + microprice | Direct OB-driven; complements current taker-flow signal | ~1 day |
| Z3.3 | **S15_VolBreakout**: enter on volatility expansion above N-bar median ATR | Currently no pure-vol-expansion strategy; ATR-scaled exits already in place | ~0.5 day |
| Z3.4 | **S16_BasisDislocation**: spot-perp basis trade | If Z2.2 included basis in state, this is the natural action counterpart | ~1 day |

**Decision gate**: each new strategy goes through the existing [data_splits](data_splits.md) eval. Keep if win-rate > 50% on val AND mean PnL > strategy median. Then retrain VOTE5 with expanded action space (10 → 13 actions).

## Phase Z4 — Architecture & training (higher-risk)

| ID | Experiment | Hypothesis | Cost |
|---|---|---|---|
| Z4.1 | **Self-distillation**: train a single net to mimic VOTE5 plurality output | One net deploys faster + more interpretable than 5-seed ensemble. Loss: maybe equal to VOTE5 | ~1 day |
| Z4.2 | **Transformer-block on state** (1 layer, 4 heads) | Sequence-aware, but state is already 50-dim engineered features — may not help | ~2 days |
| Z4.3 | **Curriculum learning by regime difficulty** | Train on calm regimes first, then gradually include trending/chop bars. Could improve fold-6 robustness | ~1 day |
| Z4.4 | **Distributional RL (C51 / QR-DQN)** | Predict Q distribution, choose by quantile / CVaR. Better for tail-aware decisions | ~2 days |

**Decision gate**: each Z4 experiment must beat the current Z2/Z3 winner on WF AND on fold-6. Otherwise dropped.

## Phase Z5 — Validation & freeze (mandatory before any deployment)

| ID | Step | Why |
|---|---|---|
| Z5.1 | **Out-of-distribution stress test**: run final policy on Apr-May 2026 data not yet used | Locked test only used once; this is the second OOD check |
| Z5.2 | **Seed variance ±2σ check** with 10 seeds | Confirm ensemble Sharpe range is wholly above +10.40, not borderline |
| Z5.3 | **Fee-curve at non-zero fees** for the new winner | Even though path is zero-fee-targeted, the fee curve tells us robustness margin |
| Z5.4 | **Freeze new baseline** in `docs/baselines.md` + tag in registry | Reproducibility requirement |

---

# Path F — Non-zero-fee improvement (PARKED)

**Status**: deferred per user direction 2026-05-09. Resume after Path Z reaches a clear winner OR if maker-only execution (Path X) is found infeasible.

Full prioritized plan in [docs/fee_improvement_proposals.md](fee_improvement_proposals.md). Summary order when work resumes:

| # | Phase | Experiments | Cost |
|---|---|---|---|
| F1 | **Cheap post-hoc improvements** | Vote-strength sizing, Q-margin threshold, tighter TP for trend strategies (audit follow-up #4 still unrun), funding-rate offset in reward | ~4 h combined |
| F2 | **Maker-fill-rate scoping** (engineering, not research) | Simulate post-only fill-rate using existing OB depth data. Path X gate. | ~2 h |
| F3 | **Architectural fee-aware retrains** | Fee as state feature, multi-fee training, distributional RL with CVaR action selection | ~3 days combined |
| F4 | **Asymmetric exits by regime** | regime-conditioned TP/SL scaling | ~2 h |

**Hard ceiling**: per the fee analysis, even the best filter at 4.5 bp gives WF +3.72 vs +10.40 zero-fee. Everything F1–F4 is bounded above by that ceiling unless F2 (maker-only) breaks it.

---

# Decision logic between paths

```
Phase Z1 done?
├─ winner > +12 WF, 6/6 folds  →  proceed to Z2
└─ no winner found             →  stop Z1, jump to Z3 (signals) or pause path
                                   while reviewing assumptions

Phase Z5 done?
├─ frozen baseline > +12 WF    →  ship as zero-fee deployable
└─ inconclusive                →  re-baseline expectations, decide F vs Z

Path X (maker scoping) done?
├─ maker-only feasible (>70% fill rate)  →  zero-fee path is real;
│                                            Path F becomes redundant
└─ maker-only NOT feasible                →  Path F becomes critical;
                                             switch active path
```

---

# What "shipping" looks like for each path

### Zero-fee deployment
- Frozen ensemble policy (5 or 10 seeds plurality) registered + documented
- Maker-only execution layer with verified fill-rate
- Live paper-trading >2 weeks with WF-Sharpe-aligned realized Sharpe
- Risk caps: max position size, max DD per fold, daily P&L stop

### Non-zero-fee deployment (fallback)
- Frozen ensemble + filter (or fee-aware policy) at 4.5 bp expected execution
- Lower expected Sharpe (+3 to +4 vs +10), proportionally smaller capital
- Live paper-trading at expected fee level

---

# Maintenance rules during execution

Per [`.claude/rules/experiments.md`](../.claude/rules/experiments.md) and [`.claude/rules/model-registry.md`](../.claude/rules/model-registry.md):

1. Every experiment in this plan generates a `docs/{experiment_name}.md` doc, a registry entry per trained model, and a `RESULTS.md` status-block update — all in one commit.
2. Update **this plan** (`docs/development_plan.md`) when an experiment completes: mark done, record headline metric, link to its doc.
3. If a phase's decision gate fails, document the failure and the path divergence here before moving on.
