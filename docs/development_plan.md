# Development Plan — zero-fee & non-zero-fee paths

This is the live forward plan as of 2026-05-10. Supersedes [next_steps.md](next_steps.md) (kept for historical reference).

## Current state — the bars to beat

### Zero-fee headline (post-Z1)
| baseline | WF | val | test | fold-6 | folds + |
|---|---:|---:|---:|---:|:--:|
| **`VOTE5_H256_DD`** ⭐ (Z1.1 winner) | **+11.05** | +3.21 | **+9.01** | **+8.23** | 6/6 |
| `BASELINE_VOTE5` (h=64 vanilla) | +10.40 | +3.53 | +4.19 | +5.20 | 6/6 |

### Non-zero-fee headline (4.5 bp/side OKX taker)
| config | WF | val | test | folds + |
|---|---:|---:|---:|:--:|
| **vanilla VOTE5 + top-5 + vote≥3** ⭐ | **+3.72** | −8.11 | +0.97 | 4/6 |
| FEE4_p005 + vote≥3 | +2.57 | −4.82 | −4.07 | 5/6 |

> Note: the fee=4.5 bp leaderboard above has **not been re-evaluated against `VOTE5_H256_DD`** yet. That's part of Z5.3 (fee-curve check) — a future step.

---

## ⏭ Current steps — what to do next, and why

### The starting point

Phase Z1 produced `VOTE5_H256_DD` at **WF +11.05 / test +9.01 / fold-6 +8.23 / 6/6 folds**. The H256 capacity + DD regularization stack worked exactly as hypothesized — capacity gave the WF lift, regularization recovered fold-6.

### The question we're answering next

**Where is the next +1 to +2 Sharpe coming from?** There are exactly three places it could be:

1. **More information** (Phase Z2) — the policy is making decisions on a state vector that's missing useful context.
2. **More actions** (Phase Z3) — the action space (9 strategies) doesn't include trades the policy would take if it could.
3. **Better learning** (Phase Z4) — the policy isn't extracting all the signal that's already in state + actions.

Each axis has different evidence. The order of next steps follows the strength of evidence.

### Evidence per axis — what we already know

**Z2 — More state information**: weak evidence base.
- State v6 added direction probabilities (4 dims) → NEGATIVE result, WF dropped by 1.78 ([state_v6_test.md](state_v6_test.md)). But those 4 dims were already implicit in the binary signal flags — that's a *weak* negative, not a *general* one.
- We have features that exist in `cache/*_meta.parquet` but are NOT in the 50-dim state: cross-asset (ETH/SOL), perp-spot basis as raw, funding-rate dynamics, OB-depth percentiles. **These have never been tested.**
- *Inference*: the bottleneck might be in *which* information is in state, not in *how much*. Worth testing.

**Z3 — More actions**: stronger evidence base.
- [strategy/agent.py](../strategy/agent.py) defines **13 strategies**, but only **9** are wired into the DQN action space. Strategies 5, 9, 11, 13 exist as code but the DQN has never seen them.
- Strategy 11 specifically uses spot-perp basis (`diff_price` z-score) — a signal type completely absent from the current 9 strategies.
- The A4 audit ([baseline_vote5_audit.md](baseline_vote5_audit.md)) showed S6 and S10 contributing *negatively* on test — the current action space is not optimally chosen, suggesting unused-code strategies could displace the weak ones.
- *Inference*: there is likely free alpha in the unused code. The cost to test is **zero training** — a standalone backtest confirms or rejects.

**Z4 — Better learning**: positive evidence base, but expensive to test.
- We *just* showed in Z1 that architecture changes work (+0.65 WF from H256+DD stack). This is direct evidence that the architecture axis has more headroom.
- But Z4's experiments (transformer, distributional RL, self-distillation) are 1-2 days each — by far the most expensive axis to explore.
- *Inference*: there's likely value here, but cost-benefit favors exhausting Z2/Z3 first.

### The strategy: cheap-first, evidence-driven, gated

The right order is **Z3 standalone validation → Z2 cheap warm-up → Z2 strong candidate → Z3 retrain → Z2 combined → Z3 novel strategy**. Reasoning:

1. **Cheapest action with the highest information value first** — Z3.1 standalone validation costs ~1 hour and tells us whether 4 unused strategies have signal. The result reshapes the entire plan: if 2+ strategies pass, Step 4 becomes high-priority; if all fail, we skip Step 4 and Step 6 entirely.

2. **Establish the retrain pipeline cheaply** — Z2.4 (price-action, 4 dims) is ~25 min training. Whether or not it lifts Sharpe, it proves the v7 retrain pipeline works, which de-risks the more expensive Z2.2.

3. **Hit the strongest theoretical add** — Z2.2 (basis + funding state) is the most promising Z2 candidate because basis & funding are documented macro signals that are completely absent from current state, AND we know strategies that *use* them (S2_Funding, the unused S11) exist or could exist. Adding the *state context* alongside *strategy actions* in parallel is multiplicative.

4. **Compose surviving wins** — Z2.5 (combined state v7) only runs if Steps 2+3 produced winners. If yes, this tests additivity. If no, skipped.

5. **Add genuine novelty last** — Z3.2 (S15_VolBreakout) is the only truly-new strategy in the plan. It runs after Step 4 confirms the action-space-expansion pipeline works.

### The 6 steps

| # | step | cost | gate | result reshapes |
|---|---|---|---|---|
| **1** ✅ | [Z3.1 standalone validation](#step-1) | DONE 2026-05-11 | — | S11+S13 kept; S5+S9 dropped. See [z3_step1_killed_strategies.md](z3_step1_killed_strategies.md). |
| **2** ❌ | [Z2.4 price-action context](#step-2) | DONE 2026-05-11 | — | **NEGATIVE.** WF +9.18 vs baseline +11.05. Drop v7_pa. See [z2_z3_results.md](z2_z3_results.md). |
| **3** 🟡 | [Z2.2 perp basis + funding state](#step-3) | DONE 2026-05-11 | — | **PARTIAL.** Val resilience confirmed (+3.46), but test/fold-6 regressed. Keep as alternative. |
| **4** ✅ | [Z3.1 wire & retrain](#step-4) | DONE 2026-05-11 | — | **WIN. `VOTE5_v8_H256_DD` PROMOTED as new primary baseline. WF +12.07, val +6.67, 6/6 folds.** |
| **5** | [Z2.5 combined state v7](#step-5) | ~30 min | Steps 3+4 winners | NEXT — combine v8 action space + v7_basis state |
| **6** | [Z3.2 S15_VolBreakout](#step-6) | ~0.5 day | Step 4 pipeline works ✓ | gated after Step 5 |

#### Step 1 — Z3.1 standalone validation
**What**: Run [backtest/run.py](../backtest/run.py) on each of `strategy_5`, `strategy_9`, `strategy_11`, `strategy_13` over the full RL period (rule-based exits, fee=0).
**Output per strategy**: win-rate, mean PnL/trade, total trades, val/test/WF Sharpe.
**Decision**: keep strategies with win-rate >50% AND mean PnL >0.15% (median across current 9). Drop the rest.
**Proof of value**: 4 strategies exist as production-ready code that has never been evaluated as part of the DQN action space. The maximum information-per-hour ratio in this entire plan. Even if all 4 fail (a real possibility), the negative result reshapes the plan by removing Steps 4 and 6.

#### Step 2 — Z2.4 price-action context (the cheap pipeline-test)

**The problem this addresses**:
The current 50-dim state vector has the *regime classifier output* (one-hot over {calm, trend_up, trend_down, ranging, chop}) and the ATR prediction. What it does NOT have is **a continuous measure of how extreme the current bar's position is relative to recent history**. The DQN sees "we're in a trend_up regime" but not "we're currently 5% below the 60-bar high" or "we just got 2× vol expansion in the last 30 bars."

A trader frames this naturally: "BTC is pulling back 5% from recent peak; vol is expanding; I'm cautious." The DQN has the regime label but lacks the *position-within-regime* signal.

**The 4 features** (all derived from the 50-bar price array — no new data dependencies):

| feature | formula | range (typical) | what it measures |
|---|---|---|---|
| `price_drawdown_60` | `1 - price_now / max(price[-60:])` | 0 to ~0.05 | how far below recent 60-bar high (0 = at high) |
| `price_runup_60` | `price_now / min(price[-60:]) - 1` | 0 to ~0.05 | how far above recent 60-bar low (0 = at low) |
| `realized_vol_60` | `std(returns_60)` × √60 | ~0.001 to ~0.01 | actual realized volatility over last 60 bars |
| `vol_ratio_30_60` | `atr_30 / median(atr_60)` | ~0.5 to ~3.0 | short-term vs medium-term vol ratio (regime transition signal) |

**Concrete example**:
```
BTC at $100,000.
  max in last 60 bars = $102,000  → price_drawdown_60 = 1.96% (pullback context)
  min in last 60 bars = $99,500   → price_runup_60    = 0.50%
  std of 60 returns   = 0.00080   → realized_vol_60   = 0.0062
  ATR_30 / median(ATR_60) = 1.85  → vol_ratio_30_60   = 1.85 (vol expanding)
```

State vector grows: 50 → 54 dims.

**Why these specific features**:
- **Cheapest possible state addition** — pure derivations from price array, no new data parsing
- **Continuous, normalized** — fits the existing state convention (all current dims are roughly normalized)
- **Orthogonal to regime classifier** — regime is discrete; these are continuous measurements within a regime
- **Used in classical trader frameworks** — pullback distance, vol expansion ratio are textbook context features

**Why this might fail**:
- Regime classifier might already encode similar info in discrete form
- ATR is already in state — `vol_ratio_30_60` might add little beyond raw ATR
- DQN might not need these because the signal-strategy interaction (e.g. S1_VolDir requiring `vol_thresh > 0.60`) already encodes vol context

**Implementation pipeline** (~25 min training, ~1 hour total):
1. Modify [models/dqn_state.py](../models/dqn_state.py) to compute these 4 features over the cached price array and append to state vector.
2. Add `--state-version v7_pa` to [models/dqn_selector.py](../models/dqn_selector.py); regenerate state cache: `cache/btc_dqn_state_{train,val,test}_v7_pa.npz`.
3. Train 5 H256+DD seeds: `python3 -m models.dqn_selector btc --tag VOTE5_v7pa_H256_DD_seed{S} --state-version v7_pa --hidden 256 --algo double_dueling --fee 0.0 --trade-penalty 0.001 --seed {S}`.
4. Evaluate as K=5 plurality vs `VOTE5_H256_DD` baseline.

**Decision criterion**: keep if WF ≥ +11.55 (current +0.5) AND no fold worse by >0.5.

**Proof of value beyond the test itself**: this is the **pipeline-test** for v7 state. It validates that:
- The state cache regeneration works correctly
- H256+DD training accepts arbitrary `state_dim` not hardcoded to 50
- Plurality voting works on networks with non-50 input dim

If this step succeeds mechanically (regardless of Sharpe outcome), Step 3 (Z2.2) can proceed with confidence. If it fails mechanically, we discover infrastructure debt cheaply.

**Risk**: low. Cost: ~1 hour. Expected lift: 0 to +1 Sharpe.

---

#### Step 3 — Z2.2 perp basis + funding state (the strong macro-context add)

**The problem this addresses**:
BTC perpetual futures trade on leverage. Two macro indicators dominate the leveraged-flow narrative — **basis** (perp-spot price spread) and **funding rate** (the 8h tax/subsidy that anchors perp to spot). Both:
1. Exist in our raw data (`spot_*_price`, `perp_*_price`, `fund_rate` in meta parquet)
2. Are NOT in the current state vector
3. Are used by traders to predict squeeze conditions, liquidation cascades, and trend exhaustion

The current DQN sees `signals[k]` for `S2_Funding` (a binary fire/no-fire based on funding magnitude). It does NOT see the *raw* funding rate value, *direction* of funding change, basis dislocation Z-score, or open-interest dynamics. This is the analog of "you can see the strategy fired" without "you can see the underlying conditions that caused the fire."

**The fee-aware retrain experiment ([fee_aware_retrain.md](fee_aware_retrain.md)) is direct evidence this matters**: adding fee to the reward (so the policy was *penalized* for it) but NOT to the state (so the policy couldn't *see* it) produced a policy that under-performed both vanilla VOTE5 + filter AND maker-only. The fix was clear: the policy needs to *see* the cost context, not just be punished for it. Same principle applies to funding/basis.

**The 5 features**:

| feature | formula | range | what it measures |
|---|---|---|---|
| `basis_z_60` | `(basis - mean(basis_60)) / std(basis_60)` | typically −3 to +3 | perp-spot dislocation in std-deviations from recent normal |
| `basis_change_1bar` | `basis[t] - basis[t-1]` (in bps) | ±5 bps usually | basis acceleration / deceleration (trend in dislocation) |
| `funding_rate_apr` | `fund_rate × (365 × 24/8)` | typically −60% to +25% APR | current annualized funding cost |
| `funding_z_120` | `(funding - mean(funding_120)) / std(funding_120)` | typically −3 to +3 | funding extreme relative to recent regime |
| `oi_change_60` | `(oi[t] - oi[t-60]) / oi[t-60]` | typically ±0.05 | open-interest growth rate (leverage building or unwinding) |

**Concrete example** (extreme leverage build-up scenario):
```
basis = +6 bps (perp expensive vs spot)
  mean basis over 60 bars = -4 bps
  std basis over 60 bars  = 2 bps
  → basis_z_60 = (6 - (-4)) / 2 = +5.0    (extreme dislocation, well above 2σ)

basis_change_1bar = +1.2 bps     (still expanding, accelerating)

fund_rate = +0.0001 / bar
  → funding_rate_apr = +0.01% × (365 × 24 / 8) × 100 = +10.95% APR
  funding_z over 120 bars = +2.1   (high vs recent regime)

oi_change_60 = +3.2%             (3.2% increase in open interest in last 60 bars)

Interpretation for the DQN: "basis dislocated extremely above recent norm,
still expanding, funding above 95th percentile, OI growing rapidly."
This is a textbook over-leveraged-long setup that often precedes long-squeezes.
```

State vector grows: 50 → 55 dims.

**Why these specific features**:
- **Each one is documented as a leading indicator** in published research on perpetual markets
- **All raw inputs already in meta parquet** — only feature engineering needed
- **Time-scales are diverse**: 1-bar (instant), 60-bar (1h), 120-bar (2h) — capturing different decay rates
- **Funding has a 8h periodicity** — z-score over 120 bars (= 15 funding cycles) is the right normalization

**Why I called this the strongest Z2 candidate**:
1. **Three macro signals (basis, funding, OI)** are completely absent from current state
2. **Documented edge** — published research shows these are leading indicators
3. **The fee experiment provided direct evidence** that important reward-side context belongs in state too
4. **Strategies that USE these signals already exist** (S2_Funding deployed, S11_Basis available) — the state additions complement them

**Why this might fail**:
- Funding only changes every 8h — most consecutive bars share the same value, so the funding features are mostly piecewise-constant. DQN might not extract signal from sparse-updating features.
- Basis has been consistently negative in this dataset (median −4.55 bps) — the policy might overfit on the asymmetry.
- OI change is normalized, but absolute OI levels (which matter for "is this big enough to move price") are not in state.

**Implementation pipeline** (~3-4 hours total):
1. Extend `models/dqn_state.py` to compute these 5 features from meta parquet columns (`fund_rate`, `oi_usd`, `spot_*_price`, `perp_*_price`).
2. Use `--state-version v7_basis` tag; regenerate state cache (`cache/btc_dqn_state_{train,val,test}_v7_basis.npz`).
3. Train 5 H256+DD seeds with `--state-version v7_basis`. Training time ~5 min/seed × 5 = ~25 min. Plus feature regen ~10 min.
4. Evaluate as K=5 plurality.

**Decision criterion**: keep if WF ≥ +11.55 (current +0.5) AND **val ≥ +3.0** (basis features should specifically help OOD val resilience — that's the main hypothesis).

**Expected lift**: +0.5 to +2.0 WF. Val should improve more than test because val period had volatile basis/funding swings (Jan-Feb 2026).

**Why this is run after Step 2**: Step 2 establishes that v7 state retrain works. Step 3 is more expensive (more features = more careful feature engineering) and we want the cheap pipeline-test to succeed first.

**Risk**: medium. Cost: ~3-4 hours. Expected lift: 0 to +2 Sharpe (highest expected value of any Z2 candidate).

#### Step 4 — Z3.1 wire & retrain (REVISED post Step 1)
**What**: Step 1 showed S11_Basis + S13_OBDiv carry unique signal types (basis momentum, cross-instrument OB disagreement) not covered by current 9. S5 + S9 dropped (redundant with S8). Add S11 + S13 only → 10 → 12 actions. Train 5 H256+DD seeds → `VOTE5_v8_H256_DD`.
**Decision**: keep if WF ≥ +11.55.
**Proof of value**: confirmed by Step 1. S11/S13 standalone Sharpes (−5.66 / −2.81 val) are no worse than currently-used S3 (−28) / S7 (−19.4), and S11 specifically covers basis-momentum which no current strategy uses. Direct test of the action-space-expansion hypothesis.
**Cost**: 5 seeds × ~5 min train + ~5 min signal regen = ~30 min.

#### Step 5 — Z2.5 combined state v7
**What**: stack survivors of Steps 2+3 into one expanded state. Retrain 5 seeds. Eval.
**Decision**: keep if WF ≥ max(Step 2, Step 3) + 0.3.
**Proof of value**: tests *additivity*. If price-action and basis features each lift independently, the question is whether they overlap (no additivity) or cover different decision contexts (additive). Cheap test gives a definitive answer.

#### Step 6 — Z3.2 S15_VolBreakout
**What**: Implement `strategy_14` with vol-ratio > 2.0 + 10-bar direction filter. Standalone-validate; if pass, expand action space again (12-15 actions including Step 4 survivors).
**Decision**: keep if standalone passes AND retrained ensemble ≥ +11.55.
**Proof of value**: this is the only truly-new strategy in the plan (no equivalent in existing 13). [Z3 feasibility check](z3_data_feasibility.md) showed `vol_ratio > 2.0` fires on ~6% of bars — non-trivial frequency, decent fire rate for a momentum continuation signal. None of S1-S13 fire on pure vol expansion.

### What we are NOT doing yet, and why

- **Z4 (architecture experiments — transformer, distributional RL, self-distillation)** — not started. Each is 1-2 days of new code. Cost-benefit favors exhausting cheaper Z2/Z3 first. We have evidence (Z1 results) that this axis has headroom; we'll come back to it.
- **Z5 (validation + freeze)** — gated; only runs once Z2/Z3/Z4 produce a winner clear enough to ship.
- **Path F (non-zero-fee)** — parked per user direction. The 7-item plan in [fee_improvement_proposals.md](fee_improvement_proposals.md) resumes when zero-fee path saturates OR if maker-only execution (Path X) proves infeasible.
- **Path X (maker-only execution scoping)** — engineering, separate track, not in this research plan.

### Decision rules (every step)

- **Pass criterion**: WF mean Sharpe lift ≥ +0.5 AND no fold worse than current baseline by >0.5.
- **Per-step deliverables (rule-enforced, see [.claude/rules/experiments.md](../.claude/rules/experiments.md))**:
  - `docs/{step_name}.md` with method, metrics, verdict
  - Every trained model registered in `model_registry.json`
  - `RESULTS.md` status block updated
  - All in **one commit** per step

### What this plan accomplishes if everything passes

If Steps 1-6 all clear their gates, the final ensemble at the end of Phase Z2/Z3 would be:
- Architecture: H256 + Double_Dueling (Z1)
- State: v7 with price-action + basis/funding features (Steps 2, 3, 5)
- Action space: 11-15 actions (Steps 4, 6 added 2-5 strategies to the original 9)
- Expected: WF ≥ +12.5, fold-6 ≥ +9, test ≥ +10, 6/6 folds positive

This becomes the input to Phase Z5 (validation + freeze) and is the candidate for production deployment.

### What this plan accomplishes if NOTHING passes

The combined cost is ~7 days of work. If every step fails, we still own:
- Hard evidence that the current baseline is at a local optimum on the (state, action, architecture) surface we've explored
- A clear case for jumping to Phase Z4 (architectural research bets)
- Or for accepting the current `VOTE5_H256_DD` as production-ready and moving to Path X (maker-only deployment)

**A negative outcome on Z2/Z3 is not a wasted investment** — it's the signal that the bottleneck has shifted to architecture or deployment.

---

## Execution status

| phase | status | notes |
|---|---|---|
| **Z1 — Stack proven winners** | ✅ **DONE 2026-05-10** | Winner: **`VOTE5_H256_DD`** (WF +11.05, test +9.01, fold-6 +8.23, 6/6 folds). See [z1_results.md](z1_results.md). |
| **Z2 — Better state** | 🔵 **NEXT** | Steps 2, 3, 5 above. Baseline: `VOTE5_H256_DD`. |
| **Z3 — Better signals** | 🟡 PRE-SCOPED | Steps 1 (validation) + 4 (retrain) + 6 (S15). [feasibility check](z3_data_feasibility.md) compressed plan. |
| Z4 — Architecture & training | ⚪ NOT STARTED | gated; defer until Z2/Z3 exhausted |
| Z5 — Validation & freeze | ⚪ NOT STARTED | gated on Z1–Z4 winner |
| Path F (non-zero-fee) | ⚫ PARKED | resume after Path Z winner OR if maker-only fails |

### Z1 outcomes
- ✅ **Z1.1 H256+DD** — WIN. Promoted as new candidate baseline.
- ❌ **Z1.2 K=10 vanilla** — NEGATIVE. Tie-driven NO_TRADE inflation; WF drops vs K=5.
- 🔍 **Z1.3 DD disjoint** — DIAGNOSTIC. DD val/WF magnitude is seed-sensitive; 6/6 folds robust.
- 🔍 **Z1.4 H128/H256 disjoint** — H128 EXPOSED as seed-luck (drop). H256 reproduces.

---

# Path Z — Zero-fee algorithm improvement (ACTIVE)

Goal: lift WF mean Sharpe meaningfully above +10.40 (vanilla VOTE5) or +11.86 (H256), while preserving 6/6 fold positivity and improving fold-6 robustness.

The zero-fee assumption corresponds to maker-only execution. Per [docs/fee_improvement_proposals.md](fee_improvement_proposals.md) #1, this is unblocked by Path X (maker-only scoping) which is engineering work, not research — separate track.

## Phase Z1 — Stack the proven winners (low-risk)

Test combinations of independently-validated improvements. None of these have been tried together; each has been validated in isolation. Phase Z1 is the cheapest with the highest probability of incremental progress — it doesn't require new logic, just new training runs with existing flags.

| ID | Experiment | Why | Cost |
|---|---|---|---|
| Z1.1 | **H256 + Double_Dueling** (5 seeds) | H256 wins WF (+11.86), DD wins val (+6.12). Stack: capacity + regularization | ~30 min × 5 = 2.5 h |
| Z1.2 | **K=10 vanilla VOTE10** (10 seeds, plurality) | K=5 already wins. K=10 should reduce ensemble variance further | ~6 h training (5 new seeds) |
| Z1.3 | **K=10 Double_Dueling VOTE10_DD** | If Z1.2 helps, repeat for DD | ~6 h |
| Z1.4 | **Disjoint validation of H256 + H128** | Confirm capacity gains are structural, not seed-luck | ~6 h |

**Decision gate**: Z1 winners go to Phase Z2 as the new baseline. WF +12 with 6/6 folds positive is the target.

### Z1 detailed sub-plan

#### Z1.1 — H256 + Double_Dueling stack

**Hypothesis**: H256 (capacity) and DD (regularization) are orthogonal axes. H256 lifts WF aggregate (+11.86 vs +10.40) but hurts fold-6 (+0.41 vs +5.20). DD wins val (+6.12 vs +3.53) and 6/6 folds with smoother per-fold distribution. **Stacking them should preserve H256's WF lift while DD's regularization recovers fold-6 robustness.**

**Method**:
```
for SEED in 42 7 123 0 99; do
  python3 -m models.dqn_selector btc \
    --tag VOTE5_H256_DD_seed${SEED} \
    --hidden 256 --algo double_dueling \
    --fee 0.0 --trade-penalty 0.001 --seed ${SEED}
done
```
Then evaluate as K=5 plurality ensemble at fee=0 across val, test, 6 WF folds.

**Code touchpoints**: no new code — `--hidden 256 --algo double_dueling` already supported in [models/dqn_selector.py](../models/dqn_selector.py).

**Decision criterion**: keep if WF ≥ +11.5 AND fold-6 ≥ +3.0 AND 6/6 folds positive. (i.e. preserves most of H256's WF lift AND recovers most of DD's fold-6.)

**Risks**:
- Higher capacity nets are slower to train; 5 seeds × ~5 min = 25 min training
- Combined regularization might *over-regularize* — DD on top of H256's already-larger param count could damp signal
- Per-seed variance is bigger at h=256 (capacity_test showed seed=42 only +8.92 WF vs the +11.86 ensemble)

**Expected outcome**: WF +11.0 to +12.5, fold-6 +3 to +6, 5/6 to 6/6 folds. If outside this range, the stack didn't compose.

#### Z1.2 — K=10 plurality ensemble (vanilla DQN)

**Hypothesis**: K=5 already smooths variance vs single-seed (+10.40 vs +9.03). K=10 should smooth further, especially on fold-6 (single-seed std ~±2.17 in seed_variance). The disjoint validation already gave us 5 *additional* seeds (1, 13, 25, 50, 77) that produced WF +10.06 — combining all 10 should beat both K=5 baselines.

**Method**:
- Members already trained: seeds 42, 7, 123, 0, 99 + 1, 13, 25, 50, 77 (10 total).
- No new training. Just construct K=10 plurality ensemble at evaluation:
```python
nets = [load_dqn(tag) for tag in [
    "BASELINE_FULL", "BASELINE_FULL_seed7", ..., "BASELINE_FULL_seed77"]]
# 10 nets, plurality mode, tie → NO_TRADE
```
Run through `evaluate_with_policy` on val + test + 6 WF folds.

**Code touchpoints**: no training. Just an eval script that calls [`models/voting_ensemble.py`](../models/voting_ensemble.py) `_VotePolicy(nets, mode="plurality")` with 10 nets.

**Decision criterion**: keep if WF ≥ +10.6 (the disjoint K=5) AND fold-6 ≥ +5.0 AND 6/6 folds positive. (Improves on the better of the two K=5 baselines.)

**Risks**:
- With 10 nets and plurality voting, ties become more frequent (10 has more ways to split than 5). Tie → NO_TRADE means more skipped trades, hurting Sharpe via √N.
- Could be that 5 already saturates the ensemble benefit and 10 just adds compute.

**Cost note**: marked ~6 h above but actual cost is **0 training, ~5 min eval** since all 10 seeds are already trained.

#### Z1.3 — K=10 Double_Dueling

**Hypothesis**: same as Z1.2 but for DD. Currently we only have 5 DD seeds (42, 7, 123, 0, 99). Need 5 additional seeds for K=10.

**Method**:
```
for SEED in 1 13 25 50 77; do
  python3 -m models.dqn_selector btc \
    --tag VOTE10_DD_seed${SEED} --algo double_dueling \
    --fee 0.0 --trade-penalty 0.001 --seed ${SEED}
done
```
Then K=10 plurality across all 10 DD seeds.

**Code touchpoints**: no new code.

**Decision criterion**: gated on Z1.2. If Z1.2 doesn't show K=10 benefit over K=5, skip Z1.3 entirely (don't burn 25 min training for no expected gain).

**Risks**:
- DD already has more bias/variance than vanilla DQN; K=10 may not buy as much
- The 5 new seeds may not match the original 5's distribution

#### Z1.4 — Disjoint validation of H256 + H128

**Hypothesis**: H128 wins fold-6 (+10.70) but is 5/6 folds. H256 wins WF aggregate (+11.86) but fold-6 is +0.41. Could these results be seed-specific (we used the same 5 seeds 42/7/123/0/99 for both capacity tests)? Train another 5 seeds at h=128 and h=256 with the disjoint pool {1, 13, 25, 50, 77} and compare.

**Method**:
```
for H in 128 256; do
  for SEED in 1 13 25 50 77; do
    python3 -m models.dqn_selector btc \
      --tag VOTE5_H${H}_DISJOINT_seed${SEED} \
      --hidden ${H} --fee 0.0 --trade-penalty 0.001 --seed ${SEED}
  done
done
```
Then evaluate two ensembles (H128_DISJOINT, H256_DISJOINT) and compare to the original.

**Code touchpoints**: no new code.

**Decision criterion**: keep if disjoint ensemble reproduces ±0.5 of original ensemble's WF AND fold-6. If disjoint diverges sharply, the original capacity result was seed-luck.

**Risks**:
- 10 new training runs (~30-50 min total at h=128, ~50-70 min at h=256). Substantial wall-time investment for what is essentially a sanity check.
- If results diverge, undermines confidence in H128/H256 as deployable baselines.

**Why it matters**: any capacity-based deployment decision needs disjoint-validated evidence. Currently H128/H256 baselines are 1-pool of 5 seeds — same epistemic standard as before plurality voting was validated structurally.

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

### Z2 detailed sub-plan

The current state vector is 50 dims, defined in [models/dqn_state.py](../models/dqn_state.py). Adding new features means: regenerate `cache/btc_dqn_state_*.npz` with the expanded vector, retrain VOTE5 (5 seeds), evaluate. State dimension changes from 50 → 50 + N where N is the number of added features.

**Common implementation per Z2 experiment**:
1. Add feature computation in [features/](../features/) (cache result if expensive).
2. Modify `_assemble_state(...)` in [models/dqn_state.py](../models/dqn_state.py) to append the new features to the existing state vector.
3. Update `state_dim` parameter in `models/dqn_selector.py` (currently hardcoded to 50 in places) — **add a `--state-dim` CLI flag** if not present.
4. Use a new `--state-version` tag (e.g. `v7_xasset`) to avoid overwriting the v5 cache.
5. Train 5 seeds with `--state-version v7_xasset --tag Z2_xasset_seed{S}`.
6. Evaluate as K=5 plurality on val + test + WF.

**Per-experiment specifics**:

#### Z2.1 — Cross-asset state (ETH + SOL features)

**Hypothesis**: BTC, ETH, SOL move with strong correlation but with lag relationships. ETH typically leads BTC by minutes in trend transitions. Adding lagged ETH/SOL signals should give the policy advance warning of regime shifts.

**Features added** (8 dims, all lagged by 1+ bars per data-integrity rule):
```
state[50] = ETH_return_1bar       (lagged 1 bar)
state[51] = ETH_return_5bar       (lagged 1 bar)
state[52] = ETH_vol_ratio         (ATR_30 / median_ATR_60)
state[53] = ETH_taker_imbalance   (net taker flow)
state[54] = SOL_return_1bar       (lagged 1 bar)
state[55] = SOL_return_5bar
state[56] = SOL_vol_ratio
state[57] = SOL_taker_imbalance
```

**Code touchpoints**:
- Feature extraction: extend [features/assembly.py](../features/) to compute ETH/SOL features from `cache/okx_ethusdt_*` and `cache/okx_solusdt_*` parquets.
- State assembly: append to `_assemble_state` in `models/dqn_state.py`.

**Decision criterion**: WF ≥ +10.9 (i.e., ≥+0.5 over Z1 winner) AND no fold worse by >0.5.

**Risks**:
- ETH/SOL data may not align timestamp-perfectly with BTC. Need to verify alignment in `data/loader.py`.
- Cross-asset features add 8 dims to state, growing param count. Could reduce sample efficiency.
- Lead-lag is unstable across regimes — strong in some periods, weak in others.

**Expected**: small lift (+0.3 to +1.0 WF) if cross-asset signal is genuinely orthogonal; null if BTC's own features already capture macro state.

#### Z2.2 — Perp basis + funding state

**Hypothesis**: Perp-spot basis and funding rate changes are leading indicators of leverage-driven moves (squeezes, liquidations). Currently in features but not in state.

**Features added** (5 dims):
```
state[50] = basis_z_60       (perp/spot z-score over 60 bars)
state[51] = basis_change_1bar (delta basis)
state[52] = funding_rate_apr  (current 8h rate annualized)
state[53] = funding_z_120     (funding z-score over 120 bars)
state[54] = oi_change_60      (open interest change rate, normalized)
```

**Code touchpoints**:
- All inputs already in `meta` parquet. Just compute and append to state.

**Decision criterion**: WF ≥ +10.9 AND val ≥ +3.0 (basis features should help OOD val period).

**Risks**:
- Funding only changes every 8h → 5/8 of the "funding signal" is constant within a window. Z-score within rolling 120 may catch transitions but the signal is sparse.
- Basis Z-score behavior is asymmetric in this dataset (median basis = −4.55 bps). Z-score normalization handles it but the policy may overfit on the asymmetry.

**Expected**: this is the most promising Z2 candidate IMO. Funding/basis are the kind of "macro context" current state lacks.

#### Z2.3 — OB depth features (deeper liquidity context)

**Hypothesis**: Current state has summary OB metrics (`spot_imbalance`, `bid_concentration`). Adding cumulative depth at multiple price ranges (±2%, ±5%, ±10%) gives the policy more nuanced liquidity context — useful for trade quality conditioning.

**Features added** (6 dims):
```
state[50] = spot_depth_2pct_imbalance   (bid - ask) / (bid + ask) at ±2%
state[51] = spot_depth_5pct_imbalance
state[52] = spot_depth_10pct_imbalance
state[53] = perp_depth_2pct_imbalance
state[54] = perp_depth_5pct_imbalance
state[55] = perp_depth_10pct_imbalance
```

**Code touchpoints**:
- Feature extraction: compute from raw OB parquet (~2 min per pass for full 384k rows × 800 cols).
- Cache result to `cache/btc_ob_depth_features.parquet`.

**Decision criterion**: WF ≥ +10.9. Fold-6 specifically should hold (recent regime is liquid; depth signal less informative there).

**Risks**:
- OB depth is noisy at 1-min granularity. Spoofing affects deeper price ranges.
- Computing 200 bin amounts × 384k rows is slow first-pass; cache is essential.

**Expected**: marginal lift unless current `spot_imbalance` summary is too lossy. Lower priority than Z2.2.

#### Z2.4 — Price action context (cheapest)

**Hypothesis**: Recent price extremes (run-up/down magnitude over last 60 bars) and realized vol are not in state. These are simple features that condition the policy on current trend strength.

**Features added** (4 dims):
```
state[50] = price_max_60 / price_now - 1   (how far from recent high)
state[51] = price_now / price_min_60 - 1   (how far from recent low)
state[52] = realized_vol_60                (std of returns over 60 bars)
state[53] = vol_ratio_30_60                (ATR_30 / ATR_60)
```

**Code touchpoints**: trivial — all derivable from price array.

**Decision criterion**: WF ≥ +10.9.

**Risks**:
- Likely redundant with regime-id and ATR already in state.
- Cheapest experiment in Z2 (~30 min training); even if it fails, low cost.

**Expected**: small or null lift. Run as warm-up because it's cheap.

#### Z2.5 — Combined state v7

**Hypothesis**: Survivors of Z2.1-Z2.4 may be additive. Stacking them into one expanded state could lift more than any single feature group.

**Method**: combine all features that survived their individual decision gates into one state vector. Train VOTE5_v7 with the combined state. Compare to Z1 winner and to each Z2.x sub-result.

**Decision criterion**: WF ≥ max(Z2.1, Z2.2, Z2.3, Z2.4) + 0.3 (combined must improve on best singleton).

**Risks**:
- State dim could balloon (e.g. 50 + 8 + 5 + 6 + 4 = 73 dims). Higher capacity needed for the same information density.
- Some features may be redundant across groups (e.g. `vol_ratio_30_60` in Z2.4 partially overlaps with the regime classifier's input).

## Phase Z3 — Better signals (medium-risk)

> **Plan compressed 2026-05-10 after data feasibility check.** [docs/z3_data_feasibility.md](z3_data_feasibility.md) revealed that `strategy/agent.py` already defines **13** strategies but only 9 are wired into the DQN action space. The 4 unused strategies (S5, S9, S11, S13) overlap with the originally-proposed S14+S16. S2_Funding (already wired in) covers my originally-proposed S13_FundingExtreme. Net result: the original 4-strategy plan compresses to **2 work items**. Original proposal preserved below as historical context.

| ID | Action | Cost | Why |
|---|---|---|---|
| **Z3.1** | Wire existing-but-unused `S5_OFISpike`, `S9_LargeImbalance`, `S11_BasisMomentum`, `S13_OBDisagreement` into `STRAT_KEYS`. Standalone-validate each via [backtest/run.py](../backtest/run.py); drop weak ones, retrain `VOTE5_v8` with expanded action space (10 → up to 14). | ~1 day | Code already exists; tests existing-but-untried alpha at the cost of just validation + retrain. Cheapest signal expansion. |
| **Z3.2** | Implement genuinely-new `S15_VolBreakout` as `strategy_14` (`vol_ratio = ATR_30 / median(ATR_60) > 2.0` + 10-bar direction filter). | ~0.5 day | No equivalent in current 13. Calibrated threshold from feasibility check (1.5 → 2.0; 25% → 6% fire rate). |
| ~~Z3.3~~ | ~~Original S13_FundingExtreme~~ | — | Subsumed by existing `S2_Funding` (filtered variant) + dataset's funding range too narrow for unfiltered version. |
| ~~Z3.4~~ | ~~Custom S14_DepthImbalance, S16_BasisDislocation~~ | — | Subsumed by existing `S9_LargeImbalance`, `S11_BasisMomentum` once wired in. |

**Decision gate (Z3.1)**: each rewired strategy must achieve win-rate > 50% on DQN-val AND mean PnL/trade > 0.15% via standalone backtest. Drop weak ones before retraining VOTE5_v8.

**Decision gate (Z3.2)**: same standalone-validation criteria as Z3.1.

**Final retrain decision**: keep `VOTE5_v8` if WF mean Sharpe ≥ +10.40 (current `BASELINE_VOTE5`) with no fold worse than current by >0.5.

### Original Z3 proposal (historical reference, before feasibility check)

The original proposal added 4 new strategies (S13–S16). It's preserved below for context. Most of the conceptual work translates: the *same alpha-discovery surface* is covered by Z3.1 (existing code) + Z3.2 (one truly new strategy).

### Z3 detailed sub-plan

The current 9 strategies in [strategy/agent.py](../strategy/agent.py) cover momentum (S1, S4, S8, S10), reversion (S3, S6), and microstructure (S7, S12). Gaps: **funding-rate signal**, **OB-depth signal**, **pure vol-expansion**, and **basis dislocation**. The four new strategies fill these.

**Implementation common to all 4**:

1. Add the strategy class to `strategy/agent.py` with `signal_for_bar(t) → {-1, 0, +1}` and `DEFAULT_PARAMS`.
2. Add the key to `STRAT_KEYS` in [models/dqn_rollout.py](../models/dqn_rollout.py) at index 9, 10, 11, 12 (after current 9).
3. Add execution config to [execution/config.py](../execution/) with TP/SL/trail/be/timestop matching the strategy's expected bar holding (short for breakouts, long for funding mean-reversion).
4. Regenerate `cache/btc_dqn_state_*.npz` to include the new signal columns (`signals[t, k]` shape becomes (n_bars, 13)).
5. Re-train BASELINE_VOTE5 with expanded action space → call it `VOTE5_v8` (state v5 unchanged; v8 is the new action space).

**Standalone validation first** (before retraining VOTE5):
- For each new strategy, run [backtest/run.py](../backtest/run.py) standalone with rule-based exits across the full WF period.
- Required pre-retrain criteria: **win-rate > 50% on DQN-val** AND **mean PnL/trade > 0.15%** (median across current 9 strategies). Otherwise the action will be NO_TRADE-dominant and waste DQN capacity on a dead action.

**Per-strategy specifics**:

#### Z3.1 — `S13_FundingExtreme` (funding mean-reversion)

**Signal logic**:
```python
# data already in features: fund_rate column
# Funding paid every 8h on OKX perp. Convert to APR for thresholding.
funding_apr = fund_rate * (365 * 24 / 8)   # rough APR
threshold = 0.50   # 50% APR (~+0.46% per 8h) is "extreme"

if funding_apr > threshold:    return -1   # short — market over-leveraged long
if funding_apr < -threshold:   return +1   # long  — market over-leveraged short
return 0
```

**Why it should work**: extreme funding indicates one-sided leverage. When new funding payments hit, leveraged side is taxed and unwinds. Documented edge in perp markets.

**Risks**:
- Funding can stay extreme for days (e.g. strong trend periods). Time-stop must be tight.
- Mean-reversion may coincide with trend reversal — direction edge real, but could be late.

**TP/SL config** (different from existing strategies — funding is a slow signal):
- TP: 1.5 ATR
- SL: 1.0 ATR
- Trail: 0.8 ATR (no early-be required — funding signal is slow)
- Time-stop: 240 bars (4 h). Funding pays every 8 h; we need to be in for at least one tick.

#### Z3.2 — `S14_DepthImbalance` (OB-driven directional)

**Signal logic** (uses raw OB parquet):
```python
# Cumulative depth at top 2% of OB on each side
bid_depth_2pct = sum(bid_amounts[bins where price >= best_bid * 0.98])
ask_depth_2pct = sum(ask_amounts[bins where price <= best_ask * 1.02])
imbalance = (bid_depth_2pct - ask_depth_2pct) / (bid_depth_2pct + ask_depth_2pct)

# microprice (size-weighted mid)
microprice = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
mid = (best_bid + best_ask) / 2
micro_drift = (microprice - mid) / mid

# Signal: large depth imbalance AND microprice agrees
if imbalance > 0.40 and micro_drift > 0:   return +1   # bid pressure + micro up
if imbalance < -0.40 and micro_drift < 0:  return -1
return 0
```

**Why it should work**: large resting size + microprice drift in the same direction is informative about near-term price (1-5 bars). Existing taker-flow strategy (S8) measures executed flow, this measures *resting* pressure.

**Risks**:
- OB depth can be spoofed; large orders sometimes pulled at price approach.
- Depth on OKX during low-volume periods is noisy.

**TP/SL config** (microstructure → fast):
- TP: 1.0 ATR
- SL: 0.6 ATR
- Trail: 0.4 ATR
- BE: 0.5 ATR (move SL to BE quickly — microstructure trades are short-horizon)
- Time-stop: 30 bars (30 min)

#### Z3.3 — `S15_VolBreakout` (pure volatility expansion)

**Signal logic**:
```python
atr_30 = ATR over last 30 bars
atr_baseline_60 = median(ATR over last 60 bars)
vol_ratio = atr_30 / atr_baseline_60

recent_dir = sign(price[t] - price[t-10])   # last 10-bar direction

if vol_ratio > 1.5 and recent_dir != 0:
    return recent_dir   # continuation in dominant direction
return 0
```

**Why it should work**: vol expansion + directional bias = breakout. None of the current strategies fire purely on vol expansion; we have S10_Squeeze (Bollinger contraction → expansion) but it is conditional and infrequent (15-27 trades).

**Risks**:
- Can fire late — by the time vol_ratio > 1.5, a chunk of the move is done.
- Whipsaws in chop regimes that briefly spike vol.

**TP/SL config** (breakouts → wide TP, ATR-scaled exits):
- TP: 2.5 ATR
- SL: 1.0 ATR
- Trail: 1.2 ATR (let breakouts run)
- BE: 1.0 ATR
- Time-stop: 120 bars (2 h)

#### Z3.4 — `S16_BasisDislocation` (perp-spot mean reversion)

**Signal logic** (uses spot+perp price columns):
```python
basis_bps = (perp_mid - spot_mid) / spot_mid * 10000
basis_z = (basis_bps - rolling_mean_basis_60) / rolling_std_basis_60

if basis_z > 2.0:   return -1   # perp expensive → short perp (we trade perp only)
if basis_z < -2.0:  return +1   # perp cheap → long perp
return 0
```

**Why it should work**: perp-spot basis tends to mean-revert when not driven by funding shifts. Z-score normalization catches dislocation regardless of trend.

**Risks**:
- Strong sustained trends can drive basis to high z-scores legitimately. Time-stop required.
- Most informative when basis decouples from funding-rate trend; must check overlap with S13 to avoid double-counting.

**TP/SL config** (slow mean-reversion):
- TP: 1.2 ATR
- SL: 0.8 ATR
- Trail: 0.6 ATR
- BE: 0.5 ATR
- Time-stop: 180 bars (3 h)

### Z3 expected outcome

If 2+ strategies clear standalone-validation gates, retrain `VOTE5_v8` with action space (NO_TRADE, S1, …, S12, S13, S14, S15, S16) → 14 actions. The DQN learns when to pick each. Expected Sharpe lift: +0.5 to +1.5 if at least one strategy is strongly orthogonal to existing ones. Lower if the new strategies overlap with current signal coverage.

---

## Phase Z4 — Architecture & training (higher-risk)

| ID | Experiment | Hypothesis | Cost |
|---|---|---|---|
| Z4.1 | **Self-distillation**: train a single net to mimic VOTE5 plurality output | One net deploys faster + more interpretable than 5-seed ensemble. Loss: maybe equal to VOTE5 | ~1 day |
| Z4.2 | **Transformer-block on state** (1 layer, 4 heads) | Sequence-aware, but state is already 50-dim engineered features — may not help | ~2 days |
| Z4.3 | **Curriculum learning by regime difficulty** | Train on calm regimes first, then gradually include trending/chop bars. Could improve fold-6 robustness | ~1 day |
| Z4.4 | **Distributional RL (C51 / QR-DQN)** | Predict Q distribution, choose by quantile / CVaR. Better for tail-aware decisions | ~2 days |

**Decision gate**: each Z4 experiment must beat the current Z2/Z3 winner on WF AND on fold-6. Otherwise dropped.

### Z4 detailed sub-plan

#### Z4.1 — Self-distillation

**Goal**: one trained DQN that approximates VOTE5 plurality output, so deployment is 1× inference rather than 5× plus voting logic.

**Approach**:
1. Run BASELINE_VOTE5 (5 nets, plurality voting) over the full RL period and the train+val splits → record `(state, vote5_action, vote5_action_distribution)` for every bar.
2. Train a fresh DQN with **two loss components**:
   - **Distillation**: cross-entropy between student's softmax-Q over actions and the teacher's vote distribution (5 nets' votes counted, normalized to a soft 5-way distribution).
   - **TD regularizer**: standard Bellman loss on the rollout (with smaller weight, e.g. 0.1×).
3. Use the same state/replay infrastructure; new training script `models/distill_vote5.py`.

**Architecture**: same DQN(50, 10, 64) — no architecture change, just loss change.

**Decision criterion**: student's WF Sharpe ≥ teacher's − 0.3. (Allow small loss for the deployment-simplicity gain.)

**Risks**:
- Distillation typically loses ensemble's tail-risk benefits — variance reduction comes from disagreement, not from capacity.
- May discover that 1-net-distilled performs *better* than VOTE5 if voting was averaging noise rather than smoothing real disagreement.

**Files to create**:
- `models/distill_vote5.py` — training loop with combined loss
- `cache/btc_dqn_distill_targets.npz` — cached teacher actions+distributions
- `docs/distill_vote5.md` — results doc

#### Z4.2 — Transformer block on state

**Goal**: replace the MLP body with a transformer block to enable attention over state features. Hypothesis: the MLP treats all 50 dims as independent linear combinations; attention may discover non-linear feature interactions (e.g. "high vol_ratio AND regime=trend AND signal_S1 active" as a joint pattern).

**Two architectural options**:

**Option A — flat tokenization** (lightweight):
```
state (50,) → reshape (50, 1) → linear (50, d_model=16)
            → +positional embedding (50, 16)
            → transformer block (4 heads, FFN=64) (50, 16)
            → mean-pool → linear (16, n_actions=10)
```
~30k params (vs current 6k).

**Option B — semantic grouping** (interpretable):
```
state (50,) → group into 6 tokens:
   regime tokens (3 dims, regime_one_hot)
   signal tokens (9 dims, per-strategy directions)
   vol tokens   (6 dims, ATR / regime probability)
   micro tokens (8 dims, OB / taker)
   equity tokens (2 dims, last_pnl + drawdown)
   misc tokens  (rest)
   → linear (each_group, d_model=16)
   → 6 tokens × 16 dim
   → +positional embedding
   → transformer block
   → cls/pool → linear → actions
```

**Decision criterion**: WF Sharpe ≥ +12, OR fold-6 ≥ +6. Otherwise the extra params + slower training aren't justified.

**Risks**:
- State is already engineered features — attention may have nothing to discover beyond what MLPs find.
- Larger network → easier to overfit on the 150k train bars (already a concern at h=256).
- Inference latency higher (matters less for 1-min bars but still).

**Files**:
- `models/dqn_network.py` — add `TransformerDQN` class.
- `models/dqn_selector.py` — `--algo transformer` (or `--arch transformer` to pair with double/dueling).
- `docs/transformer_test.md`.

#### Z4.3 — Curriculum learning by regime difficulty

**Goal**: improve fold-6 robustness (the recent-regime fold) by training the policy on simpler patterns first.

**Approach**:
1. Use the existing `regime_id` in state. Define difficulty: `calm` (0) easiest → `trend_up`/`trend_down` (1, 2) medium → `ranging` (3) → `chop` (4) hardest.
2. Three training phases sharing a single net:
   - Phase 1 (steps 0-50k): bars with regime ∈ {calm}. Other bars → policy sees state but is excluded from buffer.
   - Phase 2 (steps 50k-120k): regime ∈ {calm, trend_up, trend_down}.
   - Phase 3 (steps 120k-200k): all bars, all regimes.
3. Implement as a sample-mask in `rollout_chunk` rather than truncating the price series — this preserves the equity / last_pnl context.

**Alternative formulation**: weighted sampling instead of strict masking — bars get weights `{calm: 4, trend: 2, ranging: 1, chop: 0.5}` decaying toward uniform over training.

**Decision criterion**: improves fold-6 by ≥+2 Sharpe vs baseline (current fold-6 is +5.20 for VOTE5). Aggregate WF must not drop by more than 0.3.

**Risks**:
- Curriculum can over-specialize on early stages; the transition to harder regimes may erase calm-regime competence.
- Regime labels themselves are noisy (CUSUM-based, not perfect). Difficulty may be miscategorized.

**Files**:
- `models/dqn_rollout.py` — add `regime_filter` argument to `rollout_chunk`.
- `models/dqn_selector.py` — `--curriculum` flag with phase boundaries.
- `docs/curriculum_test.md`.

#### Z4.4 — Distributional RL (C51 / QR-DQN)

**Goal**: predict the *distribution* of Q-values per action, not just the mean. Action selection by CVaR (conservative, weights tail risk) instead of `argmax E[Q]`. Hypothesis: better fee-robustness and better fold-6 (high-variance regime).

**Approach** (start with QR-DQN, simpler than C51):
1. Architecture: final layer outputs (n_actions × n_quantiles). Use 32 quantiles.
2. Loss: quantile regression loss (Huber-quantile, asymmetric weighting per quantile).
3. Action selection at inference:
   - **mean policy**: `argmax E[Q] = argmax mean(quantiles)` (matches standard DQN behavior)
   - **CVaR policy** (new): `argmax CVaR_α[Q]` where `CVaR_α = mean(quantiles[:floor(α*n_quantiles)])` — i.e., expected Q over the worst α% outcomes. With α=0.3, action chosen by its 30%-worst-case expected return. Trade goes only if even the worst case is positive.
4. Train with both, eval both.

**Decision criterion**: CVaR policy beats baseline mean policy on (a) fold-6 Sharpe by ≥+1.5, OR (b) fee=4.5bp WF Sharpe by ≥+1.0. Otherwise mean policy reverts to baseline DQN behavior — useful diagnostic but no shipping value.

**Risks**:
- CVaR is conservative — may shrink trade count too much, killing Sharpe via √N.
- Quantile estimation is noisy; need more training data than point-estimate Q.
- Implementation complexity: must rewrite `bellman_loss`, `masked_argmax`, replay sampling.

**Files**:
- `models/dqn_network.py` — add `QRDQN(state_dim, n_actions, n_quantiles=32)` class.
- `models/qr_loss.py` — quantile regression loss.
- `models/dqn_selector.py` — `--algo qrdqn` and `--cvar-alpha 0.3` flags.
- `docs/qrdqn_test.md`.

### Z4 phase summary

The four Z4 experiments are mostly independent — Z4.1 (distillation) operates on top of an existing teacher, Z4.2/Z4.4 are architecture changes, Z4.3 is a training-procedure change. Run independently first, decide which (if any) to combine.

**Most likely winner ranked**: Z4.3 (curriculum) — cheap to test, directly targets the known fold-6 weakness. **Most expensive bet**: Z4.4 (distributional) — rewrite needed but addresses the regularization/tail-risk axis where DD already showed promise.

## Phase Z5 — Validation & freeze (mandatory before any deployment)

Z5 is **gate-keeping**, not exploratory. After Z1–Z4 produce a winning policy, Z5 is the disciplined freeze process: stress-test, confirm reproducibility, document, and lock the artifact.

| ID | Step | Why |
|---|---|---|
| Z5.1 | **Out-of-distribution stress test**: run final policy on Apr-May 2026 data not yet used | Locked test only used once; this is the second OOD check |
| Z5.2 | **Seed variance ±2σ check** with 10 seeds | Confirm ensemble Sharpe range is wholly above +10.40, not borderline |
| Z5.3 | **Fee-curve at non-zero fees** for the new winner | Even though path is zero-fee-targeted, the fee curve tells us robustness margin |
| Z5.4 | **Freeze new baseline** in `docs/baselines.md` + tag in registry | Reproducibility requirement |

### Z5 detailed sub-plan

#### Z5.1 — Out-of-distribution stress test

**Hypothesis**: Locked test split was used once during Z1–Z4 to make decisions. Any Z-phase winner has had >1 evaluation on it (per fold + final pick), so test is no longer truly OOD. Need a fresh slice that no policy has seen.

**Method**:
- Hold out the most-recent ~20-30 days of data (post-2026-04-01 if not already used) as **`OOD_LOCK`**. Verify it's not in `train`, `val`, or `test`.
- Single-shot evaluation only — record `OOD_LOCK` Sharpe + equity once; never iterate on it.
- If `OOD_LOCK` data isn't yet collected, this is a **prerequisite** before deployment: pull fresh data + recompute features.

**Decision criterion**: Sharpe on `OOD_LOCK` ≥ 0.7 × WF mean. (Allow some degradation; pure equality would be unrealistic.) If `OOD_LOCK` is sharply negative, the Z-phase winner overfit to the WF distribution and shouldn't ship.

**Code touchpoints**:
- New script `models/eval_ood_lock.py` that loads policy + OOD slice + computes single-shot metrics.
- New cache file `cache/btc_dqn_state_ood_lock.npz` for the OOD state arrays.

**Risks**:
- May not have enough OOD data on hand. Pulling fresh data is a separate engineering task (~1 day).
- Even OOD periods can correlate with prior periods if regime is similar.

#### Z5.2 — Seed variance with 10 seeds

**Hypothesis**: A K=5 ensemble's reported metric is one draw from a distribution. Need to confirm the ±2σ band is wholly above the previous baseline (+10.40 vanilla VOTE5), not borderline.

**Method**:
- For the Z-phase winner config (e.g. `VOTE5_H256_DD`), train **10 seeds** instead of 5. Use seeds {42, 7, 123, 0, 99, 1, 13, 25, 50, 77}.
- Run two K=5 ensembles using disjoint pools: `{42, 7, 123, 0, 99}` and `{1, 13, 25, 50, 77}`. Each gives one Sharpe.
- Bootstrap K=5 ensembles by sampling 5 of 10 seeds, 100 times → distribution of K=5 ensemble Sharpes.
- Report: mean, ±2σ band, min, max.

**Decision criterion**:
- Mean WF Sharpe > +10.40 (beats current baseline)
- Mean − 2σ > +9.0 (lower bound is still strong)
- No single sampled K=5 ensemble below +7.5 (no catastrophic seed combinations)

**Code touchpoints**:
- New script `models/seed_variance_z5.py` that does the bootstrap sampling.
- Reuses [`models/voting_ensemble.py`](../models/voting_ensemble.py).

**Risks**:
- 10 seeds × Z-phase-winner training time = significant cost (could be 1-2 hours per config).
- Bootstrap of 100 samples is fine for variance estimate; smaller may be noisy.

#### Z5.3 — Fee-curve robustness check

**Hypothesis**: Zero-fee winner still needs to be characterized at non-zero fees. If Sharpe collapses at fee=2 bp, deployment depends on maker tier (more risk). If it holds at fee=4.5 bp, we have margin.

**Method**: Reuse [`models/audit_vote5_dd.py`](../models/audit_vote5_dd.py) Part B-style fee sweep on the Z-phase winner. Evaluate at fees ∈ {0, 1, 2, 4, 4.5, 6, 8} bp/side.

**Decision criterion**: characterize the curve, no pass/fail. Used to inform deployment risk management — if Sharpe → 0 at 2 bp, capital allocation must reflect that fee sensitivity.

**Code touchpoints**: parameterize the existing fee-sweep script with a `--policy-tag` flag.

**Risks**: none — diagnostic.

#### Z5.4 — Freeze + register + document

**Method**:
1. Add policy entry to `model_registry.json` per [`.claude/rules/model-registry.md`](../.claude/rules/model-registry.md). Include all eval results from Z5.1-Z5.3.
2. Add the policy to [docs/baselines.md](baselines.md) with full reproduction spec (training command, seed list, hyperparameters, eval results).
3. Update [RESULTS.md](../RESULTS.md) status block with the new frozen baseline as the headline.
4. Update this development plan: mark Z1–Z5 phases as DONE in the execution-status table; record the frozen baseline tag.
5. Tag the git commit as `baseline-{name}-frozen-YYYY-MM-DD` for fast retrieval.

**Decision criterion**: nothing — Z5.4 is mechanical. Either the Z5.1-Z5.3 results pass the bar or they don't.

**Risks**:
- Documentation drift: future-me will need to find this baseline by tag. If the tag isn't applied, the baseline becomes hard to reconstitute later.

### Z5 sequencing

Z5.1, Z5.2, Z5.3 can run in parallel (independent computations). Z5.4 must run last (it's the freeze). Total wall-time: ~3-5 hours depending on Z5.2's training cost.

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

### Path F detailed sub-plan (high-level — full plan in [fee_improvement_proposals.md](fee_improvement_proposals.md))

#### F1 — Cheap post-hoc improvements (highest cost-benefit)

**Why first**: all four sub-experiments are no-retrain or minor-retrain. Combined wall-time ~4 h. Even small Sharpe lifts compound when fees are eating most of the alpha.

| F1 sub | Approach | Expected lift @ 4.5 bp |
|---|---|---|
| F1.1 | **Vote-strength sizing** — size = {3v: 0.4, 4v: 0.7, 5v: 1.0}. A2 audit showed 5-vote trades have ~3× mean PnL of 3-vote trades. | +0.5 to +1.5 Sharpe |
| F1.2 | **Q-margin threshold** — `Q[a*] − Q[no_trade] ≥ τ` per net before voting. Calibrate τ on train. | +0.3 to +0.8 |
| F1.3 | **Tighter TP for trend strategies** — `--tp-scale 0.85` and `0.70` (audit follow-up #4 from `audit_followup_tests.md`, never run) | +1.0 to +2.0 |
| F1.4 | **Funding-rate offset in reward** — add `fund_rate × bars_held / 525960` to per-trade pnl. Marginal trades may flip net-positive. | +0.5 (asymmetric: helps shorts in bear funding) |

**Decision criterion**: each sub-experiment passes if WF lift ≥ +0.3 at fee=4.5 bp with no fold worse by >0.5. Combine survivors in a stacked test.

#### F2 — Maker-fill-rate scoping (engineering)

**This is not research; it's a feasibility study.** Output: a number — what fraction of post-only orders fill within 1-2 bars of placement, given OB depth conditions in the dataset?

**Method**:
1. For every entry signal in the existing `BASELINE_VOTE5` audit log (1,122 trades), simulate placing a post-only order at the best price.
2. Walk forward through the OB parquet — does the price come back to the post-only level within N bars?
3. Compute fill rate per regime, per strategy, per trade direction.

**Decision criterion**: maker-only viable if fill rate ≥ 70% within 2 bars across all regimes. Below that, must accept partial maker / partial taker hybrid.

**Code touchpoints**:
- New `models/maker_fill_simulation.py` — uses `cache/okx_btcusdt_*_ob.parquet` directly.
- Output: per-strategy, per-regime fill-rate table.

**Risks**:
- OB data is sampled at 1-min snapshots — sub-second behavior is invisible.
- Fill simulation assumes price-time priority but real exchanges have queue position effects.

**Cost**: ~2 h script + analysis.

**If it works**: Path F is largely redundant — zero-fee path is real. If maker-only fails this gate, Path F becomes critical.

#### F3 — Architectural fee-aware retrains

**Hypothesis**: The current FEE4_p001 / FEE4_p005 retrains had fee in reward but not in state. Adding fee as a state feature lets the policy learn fee-conditional behavior; one model trained at multiple fee levels generalizes.

**F3.1 — Fee in state**:
- Add `state[N] = fee_bp / 10` to state vector.
- Train with random fee per episode ∈ {0, 2, 4.5, 8} bp.
- Eval at each fee level and check if one model dominates the per-fee-trained models.

**F3.2 — QR-DQN with CVaR action selection**:
- Same as Z4.4 but specifically for fee robustness.
- Hypothesis: tail-aware action selection naturally penalizes high-variance trades that fees can flip.

**F3.3 — Direct net-edge regression**:
- Skip Q-learning; regress `E[pnl − 2×fee]` directly per (state, action).
- Trade only if net-edge prediction > 0.

**Decision criterion**: each F3 sub must beat the F1 winner (vanilla + filter or vote-strength sized) on WF at fee=4.5 bp by ≥+0.5 Sharpe.

**Risks**: 1-2 days each. High implementation cost for uncertain gain. Defer until F1 winners established.

#### F4 — Asymmetric exits by regime

**Hypothesis**: The 5 regime classes have different optimal TP/SL. Current configs are static per strategy. Trending regimes warrant wider TP; chop wants tighter (fewer time-stops, fewer fee-eating slow trades).

**Method**: `tp_scale[regime] × base_tp` lookup. Calibrate scaling factors on train (e.g. `{calm: 1.0, trend_up: 1.3, trend_down: 1.3, ranging: 0.85, chop: 0.7}`).

**Decision criterion**: WF lift ≥+0.5 at fee=4.5 bp.

**Cost**: ~2 h.

**Risk**: regime labels are CUSUM-derived and noisy; per-regime scaling can amplify noise.

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
