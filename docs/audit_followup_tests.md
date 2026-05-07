# Audit Follow-up Tests — Tests 1-5 from a2_rule_audit.md

Companion to [a2_rule_audit.md](a2_rule_audit.md). The audit surfaced 5 observations marked "worth pursuing" — this doc records the experimental outcome of testing each.

**Baseline**: A2 + rule-based exits, walk-forward across 6 RL folds at fee=0
- **Mean Sharpe +9.034, median +8.865, 6/6 folds positive**
- Per-fold equity 1.07× to 2.23× over ~32-day windows
- Per-fold details: see [data_splits.md](data_splits.md) for fold boundaries

All 5 tests run **independently against this baseline**. Decision criterion per test in plan; outcome decisions below.

## Result table — all tests at a glance

| # | Test | Change | Mean Sharpe | Median | Δ vs baseline | Decision |
|---|---|---|---|---|---|---|
| — | **baseline** | — | **+9.034** | +8.865 | 0 | — |
| 1 | Ablate S6_TwoSignal | mask action 5 | +8.563 | +8.235 | **−0.471** | **drop** |
| 2 | Ablate S10_Squeeze | mask action 8 | +7.597 | +7.171 | **−1.437** | **drop** (severe) |
| 3 | Ablate S7_OIDiverg | mask action 6 | +7.643 | +7.117 | **−1.391** | **drop** (severe) |
| 4a | Tighten TP × 0.85 | trend strats only | +8.073 | +8.119 | **−0.961** | **drop** |
| 4b | Tighten TP × 0.70 | trend strats only | +8.432 | **+9.036** | −0.602 | **drop** (median equal but mean lower) |
| 5 | Long-bias diagnostic | analysis-only | n/a | n/a | **+9.034** unchanged | **diagnostic only — no change** |
| 5b | Synthetic price inversion | stress test | +1.095 (inv) | +1.379 (inv) | n/a (orthogonal) | **diagnostic — see verdict below** |

## Test 1 — Ablate S6_TwoSignal

**Hypothesis**: S6 had val −1.32% / test −2.93% PnL across 8/5 trades. Removing it should be at worst neutral.

**Result** ([cache/btc_groupC2_walkforward_test1_ablate_s6.json](../cache/btc_groupC2_walkforward_test1_ablate_s6.json)):

| Fold | Baseline | With S6 ablated | Δ |
|---|---|---|---|
| 1 | +13.029 | +12.001 | −1.03 |
| 2 | +14.820 | +14.502 | −0.32 |
| 3 | +6.292 | +6.134 | −0.16 |
| 4 | +9.561 | +8.510 | **−1.05** |
| 5 | +8.170 | +7.960 | −0.21 |
| 6 | +2.334 | +2.272 | −0.06 |
| **Mean** | **+9.034** | **+8.563** | **−0.47** |

**Interpretation**: ablation HURTS in 4 of 6 folds even though S6 was negative on val/test alone. Why?

S6 fires on a small subset of bars (~2% bar coverage). When A2 picks S6, it's because S6 is the best valid action *given the state*. Masking S6 forces A2 to pick NO_TRADE or a different strategy — typically a worse alternative. The audit's −1.3% / −2.9% per-strategy attribution measures sum-of-PnL, not opportunity cost. The "underperforming" strategy was filling a niche role.

**Decision**: drop the ablation. Keep S6 in the action mask.

## Test 2 — Ablate S10_Squeeze

**Hypothesis**: S10 flipped sign val→test (+1.97% → −3.09%, 27/15 trades). Marginal.

**Result** ([cache/btc_groupC2_walkforward_test2_ablate_s10.json](../cache/btc_groupC2_walkforward_test2_ablate_s10.json)):

| Fold | Baseline | With S10 ablated | Δ |
|---|---|---|---|
| 1 | +13.029 | +12.930 | −0.10 |
| 2 | +14.820 | +11.363 | **−3.46** |
| 3 | +6.292 | +6.087 | −0.21 |
| 4 | +9.561 | +8.255 | −1.31 |
| 5 | +8.170 | +4.444 | **−3.73** |
| 6 | +2.334 | +2.504 | +0.17 |
| **Mean** | **+9.034** | **+7.597** | **−1.44** |

**Interpretation**: S10 contributes meaningful alpha in folds 2 and 5 — total −7 Sharpe across those two folds when ablated. Only fold 6 (the test split) marginally improves, matching the audit observation that S10 underperformed there. Net heavily negative.

**Decision**: drop the ablation. S10's val/test sign-flip was real but the in-sample folds where it works dominate the aggregate.

## Test 3 — Ablate S7_OIDiverg

**Hypothesis**: S7 flipped sign val→test (+5.87% → −1.73%, 54/45 trades — high frequency, ~25% of all trades).

**Result** ([cache/btc_groupC2_walkforward_test3_ablate_s7.json](../cache/btc_groupC2_walkforward_test3_ablate_s7.json)):

| Fold | Baseline | With S7 ablated | Δ |
|---|---|---|---|
| 1 | +13.029 | +7.441 | **−5.59** |
| 2 | +14.820 | +12.498 | −2.32 |
| 3 | +6.292 | +8.820 | **+2.53** |
| 4 | +9.561 | +6.087 | **−3.47** |
| 5 | +8.170 | +6.793 | −1.38 |
| 6 | +2.334 | +4.220 | **+1.89** |
| **Mean** | **+9.034** | **+7.643** | **−1.39** |

**Interpretation**: most striking heterogeneity of any test. S7 ablation IMPROVES folds 3 and 6 (the two folds with positive BTC return) but devastates fold 1 (−5.59) and fold 4 (−3.47). S7 is a regime-conditional strategy — works in fold-1/2/4-style down/volatile regimes, hurts in fold-3/6-style up regimes. A2 doesn't have visibility into which regime is which, so removing S7 entirely is too crude.

**Decision**: drop the ablation. Possible follow-up (out of scope here): regime-conditional strategy gating where S7 is masked only in detected up-regimes.

## Test 4a/4b — TP tightening for trend strategies

**Hypothesis**: only 3-5% of trades hit TP. Tightening TP for trend strategies (S1, S4, S6, S8, S10) might raise TP-hit rate at modest TSL cost.

**Result**: ([cache/btc_groupC2_walkforward_test4a_tp0.85.json](../cache/btc_groupC2_walkforward_test4a_tp0.85.json), [_test4b_tp0.70.json](../cache/btc_groupC2_walkforward_test4b_tp0.70.json))

| Fold | Baseline | tp × 0.85 | tp × 0.70 |
|---|---|---|---|
| 1 | +13.029 | +11.709 | +10.705 |
| 2 | +14.820 | **+15.203** | +13.343 |
| 3 | +6.292 | +4.133 | +5.922 |
| 4 | +9.561 | +8.568 | **+9.595** |
| 5 | +8.170 | +7.671 | +8.476 |
| 6 | +2.334 | +1.153 | +2.548 |
| **Mean** | **+9.034** | +8.073 (Δ −0.96) | +8.432 (Δ −0.60) |
| **Median** | +8.865 | +8.119 | **+9.036** (Δ +0.17!) |

**Interpretation**: non-monotonic. Mild tightening (×0.85) is worse than aggressive tightening (×0.70), suggesting the relationship between TP threshold and Sharpe isn't smooth. The tp×0.70 variant has a *higher* median than baseline but lower mean — fold 1 and fold 6 take a hit while fold 4 marginally improves.

The reason this is non-monotonic: ATR-scaling already adapts TP per bar. Tightening base_tp_pct compresses *all* trades (including high-ATR trades that need wider TP). The ATR scaling and the static threshold interact.

**Decision**: drop both variants. The current TP thresholds are non-trivially well-tuned despite the low TP-hit rate.

## Test 5 — Long-bias diagnostic + synthetic price inversion

The audit flagged 65% long trades vs 34% short trades as a possible long-bias-dependence warning. Two parts:

### Part A — Per-fold direction analysis (no model changes)

Per-fold table from baseline walk-forward:

| Fold | BTC return | A2 Sharpe | Long PnL | Short PnL | Long n / Short n |
|---|---|---|---|---|---|
| 1 | **−6.43%** | +13.029 | +19.41% | **+35.28%** | 65 / 48 |
| 2 | **−16.88%** | +14.820 | +34.88% | **+46.91%** | 109 / 102 |
| 3 | +6.33% | +6.292 | +17.03% | +2.74% | 90 / 45 |
| 4 | **−29.83%** | +9.561 | +10.37% | **+40.17%** | 159 / 103 |
| 5 | +6.02% | +8.170 | +24.00% | +13.02% | 157 / 82 |
| 6 | +9.11% | +2.334 | +11.62% | −4.47% | 96 / 65 |
| **Total** | mean −5.28% | mean +9.034 | **+117.31%** | **+133.66%** | 676 / 445 |

**Three findings**:

1. **BTC has NEGATIVE mean return across folds** (-5.28% mean, fold 4 was -29.83%). The eval period was not a buy-and-hold-friendly market.

2. **Short PnL aggregate (+133.66%) > long PnL aggregate (+117.31%)** despite fewer short trades. **Per-trade short alpha is larger than per-trade long alpha.**

3. **Correlation A2 Sharpe vs BTC return = −0.632**. A2 makes its largest Sharpe in the most negative BTC folds (folds 1, 2, 4 with returns −6%, −17%, −30%) and its smallest Sharpe in the most positive BTC fold (fold 6 with +9%).

![Per-fold scatter: BTC return vs A2 Sharpe](../cache/test5_btc_vs_sharpe.png)

The 65% long bias of trade COUNT was misleading. **A2 is anti-correlated with BTC moves** — it tends to do *better* when BTC trends down. The audit's long-bias warning was incorrect; the system is in fact closer to defensive than directional.

### Part B — Synthetic price-inversion stress test

Replaces price array with `2 × mean(prices) − prices` (mirror around period mean). State vectors are unchanged — A2's decisions are identical, but the simulated trade outcomes use inverted prices. Tests whether A2's edge survives complete trajectory inversion.

| Fold | BTC orig | BTC inv | Orig Sharpe | Inv Sharpe | Δ |
|---|---|---|---|---|---|
| 1 | −6.43% | +12.72% | +13.029 | +3.050 | **−9.98** |
| 2 | −16.88% | +27.65% | +14.820 | −1.761 | **−16.58** |
| 3 | +6.33% | −6.75% | +6.292 | +0.129 | −6.16 |
| 4 | −29.83% | +36.25% | +9.561 | +2.629 | −6.93 |
| 5 | +6.02% | −3.76% | +8.170 | +4.086 | −4.08 |
| 6 | +9.11% | −6.28% | +2.334 | −1.564 | −3.90 |
| **Mean** | −5.28% | +9.97% | **+9.034** | **+1.095** | **−7.94** |

**Interpretation**: A2 stays positive on inverted data (4/6 folds positive, mean +1.10 Sharpe). But it loses ~8 Sharpe of edge. The fact that it doesn't COLLAPSE confirms partial direction-symmetry; the fact that it loses substantial edge confirms the entry signals (state vector inputs) carry direction-specific information that's broken when prices move opposite to the state's expectations.

This is expected — the strategies use signed signals (e.g., direction model probabilities, momentum indicators) that depend on price-path orientation. Inverting only the prices and not the state vectors creates an artificial "wrong direction signal" scenario that A2 can't cleanly handle. The real-world equivalent would be a regime where the predictive features stop mapping correctly to future returns.

### Test 5 verdict

✓ **A2 is direction-symmetric on aggregate** (anti-correlated with BTC, short alpha exceeds long alpha)
◐ **A2 has direction-conditional state mapping** — drops 8 Sharpe under synthetic inversion. Real-world risk: regime where features stop predicting forward returns.

No model change is justified by Test 5. The "65% long bias" observation was a false alarm of long-trend dependence.

## Cumulative analysis — what we learned

| Test | Outcome | Key insight |
|---|---|---|
| 1 (S6 ablation) | drop | A2 uses S6 in a niche role; per-strategy negative PnL doesn't justify removal |
| 2 (S10 ablation) | drop | S10 is critical to folds 2 and 5; val/test sign-flip is regime, not signal |
| 3 (S7 ablation) | drop | S7 is regime-conditional; helpful in down/volatile, hurtful in up — but A2 needs both |
| 4a/4b (TP tightening) | drop | ATR-scaling and base_tp interaction is non-monotonic; current tuning is near-optimal |
| 5 (long-bias) | diagnostic only | A2 is anti-correlated with BTC; not long-bias-dependent. Direction-symmetric in aggregate. |

**No proposed change improves the baseline. The baseline is at a (local) optimum of the search space we explored.**

The audit's per-strategy attributions were *descriptive* (sum of trade PnL) but not *prescriptive* (don't tell us what to ablate). A2's policy has learned compositional dependencies among strategies that aren't visible at per-strategy granularity.

User asked: "firstly independent, then we make analysis and decide how better to stack improvements." Since **no individual test won, no stacking is justified**. The deployment target remains:

> **A2 entry DQN + rule-based exits with original `EXECUTION_CONFIG`** — walk-forward mean Sharpe +9.034, 6/6 folds positive, anti-correlated with BTC moves (counter-trend tilted).

## Implications for production

1. **Don't tune individual strategy weights** — A2 has learned the aggregation already. Per-strategy losses can be subsidized by per-strategy gains in different regimes.

2. **A2 is defensive against BTC drawdowns** — the −0.63 correlation with BTC return is a feature, not a bug. In a sustained bull market with low volatility (like fold 6 conditions), A2's Sharpe would compress; in volatile or down regimes, A2 would shine.

3. **The system needs ALL 8 active strategies** (S12 has 0 trades anyway). Each carries weight in some regime. Don't simplify the action space.

4. **TP thresholds are stable** — small perturbations don't translate to monotone Sharpe changes due to ATR-scaling interaction. Don't grid-search this in production.

5. **Future research direction** (out of scope here): regime-conditional strategy gating (e.g., mask S7 only in detected up-regimes). Group A's `regime_id` state input could be used to re-train an A3-style policy that learns regime-conditional masks. Worth ~1 day of work if alpha-bound after deployment.

## Test 6 — Ablate S6 / S7 / S10 with **fresh A2 retraining** (added 2026-05-07)

The 5 tests above all applied the action mask **only at evaluation time** on the original A2 policy. That leaves the question: *would a fresh A2 trained from scratch with the strategy disabled learn to reallocate to other strategies and recover the lost Sharpe?*

Three new policies trained at the same hyperparameters as A2 (fee=0, trade_penalty=0.001, seed=42, action_mode=all but with `--ablate-actions` plumbed through both rollouts AND validation):

| Variant | Train-val best Sharpe | Walk-forward mean Sharpe | Δ vs baseline | Eval-only Δ (for comparison) |
|---|---|---|---|---|
| **A2 baseline** | +7.30 | **+9.034** | 0 | 0 |
| A2_no_s6 (retrain) | +5.19 | +7.398 | **−1.636** | −0.471 |
| A2_no_s7 (retrain) | +4.37 | +6.994 | **−2.040** | −1.391 |
| A2_no_s10 (retrain) | +2.86 | +5.860 | **−3.174** | −1.437 |

**Per-fold rule Sharpe (retrain runs):**

| Variant | F1 | F2 | F3 | F4 | F5 | F6 |
|---|---|---|---|---|---|---|
| baseline | 13.03 | 14.82 | 6.29 | 9.56 | 8.17 | 2.33 |
| no_s6  retrain | 9.67 | 11.95 | 5.74 | 8.59 | 5.24 | 3.20 |
| no_s7  retrain | 7.58 | 8.79 | 11.07 | 6.58 | 5.09 | 2.85 |
| no_s10 retrain | 11.59 | 12.93 | 5.54 | 1.49 | 2.72 | 0.90 |

**Outcome — retraining is strictly worse than eval-only masking** in all 3 cases. The hypothesis that "the DQN will reallocate" is falsified.

**Why this is the result**:
- *Eval-only* masking preserves A2's learned Q-function; only the bars where A2 picks the masked action lose value (they get NO_TRADE substituted).
- *Retraining* reshapes the entire policy. The DQN sees a smaller action set during training, which means it sees fewer reward signals and the value estimates for ALL state-action pairs shift. The training-val Sharpe drop (+7.30 → +2.86 for no_s10) shows the *training problem itself* gets harder when the strategy is removed.
- The masked strategies aren't redundant — they carry specialized signal that A2's policy depends on for context, even when their per-trade contribution looks marginal.

**This is the most direct possible test of the audit's per-strategy attribution claim** — and it confirms: removing a strategy degrades A2 even when given full opportunity to retrain around the gap. The audit's "tests worth pursuing" framing was wrong; per-strategy summed PnL is purely descriptive.

**Decision**: drop all three retrained variants. Final verdict on the audit's surfaced issues stands at "all 5 + 3 retraining variants degrade baseline; no winning variant found."

### Test 6b — Triple ablation (S6 + S7 + S10 simultaneously)

Hypothesis: maybe the three "negative" strategies are jointly redundant, and removing all three together would let the DQN settle into a cleaner policy on the remaining 5 strategies (S1, S2, S3, S4, S8).

```
python3 -m models.dqn_selector btc --tag A2_no_s6_s7_s10 \
        --fee 0.0 --trade-penalty 0.001 --ablate-actions 5,6,8
python3 -m models.group_c2_walkforward --policy-tag A2_no_s6_s7_s10 \
        --ablate-actions 5,6,8 --no-b5 --out-tag retrain_no_s6_s7_s10
```

| Metric | Baseline | Triple ablation | Δ |
|---|---|---|---|
| Train val Sharpe | +7.30 | +4.66 | −2.64 |
| WF mean Sharpe | +9.034 | +6.756 | **−2.28** |
| WF median Sharpe | +9.561 | +8.495 | −1.07 |
| WF folds positive | 6/6 | 6/6 | — |

**Per-fold:**

| Fold | Baseline | Triple ablation | Δ |
|---|---|---|---|
| 1 | +13.03 | +6.36 | −6.67 |
| 2 | +14.82 | +9.24 | −5.58 |
| 3 | +6.29 | +8.49 | **+2.20** |
| 4 | +9.56 | +9.28 | −0.28 |
| 5 | +8.17 | +4.07 | −4.10 |
| 6 | +2.33 | +3.09 | **+0.76** |

**Observations**:
- Triple ablation (Δ −2.28) is *less bad* than single S10 ablation (Δ −3.17). The strategies have overlapping signal — when only S10 is removed, A2's Q-estimates have a "hole" that's filled poorly; when all three are removed, the policy reorganizes more coherently around the remaining 5.
- Trade count rose to 123–247 trades/fold (vs baseline ~150) — A2 trades more on the remaining strategies.
- Win rate rose (54–62% vs baseline 50–65%) but Sharpe fell — smaller but more frequent wins, lower per-trade alpha.
- Folds 3 and 6 improved (+2.20 and +0.76). Folds 1, 2, 5 collapsed (−6.67, −5.58, −4.10).

**Verdict**: still worse than baseline. The signal-overlap among the three ablated strategies is real but not large enough to prevent aggregate degradation. The pattern of improvement on folds 3 and 6 suggests there *might* be regimes where these strategies hurt — but identifying those regimes a priori would require regime-conditional gating, which is out of scope for this experiment series.

Final tally: **6 audit hypotheses × 9 variants tested → 0 winners.**

## Files / artefacts

| File | Contents |
|---|---|
| [models/group_c2_walkforward.py](../models/group_c2_walkforward.py) | extended with `--ablate-actions`, `--tp-scale`, `--out-tag`, `--no-b5` flags |
| [models/dqn_selector.py](../models/dqn_selector.py) | `evaluate_policy` extended to also return per-trade `trade_dirs`, `trade_strats`; eq_arr last-bar fix |
| [models/test5_long_bias.py](../models/test5_long_bias.py) | Test 5 module with two parts (diagnostic + synthetic inversion) |
| `cache/btc_groupC2_walkforward_verify_baseline.json` | baseline reproduction (mean +9.034) |
| `cache/btc_groupC2_walkforward_test{1..3}_ablate_*.json` | ablation tests |
| `cache/btc_groupC2_walkforward_test4{a,b}_tp*.json` | TP tightening tests |
| `cache/test5_inversion_results.json` | synthetic inversion per-fold details |
| `cache/test5_btc_vs_sharpe.png` | scatter plot of BTC return vs A2 Sharpe |
| `cache/btc_dqn_policy_A2_no_{s6,s7,s10}.pt` | retrained ablation policies (Test 6) |
| `cache/btc_dqn_train_history_A2_no_{s6,s7,s10}.json` | retraining histories |
| `cache/btc_groupC2_walkforward_retrain_no_{s6,s7,s10}.json` | retrain walk-forward results |
| `cache/btc_dqn_policy_A2_no_s6_s7_s10.pt` | triple-ablation policy (Test 6b) |
| `cache/btc_groupC2_walkforward_retrain_no_s6_s7_s10.json` | triple-ablation walk-forward result |
