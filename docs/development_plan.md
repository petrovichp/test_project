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

## Execution status

| phase | status | notes |
|---|---|---|
| Z1 — Stack proven winners | NOT STARTED | first to run; ~2 days |
| Z2 — Better state | NOT STARTED | depends on Z1 winner as baseline |
| Z3 — Better signals | **PARTIALLY SCOPED** | Z3 plan revised after [feasibility check](z3_data_feasibility.md); compressed from 4 strategies to 2 work items |
| Z4 — Architecture & training | NOT STARTED | each sub-experiment independent of others |
| Z5 — Validation & freeze | gated on Z1–Z4 winners | |
| Path F (non-zero-fee) | PARKED | resume after Path Z winner OR if maker-only fails |

**Suggested first action**: launch Z1.1 (H256 + Double_Dueling stack, 5 seeds, ~2.5 h) — cheapest test combining two independently-validated improvements (capacity from H256 + regularization from DD).

**Parallel cheap action**: Z3.1 standalone validation of S5/S9/S11/S13 — no training required, just backtest the 4 unused strategies to see which carry signal. Could run while Z1.1 trains.

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
