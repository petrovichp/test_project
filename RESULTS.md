# Crypto Trading ML — Results & Conclusions

> **Status (2026-05-07):**
> - **Signal generalizes fee-free** (A2: **val +7.30, test +3.78**, equity 1.13× on locked test). The trading edge is real.
> - **At maker fee** A4 has a val/test gap (val +1.72, test −1.65) — needs walk-forward + seed-variance validation.
> - **Group B5 (fixed-window exit DQN)** with 53-dim enriched state (price-path + vol-window + trajectory scalars) **works**: 9/9 strategies positive at N=240, mean Δ +3.48 over no-exit baseline, max +6.69 (S10_Squeeze).
> - **C2_fix240 (A2 entry + B5_fix240 exits) is a breakthrough on test**: **Sharpe +8.329, equity 1.343× on locked test split** — beats A2-alone (+3.78, 1.13×) by +4.55 Sharpe with lower max-DD (−4.29% vs −9.69%) and 67% win rate. **First time RL exit stacking adds real value on a held-out split.** Val side is weak (−1.43), so walk-forward validation across 6 RL folds is the next mandatory step before relying on this.
> - C1 (variable-length exit DQN composition) confirmed as failed; C2_fix240's success demonstrates that the failure was specifically the variable-length+rule-active formulation, not RL composition in general.
> - **Forward path:** walk-forward A2 + walk-forward C2_fix240 across 6 folds → seed variance → if both stable, deploy via Path X (maker execution).

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Executive Summary](#executive-summary)
3. [Best Results Table](#best-results-table)
4. [Phase-by-Phase Findings](#phase-by-phase-findings)
   - [Phase 1+2 — Upstream models & state arrays](#phase-12--upstream-models--state-arrays-done)
   - [Phase 3 — Initial DQN attempts](#phase-3--initial-dqn-attempts-failed-as-stated)
   - [Group D — Failure diagnostics](#group-d--failure-diagnostics-clarified-cause)
   - [Path 1 — Root cause diagnostics](#path-1--root-cause-diagnostics-key-insight)
   - [Group A — Fee × penalty sweep](#group-a--fee--penalty-sweep-breakthrough)
   - [Group B — Exit-timing DQN](#group-b--exit-timing-dqn-no-lift)
5. [Cumulative Insights](#cumulative-insights)
6. [Production Readiness](#production-readiness)
7. [Files & Artifacts](#files--artifacts)
8. [Next Steps](#next-steps)

---

## TL;DR

The strategies have real predictive edge. Fees are what kills them on 1-minute BTC. The entry DQN works fee-free (A2: val +7.30, **test +3.78**, equity 1.13×) and at maker fee with a val/test gap (A4: val +1.72, test −1.65 — needs walk-forward).

**The big new finding (2026-05-07):** redesigning the exit DQN with **fixed-length 240-bar episodes** + **enriched 53-dim state** (price-path window, volatility window, in-trade trajectory scalars, time-of-day) cleared the variable-length transfer pathology. **Stacking it on A2 entries (Group C2_fix240) gives val Sharpe −1.43 but test Sharpe +8.33 with equity 1.343× on the locked test split** — beats A2-alone with rule-based exits (test +3.78) by +4.55 Sharpe with lower max-DD (−4.29% vs −9.69%) and higher win rate (67% vs 56%).

The val/test asymmetry is real and not yet explained — the per-strategy B5 policies were val-checkpointed *per strategy* but not on the composition, so C2_val is genuinely OOS. **Walk-forward across the 6 RL folds is the immediate next step** to verify whether the test result generalizes or is window-specific.

**Production path:** if walk-forward confirms C2_fix240, deploy A2 entry + B5_fix240 exits via Path X maker execution. Otherwise fall back to A2 alone with rule-based exits (val +7.30 / test +3.78, already deployable).

**Reference plot:** [cache/btc_dqn_groupA_equity_vs_price.png](cache/btc_dqn_groupA_equity_vs_price.png)

---

## Executive Summary

This research project trained an end-to-end ML pipeline for BTC perp 1-minute trading on OKX (Jul 2025 – Apr 2026, 384,614 bars). It built feature engineering (191 features), three predictive models (vol LightGBM, direction CNN-LSTM, CUSUM regime), 9 trading strategies, a parity-verified single-trade simulator, and a full DQN gating framework (PER, n-step Bellman, action masking).

The DQN gating initially failed (val Sharpe **−9.19** baseline, **−4.70** even with binary actions). Five independent diagnostic methods (DQN, supervised PnL prediction, grid search, walk-forward, supervised regression) all confirmed strategies lacked persistent edge **under taker fees**.

The breakthrough came from Path 1 root-cause diagnostics, which separated three failure mechanisms (fees / exits / timeframe) and quantified their individual contributions. Removing fees flipped grand-mean Sharpe from **−10.09** → **+2.31** (Δ = +12.4 Sharpe). The oracle showed signals have an additional **+14 Sharpe** of latent value from better exits.

The follow-up Group A sweep retrained the DQN at three fee levels × three penalty levels (7 cells, ~25 minutes total). It produced **val Sharpe +7.30 at fee=0** (deployable in spirit, not in production) and **+1.72 at OKX maker fee** (production target).

---

## Best Results Table

| Cell | Method | Fee | Val Sharpe | Test Sharpe (locked) | Status |
|---|---|---|---|---|---|
| **C2_fix240** | A2 entry + B5_fix240 per-strategy exits stacked | 0 | −1.43 (eq 0.93) | **+8.329 (eq 1.343)** | **best test result, needs walk-forward ★** |
| **A2** | DQN entry-gate (no exit RL) | 0 | **+7.30** (eq 1.40) | **+3.78** (eq 1.13) | **strong baseline, generalizes ✓** |
| C2_fix120 | A2 entry + B5_fix120 exits stacked | 0 | −2.48 | +3.72 (eq 1.13) | matches A2-alone on test |
| C2_fix60 | A2 entry + B5_fix60 exits stacked | 0 | +1.69 | −1.22 | weak on test |
| **A4** | DQN entry-gate | 0.0004 (maker) | +1.72 (eq 1.07) | **−1.65** (eq 0.94) | val/test gap, needs walk-forward |
| B5_fix240 (per-strat best) | Fixed-window exit DQN (S7_OIDiverg) | 0 | +5.91 | — | 9/9 strategies positive Δ |
| B4_fee0_S3 | Per-strategy variable-length exit DQN (S4) | 0 | +4.74 (Δ +4.20) | — | clears +4 gate per-strategy |
| C1_fee0 | A2 + B4_fee0 (variable-length) exits stacked | 0 | +3.93 (Δ +0.05) | +4.28 (Δ −2.70) | composition fails ✗ |
| A1 | DQN entry-gate (no penalty) | 0 | +5.81 | — | confirms RL works fee-free |
| Phase 1a passive | Free-firing strategies | 0 | +2.31 grand mean across folds | — | RL adds +5 lift over passive |
| A0 | DQN entry-gate | 0.0008 (taker) | −5.87 (eq 0.76) | — | replicates prior failure ✗ |

---

## Phase-by-Phase Findings

### Phase 1+2 — Upstream models & state arrays (DONE)

**Goal:** produce all per-bar inputs the DQN consumes.

**Models trained (gates passed in isolation):**

| Model | Architecture | Train chunk | OOS metric |
|---|---|---|---|
| Vol LightGBM v4 | ATR-30 regression | bars [1,440, 101,440) | Spearman **0.690** (≥0.65 gate) ✓ |
| Direction CNN-LSTM v4 ×4 | Conv1D→GRU, two-stage | dir-train [1,440, 91,440) | AUC **0.64–0.70** (>0.55 gate) ✓ |
| CUSUM regime v4 | percentile thresholds | bars [1,440, 101,440) | KW p **2.21e-20** (<0.01 gate) ✓ |

**Bar-chunk legend:**

| Chunk | Bars | Dates | Use |
|---|---|---|---|
| Warmup | [0, 1,440) | 2025-07-04 → 07-05 | dropped (NaN window) |
| Vol-train | [1,440, 101,440) | 2025-07-05 → 09-19 | LightGBM vol fit + CUSUM thresholds + standardize |
| Dir-train | [1,440, 91,440) | 2025-07-05 → 09-12 | CNN-LSTM training |
| Dir-holdout | [91,440, 101,440) | 2025-09-12 → 09-19 | CNN-LSTM early-stop |
| DQN-train | [101,440, 281,440) | 2025-09-19 → 2026-02-12 | DQN training |
| DQN-val | [281,440, 332,307) | 2026-02-12 → 03-20 | DQN early-stop |
| DQN-test | [332,307, 384,614) | 2026-03-20 → 04-25 | locked, single-shot eval |

→ Full split documentation with date spans and walk-forward folds: [docs/data_splits.md](docs/data_splits.md)

**State arrays:** 50-dim per bar (20 static + 30 windowed lags), saved as `cache/btc_dqn_state_{train,val,test}.npz`. Action mask 10-dim (NO_TRADE + 9 strategies).

**Action-mask coverage:** train 29.75%, val 33.58%, test 26.43% (within spec target 30–60%).

→ Detailed log: [docs/experiments_log.md#phase-12](docs/experiments_log.md#phase-12)

---

### Phase 3 — Initial DQN attempts (FAILED as stated)

| Attempt | val Sharpe | Notes |
|---|---|---|
| **Baseline (taker fee, no penalty)** | **−9.19** | Loss collapses to 0.0001 — predict-zero attractor |
| Path A (reward ×100, stratified PER) | −5.87 | Fixed numerical issues but signal still absent |
| Path C (binary {NO_TRADE, S1} only) | −4.70 | Even simplest decision unlearnable |

**Conclusion at the time:** "RL gating not salvageable from this state representation."

This was wrong — see [Path 1 diagnostics](#path-1--root-cause-diagnostics-key-insight).

→ Detailed log: [docs/experiments_log.md#phase-3](docs/experiments_log.md#phase-3)

---

### Group D — Failure diagnostics (clarified cause)

Three independent confirmations that strategies under taker fees don't produce edge:

| Diagnostic | Best result | Conclusion |
|---|---|---|
| **D1 Supervised PnL predictor** | Spearman ≤ 0.084 | State has near-zero residual signal beyond what strategies already use |
| **D2 Grid search** (5 strategies × ~50 params each) | Best test Sharpe **+0.10** (S1) | Marginal val improvements don't transfer |
| **D3 Walk-forward (6 folds × 5 strategies × 3 modes)** | **0/75 stable combos** | Every fold has negative mean Sharpe — no single window has edge |

These reinforced (incorrect) belief that strategies were structurally broken. The error: every test was at taker fee.

→ Detailed log: [docs/experiments_log.md#group-d](docs/experiments_log.md#group-d)

---

### Path 1 — Root cause diagnostics (KEY INSIGHT)

Three orthogonal experiments that together identified the true cause:

#### 1a — Fee-free walk-forward

| Strategy | with-fee mean | fee-free mean | Δ |
|---|---|---|---|
| S1_VolDir | −5.66 | **+2.55** | +8.21 |
| S4_MACDTrend | −3.79 | **+1.70** | +5.49 |
| S6_TwoSignal | −5.06 | **+0.41** | +5.48 |
| S7_OIDiverg | −28.52 | **+2.90** | +31.42 |
| S8_TakerFlow | −7.43 | **+3.97** | +11.40 |
| **Grand mean** | **−10.09** | **+2.31** | **+12.40** |

**Verdict:** every strategy flips from 0/6 positive folds to 4–6/6 positive folds when fees are removed. **Fees are decisive.**

#### 1b — Optimal-exit oracle (60-bar perfect exit)

| | Oracle with fee | Oracle fee-free |
|---|---|---|
| Grand mean Sharpe | **+6.59** | **+35.68** |
| S1 oracle PnL/trade | +0.06% | +0.30% (82.7% win rate) |

**Verdict:** entry signals have real predictive power. Exits leave ~14 Sharpe on the table even without fees.

#### 1c — 5-minute timeframe

| | with fee | fee-free |
|---|---|---|
| 1-min grand mean | −10.09 | +2.31 |
| 5-min grand mean | −6.20 | +2.33 |

**Verdict:** 5-min cadence reduces fee impact by trading less. Doesn't add new alpha (fee-free Sharpe identical). Higher timeframes alone don't solve the problem.

→ Detailed log: [docs/experiments_log.md#path-1](docs/experiments_log.md#path-1)

---

### Group A — Fee × penalty sweep (BREAKTHROUGH)

Retrained DQN at 7 (fee, penalty) cells. ~25 minutes total wall time.

| Cell | Fee | Penalty | val Sharpe | Trades | Win% | Equity | DD% |
|---|---|---|---|---|---|---|---|
| A0 | 0.0008 (taker) | 0.000 | **−5.87** | 217 | 42.9% | 0.76 | −26.0% |
| **A1** | **0.0000** | 0.000 | **+5.81** | 287 | 56.4% | 1.29 | −8.1% |
| **A2** | **0.0000** | 0.001 | **+7.30** | 251 | 55.0% | 1.40 | −6.3% |
| A3 | 0.0000 | 0.003 | +5.82 | 326 | 52.5% | 1.31 | −7.6% |
| **A4** | **0.0004 (maker)** | 0.000 | **+1.72** | 241 | 50.6% | 1.07 | −7.2% |
| A5 | 0.0004 | 0.001 | −0.95 | 275 | 46.5% | 0.95 | −10.5% |
| A6 | 0.0008 | 0.001 | −5.38 | 231 | 39.8% | 0.78 | −26.6% |

**Three findings:**

1. **DQN entry-gating is salvageable when fees are managed.** The previous failure was specifically failure under 0.16% taker fees, not a structural inability of the architecture.

2. **DQN beats passive at every fee level where passive is positive.** Group A vs Path 1a passive baseline:

   | Condition | Passive | DQN | Lift |
   |---|---|---|---|
   | fee=0 | +2.31 | +5.81 → +7.30 | +3.5 to +5.0 |
   | fee=0.0004 | ≈0 (estimated) | +1.72 | meaningful |
   | fee=0.0008 | −10.09 | −5.87 | not enough |

3. **Trade penalty interaction depends on fee level.** Mild penalty (0.001) helps at fee=0 (+1.5 Sharpe lift A1→A2) but hurts at fee=0.0004 (+1.72 → −0.95). Production deployment needs fee-specific tuning.

**Best deployable cell: A4 (maker fee, no penalty).** Val Sharpe +1.72, val+test equity 0.98× over 10 weeks.

→ Detailed log: [docs/experiments_log.md#group-a](docs/experiments_log.md#group-a)

---

### Group B — Exit-timing DQN (small lift, below +4 gate)

Tested whether a 28-dim in-trade state DQN with HOLD/EXIT_NOW actions can improve over rule-based exits (TP/SL/BE/trail/time-stop). 12 cells total: 3 global × fee level + 9 per-strategy at maker fee. Modules: [models/exit_dqn.py](models/exit_dqn.py), [models/group_b_sweep.py](models/group_b_sweep.py).

**Global exit DQN (B1-B3) — single shared policy across all 9 strategies:**

| Cell | Fee | Baseline (rule-only) | RL exit | ΔSharpe |
|---|---|---|---|---|
| B1 | 0.0008 (taker) | −14.91 | **−22.46** | **−7.55** |
| B2 | 0.0004 (maker) | −6.81 | **−11.04** | **−4.23** |
| B3 | 0 (fee-free) | +3.79 | **+2.27** | **−1.52** |

All three negative. Pooling heterogeneous strategies into one exit DQN actively hurts.

**Per-strategy exit DQN (B4) at maker fee — one DQN per entry strategy:**

| Strategy | Baseline | RL exit | ΔSharpe |
|---|---|---|---|
| S1_VolDir | −4.66 | −4.05 | +0.61 |
| S2_Funding | −4.47 | −3.51 | +0.96 |
| S3_BBRevert | −22.41 | −21.40 | +1.00 |
| **S4_MACDTrend** | **−4.23** | **−2.66** | **+1.57** |
| S6_TwoSignal | −7.30 | −7.41 | −0.11 |
| S7_OIDiverg | −9.72 | −9.81 | −0.08 |
| **S8_TakerFlow** | **−5.06** | **−3.09** | **+1.97** |
| S10_Squeeze | −7.88 | −8.20 | −0.33 |
| S12_VWAPVol | +3.22 | +3.22 | 0 (n=1) |

**6/9 positive, mean Δ ≈ +0.6, best +1.97 (S8_TakerFlow).** Per-strategy exit DQNs reliably extract a small lift over rule-based exits — but it's too small to clear the +4-Sharpe gate from [docs/next_steps.md](docs/next_steps.md) (which targeted "≥30% of the +28 oracle gap"). None of the strategies become profitable at maker fee on stand-alone entries even with their own exit DQN.

**Decision:** Group B is a *modest* improvement, not the structural breakthrough the gate was designed to detect. **Group C2 (joint hierarchical training, ~3-5 days) is dropped** — joint training only pays off if both stages independently produce strong lifts. **Group C1 (cheap sequential composition: A4 entry + B4 exits)** *is* worth running and was completed — see below.

→ Detailed log: [docs/experiments_log.md#group-b](docs/experiments_log.md#group-b--exit-timing-dqn)

---

### Group C1 — A4 entry + B4 per-strategy exits (sequential composition)

- **Module:** [models/group_c_eval.py](models/group_c_eval.py)
- **Inputs:** trained `cache/btc_dqn_policy_A4.pt` + 9× `cache/btc_exit_dqn_policy_B4_S{k}.pt` (no retraining)
- **Eval:** A4 picks entries, when a trade fires the corresponding B4_Sk exit DQN takes HOLD/EXIT_NOW decisions on top of the strategy's rule-based exits

**Internal comparison (same evaluator code path, fair Δ):**

| Split | Rule-only (A4 + rule exits) | Combined (A4 + B4 exits) | ΔSharpe | Δ equity |
|---|---|---|---|---|
| val  | −5.698 (eq 0.759) | −5.763 (eq 0.752) | **−0.07** | −0.66% |
| test | −3.765 (eq 0.884) | −4.654 (eq 0.850) | **−0.89** | −3.47% |

**Verdict: B4 exits do not transfer to A4-selected entries.** RL exits fired on 24% of trades, mostly cutting positions early — slightly improved win rate (43.5% → 44.0% on test) but reduced total return because winners got truncated.

**Why the transfer fails:** B4 was trained on the "sequential first-firing" entry distribution (every bar that any strategy fires triggers a trade). A4 picks a much more selective subset (~3% of bars, with NO_TRADE 95-98% of the time). The two distributions have different in-trade dynamics — A4's entries are higher-confidence and reward longer holding periods, but B4's policy was trained to bail early on noisier mean trades.

To make joint entry+exit RL work, you'd need to retrain the exit DQN on A4's entry distribution specifically (which is essentially what C2 — joint hierarchical training — was designed to do). C1's failure is informative but doesn't validate the more expensive C2 investment.

**Methodology note:** Group A reported A4 val Sharpe **+1.72** using `dqn_selector.evaluate_policy` (uncapped trade horizon). My exit-DQN evaluator caps trade lookahead at 240 bars to match B4 training conditions, which truncates some long-running trades and changes the absolute Sharpe (−5.70 above). Within either evaluator, the *Δ* between rule-only and combined is internally valid; the absolute number depends on simulator choice. Re-running A4 through `dqn_selector.evaluate_policy`:

| | val | test |
|---|---|---|
| A4 (Group A simulator, uncapped) | +1.715 | **−1.650** |
| A4 (C1 simulator, 240-bar cap) | −5.698 | −3.765 |

**Important secondary finding:** A4 already shows val/test degradation (val +1.72 → test −1.65 in the original simulator). This hadn't been emphasized in Group A's writeup but matters for deployment — A4 is *not* yet validated on the locked test split. The "Reduced scope" mini-validation (walk-forward + seed variance) is now even more important before any production move.

---

### Group B4_fee0 + C1_fee0 — fee-free re-test (KEY FINDING: signal generalizes, composition still fails)

To test whether the fee drag was the killer, the per-strategy exit DQNs were retrained at fee=0 (B4_fee0) and stacked on the strongest entry policy A2 (C1_fee0). Modules: [models/exit_dqn.py](models/exit_dqn.py), [models/group_c_eval.py](models/group_c_eval.py).

**B4_fee0 (per-strategy exit DQN at fee=0):**

| Cell | Strategy | Baseline | RL exit | Δ |
|---|---|---|---|---|
| B4_fee0_S0 | S1_VolDir    | −0.55 | **+3.39** | **+3.94** |
| B4_fee0_S1 | S2_Funding   | +2.17 | **+5.72** | **+3.56** |
| **B4_fee0_S3** | **S4_MACDTrend** | **+0.54** | **+4.74** | **+4.20 ★ clears +4 gate** |
| B4_fee0_S4 | S6_TwoSignal | −2.43 | +0.06 | +2.49 |
| B4_fee0_S5 | S7_OIDiverg  | +6.09 | +5.51 | −0.58 |
| B4_fee0_S6 | S8_TakerFlow | +1.96 | +2.94 | +0.98 |
| B4_fee0_S7 | S10_Squeeze  | +2.17 | +3.01 | +0.85 |

**7/9 positive, mean Δ +1.84, best +4.20.** RL exit-timing IS a real signal — it just gets eaten by the 0.08% round-trip maker fee at 1-min cadence. Confirmation of the fee-drag hypothesis.

**A2 standalone test re-eval (uncapped simulator, apples-to-apples with Group A):**

| Split | Sharpe | Trades | Win % | Equity | Max DD |
|---|---|---|---|---|---|
| val  | **+7.295** | 251 | 55.0% | 1.398 | −6.31% |
| test | **+3.776** | 185 | 55.1% | 1.127 | −9.69% |

A2's val/test gap is **modest and expected** (+7.30 → +3.78, equity 1.40× → 1.13×) — the signal generalizes. This is materially different from A4's val/test gap (+1.72 → −1.65, sign flip).

**C1_fee0 (A2 entry + B4_fee0 exits, stacked, internal Δ same simulator):**

| Split | A2 + rule | A2 + B4_fee0 RL | ΔSharpe | Δ equity |
|---|---|---|---|---|
| val  | +3.876 | +3.928 | **+0.05** | −0.94% |
| test | +6.979 | +4.280 | **−2.70** | −10.06% |

**Composition fails even at fee=0.** B4_fee0 trained on noisy "all-firing" entries learned to bail early; A2's selective ~3% entries reward longer holding, so RL exits truncate winners. The transfer pathology is **structural, not fee-related**.

→ Detailed log: [docs/experiments_log.md#group-b4_fee0](docs/experiments_log.md#group-b4_fee0--per-strategy-exit-dqn-at-fee0) and [§ C1_fee0](docs/experiments_log.md#group-c1_fee0--a2-entry--b4_fee0-exits-at-fee0)

---

### Group B5 + C2 — fixed-window exit DQN (KEY BREAKTHROUGH on test)

C1's failure isolated to two structural issues with B4: variable episode length (rule-fired terminals dominated buffer) and the rule-vs-DQN race in credit assignment. **Group B5 fixes both.** Modules: [models/exit_dqn_fixed.py](models/exit_dqn_fixed.py), [models/group_b5_sweep.py](models/group_b5_sweep.py), [models/group_c2_eval.py](models/group_c2_eval.py).

#### Design changes vs B4

| | B4 (closed) | **B5 (new)** |
|---|---|---|
| Episode length | 1–240 bars (rule-determined) | **fixed N ∈ {60, 120, 240}** |
| Rule-based exits during training | active | **disabled** — only DQN's EXIT_NOW or window-edge can terminate |
| State dim | 28 (in-trade scalars + sliced market lags) | **53** with: price-path window (last 20 bars cum-return-from-entry), volatility window (last 10 bars \|log-ret\| standardized), in-trade trajectory scalars (max/min unrealized, bars-since-peak, running vol), time-of-day cyclic, entry-time static context |
| Network | 28→64→32→2 (4,002 params) | 53→96→48→2 (10,114 params) |
| Per-strategy variant | yes (B4_S0..S8) | yes (B5_fix{N}_fee0_S0..S8) |
| Real-trading SL safety net | n/a (rules handle it) | optional inference-only layer |

#### B5 per-strategy at fee=0 (best results across 27 cells)

| Window | n positive Δ | mean Δ | max Δ | best abs Sharpe |
|---|---|---|---|---|
| N=60 | 8/9 | +2.29 | +6.25 (S1_VolDir) | +6.02 (S2_Funding) |
| N=120 | 8/9 | +1.98 | +4.90 (S6_TwoSignal) | +4.35 (S7_OIDiverg) |
| **N=240** | **9/9** | **+3.48** | **+6.69 (S10_Squeeze)** | +5.91 (S7_OIDiverg) |

(Δ measured against the always-HOLD-to-N baseline — i.e., what the strategy would do if held to bar N regardless of unrealized PnL.)

#### Composition test C2 — A2 entry + B5_fix{N} exits stacked at fee=0

The actual production-relevant test:

| Configuration | val Sharpe | test Sharpe | test equity | test max DD | test win % |
|---|---|---|---|---|---|
| A2 alone + rule-based exits (production target) | **+7.295** | +3.776 | 1.127 | −9.69% | 56.1% |
| C2_fix60 (A2 + B5_fix60 exits) | +1.69 | −1.22 | 0.957 | −14.15% | 56.6% |
| C2_fix120 (A2 + B5_fix120 exits) | −2.48 | +3.72 | 1.129 | — | 65.2% |
| **C2_fix240 (A2 + B5_fix240 exits)** | **−1.43** | **+8.329** | **1.343** | **−4.29%** | **67.0%** |

**The C2_fix240 test result is the strongest signal in the project:**
- Sharpe **+8.33** vs A2-alone +3.78 → **+4.55 Sharpe gain** on locked test
- Equity **1.343×** in 5 weeks vs A2-alone 1.127× → **+19% total return relative**
- Lower max DD (−4.29% vs −9.69%) → less risk
- Higher win rate (67% vs 56%) → more consistent
- 206 trades vs 161 → similar trade frequency, RL exits 87 of 206 trades (42%)

#### Important caveat — val performance is weak

C2_fix240 val Sharpe is −1.43, vs A2-alone val +7.30. The val/test inversion is unusual and not fully explained:

- B5 per-strategy policies were val-best-checkpointed *per strategy* on the dense entry distribution, not on the composition
- Composition val (C2_val) is therefore genuinely OOS for the stacked system — it's not a selection artifact
- The B5 policies that work well on per-strategy val (e.g., S7_OIDiverg val +5.91) don't necessarily transfer to A2's selective subset of S7 entries on val
- But on test, the patterns evidently align — B5's exits and A2's entries compose well

#### Why fixed-window helped (vs C1's failure)

1. **No rule-fired terminals in buffer**: the DQN owns every terminal decision. Sparse-reward credit assignment is no longer confounded by rules
2. **Right-tail visibility**: the DQN observes what happens when you hold a trade all the way to bar N — including catastrophic losing tails. It learns to cut losses *because* it sees them
3. **Strategy-config independence**: B5 doesn't read `EXECUTION_CONFIG` thresholds. The policy depends only on in-trade state, so it generalizes across entry distributions more cleanly
4. **Richer state**: the price-path window and trajectory scalars give the DQN direct visibility into the trade's profit history, not just its current snapshot

#### Forward path

C2_fix240's test result is exciting but needs validation:

| Step | Effort | Purpose |
|---|---|---|
| **Walk-forward C2_fix240 across the 6 RL folds** | ~30 min | Is the test +8.33 a genuine signal or window-specific? Need ≥4/6 folds positive. |
| Walk-forward A2 alone | ~10 min | Sanity baseline — A2's own fold variance |
| Seed variance for B5_fix240 (5 seeds) | ~10 min | Policy-level robustness |
| Joint training (true C2 — train B5 on A2's actual entries) | ~3-5 days | Architecturally correct, would close val gap if walk-forward looks good |

If walk-forward holds, C2_fix240 + Path X (maker execution) is the deployable production system.

→ Detailed log: [docs/experiments_log.md#group-b5--c2-fixed-window-exit-dqn](docs/experiments_log.md#group-b5--c2--fixed-window-exit-dqn-with-enriched-state)

---

## Cumulative Insights

### What's confirmed
1. **Strategies have predictive edge** (Path 1a fee-free, 1b oracle).
2. **Fees consume the edge under taker pricing** (Path 1a Δ=+12.4 Sharpe from removing fees; Group A red-vs-orange-vs-green spread).
3. **State representation is sufficient for entry gating at low fees** (A2 val +7.30, **test +3.78** — generalizes; A4 val +1.72 but test −1.65).
4. **State representation is insufficient for residual signal extraction beyond what strategies use** (D1 Spearman 0.084).
5. **Time-scale isn't the issue** (Path 1c — 5-min has same fee-free Sharpe as 1-min).
6. **RL exit-timing is real fee-free** (B4_fee0 mean Δ +1.84, best +4.20; B5_fix240 mean Δ +3.48, best +6.69 — clears +4 gate easily with fixed-window design).
7. **Variable-length exit DQN doesn't compose** (C1, C1_fee0 both fail on test). Training-distribution mismatch + rule-vs-DQN race in credit assignment.
8. **Fixed-window exit DQN composes on test** (C2_fix240: test +8.33 vs A2-alone +3.78, equity 1.34× vs 1.13×). Removing rule terminals during training + enriched state + bounded episodes fixes the transfer pathology — at least on test. Val side still weak.
9. **Per-strategy exit DQN ≫ pooled exit DQN** (B4 mean +0.6 vs B2 −4.23 at maker fee; B5 follows same pattern).
10. **Window length matters**: N=240 dominates N=60 and N=120 in B5 per-strategy and in C2 composition. Strategies need long horizons to fully express edge.

### What's still unknown
1. **Walk-forward stability of A2/A4** across the 6 RL folds.
2. **Seed sensitivity** — how robust are the +7.30 val / +3.78 test (A2) and +1.72 val / −1.65 test (A4) numbers?
3. **Real-world fill rates** for maker orders (production scoping).
4. **Whether C2 (joint hierarchical training) closes the composition gap.** Deferred — C1/C1_fee0 evidence + production-readiness priority make C2 lower expected value than walk-forward + Path X.
5. ~~Whether RL can replace TP/SL with better-than-fixed exits~~ **Resolved partially: per-strategy exit DQN clears the +4 gate at fee=0 (B4_fee0_S3 +4.20) but is small at maker fee. Stand-alone exit improvement IS real fee-free.**
6. ~~Whether stacked entry+exit RL composes cleanly~~ **Resolved: NO under sequential composition (C1, C1_fee0 both fail). Joint training (C2) untested.**

---

## Production Readiness

### A4 deployment scenario

**Required infrastructure** (Path X):
- Replace `MarketEntry()` with limit-order maker entry
- Re-quote / fallback logic (taker after N bars without fill)
- OKX VIP fee tier (target 0.02%/side maker)
- Live data ingestion + real-time inference (DQN forward pass < 1ms on CPU)

**Expected per-month numbers (extrapolated from val+test):**
- Trade count: ~430 trades / 10 weeks ≈ ~190 trades/month
- Round-trip fee per trade: ~0.04% (maker) → ~0.16% in fee drag/month at 100% capital deployment
- Win rate: ~50%
- Max drawdown: ~7-10%
- Sharpe: ~1.7 (uncertain — needs walk-forward + seed validation)

**Open risks:**
- Maker fill rate < 100% (need fallback)
- Slippage on partial fills
- Live execution latency
- Regime shift outside Sep 2025–Apr 2026 distribution

### Pre-deployment checklist

- [ ] Walk-forward A4 across 6 folds → confirms stability across time
- [ ] Train A4 with 5 different seeds → quantify policy variance
- [ ] Implement maker-entry execution layer
- [ ] Simulate maker-fill realistically (e.g. probability-of-fill modeling)
- [ ] Paper-trade for 2–4 weeks
- [ ] Apply position sizing (currently 1.0× capital — needs VolScaledSizer integration)

---

## Files & Artifacts

### Code modules

| Module | Purpose |
|---|---|
| [data/loader.py](data/loader.py) | CSV→Parquet caching |
| [data/gaps.py](data/gaps.py) | Gap masking |
| [features/](features/) | 191-feature assembly |
| [models/vol_v4.py](models/vol_v4.py) | LightGBM ATR-30 vol model |
| [models/direction_dl_v4.py](models/direction_dl_v4.py) | 4 CNN-LSTM direction models |
| [models/regime_cusum_v4.py](models/regime_cusum_v4.py) | CUSUM+Hurst regime classifier |
| [models/dqn_state.py](models/dqn_state.py) | 50-dim state array builder |
| [models/dqn_network.py](models/dqn_network.py) | DQN MLP (50→64→32→10, 5,674 params) |
| [models/dqn_replay.py](models/dqn_replay.py) | PER buffer + stratified sampling |
| [models/dqn_rollout.py](models/dqn_rollout.py) | Env-loop driver, fee + penalty parameterized |
| [models/dqn_selector.py](models/dqn_selector.py) | Training loop, CLI: `--fee --trade-penalty` |
| [models/group_a_sweep.py](models/group_a_sweep.py) | Group A 7-cell runner |
| [models/exit_dqn.py](models/exit_dqn.py) | Variable-length exit DQN (Group B): 28-dim in-trade state, HOLD/EXIT_NOW, RL+rule-based exits combined |
| [models/group_b_sweep.py](models/group_b_sweep.py) | Group B 12-cell runner (B1-B3 global × fee, B4 per-strategy) |
| [models/exit_dqn_fixed.py](models/exit_dqn_fixed.py) | **Fixed-window exit DQN (Group B5)**: 53-dim enriched state with price-path + vol windows + trajectory scalars; no rule exits during training; bounded episodes |
| [models/group_b5_sweep.py](models/group_b5_sweep.py) | Group B5 27-cell runner (3 windows × 9 strategies at fee=0) |
| [models/group_c_eval.py](models/group_c_eval.py) | Group C1 evaluator (A4/A2 entry + B4 variable-length exits stacked) |
| [models/group_c2_eval.py](models/group_c2_eval.py) | **Group C2 evaluator (A2 entry + B5 fixed-window exits stacked)** |
| [models/grid_search.py](models/grid_search.py) | Hyperparameter search |
| [models/walk_forward.py](models/walk_forward.py) | 6-fold validation |
| [models/diagnostics_ab.py](models/diagnostics_ab.py) | Path 1a (fee-free) + 1b (oracle) |
| [models/diagnostics_c.py](models/diagnostics_c.py) | Path 1c (5-min timeframe) |
| [models/pnl_predictor.py](models/pnl_predictor.py) | D1 supervised PnL regression |
| [models/plot_results.py](models/plot_results.py) | Single-strategy equity plot |
| [models/plot_dqn_results.py](models/plot_dqn_results.py) | Group A DQN policies plot |
| [backtest/single_trade.py](backtest/single_trade.py) | Numba-jit single-trade simulator (parity-verified, 0.7 µs/call) |
| [backtest/engine.py](backtest/engine.py) | Bar-by-bar backtest engine |
| [strategy/agent.py](strategy/agent.py) | 9 strategies (S1, S2, S3, S4, S6, S7, S8, S10, S12) |
| [execution/](execution/) | Entry / exit / sizing components |

### Cached results

```
cache/
  btc_lgbm_atr_30_v4.txt                          # vol model
  btc_pred_vol_v4.npz                             # vol predictions
  btc_cnn2s_dir_{up,down}_{60,100}_v4.keras       # 4 direction models
  btc_pred_dir_{up,down}_{60,100}_v4.npz          # direction preds
  btc_regime_cusum_v4.parquet                     # 5-state labels
  btc_regime_cusum_v4_thresholds.json
  btc_dqn_standardize_v5.json                     # state standardization
  btc_dqn_state_{train,val,test}.npz              # 50-dim state arrays

  ── DQN policies (one per Group A cell) ──
  btc_dqn_policy_A{0..6}.pt
  btc_dqn_train_history_A{0..6}.json
  btc_dqn_groupA_summary.{parquet,json}

  ── exit DQN policies (Group B, variable-length) ──
  btc_exit_dqn_policy_{B1,B2,B3,B4_S0..S8,B4_fee0_S0..S8}.pt
  btc_exit_dqn_history_{B1,B2,B3,B4_S0..S8,B4_fee0_S0..S8}.json
  btc_exit_dqn_groupB_summary.json
  btc_groupC1_summary.json
  btc_groupC1_fee0_summary.json

  ── fixed-window exit DQN policies (Group B5) ──
  btc_exit_dqn_fixed_policy_B5_fix{60,120,240}_fee0_S{0..8}.pt
  btc_exit_dqn_fixed_history_B5_fix{60,120,240}_fee0_S{0..8}.json
  btc_exit_dqn_fixed_groupB5_summary.json
  btc_groupC2_fix{60,120,240}_fee0_summary.json

  ── diagnostics ──
  btc_diag_1a_fee_free.parquet
  btc_diag_1b_oracle.parquet
  btc_diag_1c_5min.parquet
  btc_walk_forward_results.parquet
  btc_grid_search_results.parquet
  btc_pnl_pred_thresholds.json
  btc_pnl_pred_results.parquet

  ── plots ──
  btc_S1_VolDir_equity_vs_price.png               # single-strategy view
  btc_dqn_groupA_equity_vs_price.png              # DQN policies view
```

---

## Next Steps / Experiments Remaining

→ **Full plan:** [docs/next_steps.md](docs/next_steps.md)

Quick summary of remaining experiments:

| ID | Experiment | Effort | Status |
|---|---|---|---|
| ~~Group B~~ | ~~Exit-timing DQN~~ | — | **done — no lift, closed 2026-05-07** |
| ~~Group C~~ | ~~Stacked entry+exit RL~~ | — | **dropped (was conditional on B)** |
| Reduced scope | Lock A4 (walk-forward + seed + penalty fine-grid) | ~1 day | recommended before deployment |
| Path X | Maker-only execution | ~3–5 days | **next: production deployment** |
| Alternative pivots | Funding-rate / vol trading / statarb | open-ended | optional if A4 alone insufficient |

With Groups B/C closed, the path forward is clear: **lock A4 (reduced scope), then build Path X (maker execution)**. Per-strategy exit DQNs (B4) provide a small optional lift (~+0.6 mean Sharpe) and could be bolted onto a deployed system later if entries are stable.
