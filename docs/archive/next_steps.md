# Experiments Remaining + Next Steps

Forward-looking plan after Group A breakthrough and Group B closeout. Companion to [RESULTS.md](../RESULTS.md) and [experiments_log.md](../reference/experiments_log.md).

---

## Experiments Remaining (quick index)

| ID | Experiment | Effort | Status |
|---|---|---|---|
| **Path X** | Maker-only execution scoping | ~3–5 days | **next priority** — A2+rule passed walk-forward 6/6 (mean +9.00). Deploy with confidence. |
| **Seed variance for A2** | Train A2 with 5 seeds, measure Sharpe std | ~30 min | policy-level robustness check before live deployment |
| **Joint hierarchical training (C2-true)** | Train B5 on A2's actual entry distribution | ~3–5 days | optional — could let B5 universally beat rule-based; only worth running if alpha-bound after deployment |
| **Alternative pivots** | Funding-rate / vol trading / statarb / cross-timeframe | open-ended | only if Path X paper-trade fails |
| ~~Walk-forward C2_fix240~~ | — | — | **completed 2026-05-07: A2+rule wins 5/6 folds, A2+B5 wins 1/6. Rule-based is the deployment target.** |
| ~~Group B (variable-length)~~ | — | — | **closed: per-strategy mean +0.6, below +4 gate at maker fee** |
| ~~Group B4_fee0~~ | — | — | **closed: per-strategy mean +1.84, best +4.20 clears gate** |
| ~~Group C1, C1_fee0~~ | — | — | **closed: variable-length composition fails (Δ −0.89 / −2.70 on test)** |
| ~~Group B5 (fixed-window)~~ | — | — | **completed: 9/9 positive at N=240, mean Δ +3.48** |
| ~~Group C2_fix240~~ | — | — | **completed: test +8.33 was fold-6-specific; walk-forward shows rule-based wins 5/6** |

Detailed breakdowns of each remaining experiment in the sections below; Group B / C summaries kept for the record.

---

## Decision Tree (post-Group-B)

```
Step 1: Reduced scope (~1 day)
        ├─ Walk-forward A4 across 6 RL folds
        ├─ Seed variance (5 seeds)
        └─ Penalty fine-grid at fee=0.0004
        ↓
Step 2: A4 stable?  ── yes → Path X (3-5 days) → paper-trade → live
                    └─ no  → pivot (funding-rate / vol / statarb)
```

---

## Group B — closed (small lift, below gate)

Tested in May 2026: 12 cells (3 global × fee level + 9 per-strategy at maker).

- **B1-B3 (global exit DQN, all strategies pooled):** negative Δ at every fee level (worst −7.55 at taker, best −1.52 at fee=0). Pooling heterogeneous strategies into one shared exit policy is actively harmful.
- **B4 (per-strategy):** **6/9 strategies improve, mean Δ +0.6, best +1.97** (S8_TakerFlow). Real but small — well below the +4-Sharpe gate set as the success criterion.
- **C1 (A4 entry + B4 exits, sequential composition):** **does not transfer**. Δ −0.07 val, −0.89 test. B4 was trained on a "sequential first-firing" entry distribution; A4's selective entries (~3% of bars) have different in-trade dynamics, and B4's policy bails too eagerly on them.

Full numbers in [experiments_log.md § Group B](../reference/experiments_log.md#group-b--exit-timing-dqn) and [§ Group C1](../reference/experiments_log.md#group-c1--a4-entry--b4-per-strategy-exits-sequential-composition).

Lesson: the +28-Sharpe oracle gap most likely lives in *intra-bar entry timing* (sub-1-minute resolution that the current architecture cannot see), not in exit selection. Future exit work would need a different formulation than HOLD/EXIT_NOW (e.g., dynamic SL placement, signal-driven exit thresholds inside the strategies themselves), and would need to be co-trained with the entry policy to avoid the C1 transfer failure.

---

## Group B (original spec, kept for reference) — Exit-timing DQN

**Question:** can RL replace fixed TP/SL/BE/trail and capture part of the ~14-Sharpe oracle gap?

### Cells

| ID | Variant | Fee | Tests |
|---|---|---|---|
| B1 | DQN exit, default entries | 0.0008 (taker) | Can RL exit beat fixed TP/SL with realistic fees? |
| B2 | DQN exit, default entries | 0.0004 (maker) | Production scenario for exit RL |
| B3 | DQN exit, default entries | 0 | Pure exit-quality lift, fee-free |
| B4 | DQN exit, **per-strategy** | 0.0004 (maker) | One exit DQN per entry strategy |

### Formulation

Different RL setup from Group A:

| | Group A (entry) | Group B (exit) |
|---|---|---|
| State | 50-dim per bar | ~30-dim per bar inside open trade |
| Actions | NO_TRADE + 9 strategies | HOLD / EXIT_NOW |
| Reward | trade pnl on close | realized pnl on EXIT |
| Episode | continuous | one trade |

**State (in-trade):** `[unrealized_pnl_pct, n_bars_in_trade, current_trail_sl_dist, recent_log_return×6_lags, recent_taker_net_60_z×6, recent_ofi_perp_10×6, recent_vwap_dev_240×6, entry_direction]` ≈ 30 dims.

### Implementation cost

| Step | Effort |
|---|---|
| Per-trade-bar state builder | ~half day |
| Episodic training loop | ~half day |
| Evaluator + walk-forward integration | ~half day |
| **Total** | **~1.5–2 days** |

Training time per cell: ~10 min. Total: ~40 min.

### Decision rule

| Outcome | Action |
|---|---|
| RL exit captures ≥30% of actual-vs-oracle gap (≥+4 Sharpe at fee=0) | Real value-add. Deploy with Group A entry stack. |
| RL exit ≈ rule-based exit | Strategies are exit-optimal already. Drop B, focus elsewhere. |
| RL exit < rule-based exit | Exit RL formulation needs redesign or skip. |

---

## Group C (dropped) — Stacked entry+exit RL

**Was conditional on Group B success. Group B did not clear its gate, so C is dropped.** The original spec is kept below for the record.

| ID | Variant | Tests |
|---|---|---|
| C1 | Best-A entry DQN + Best-B exit DQN, sequential | Naive composition (no joint training) |
| C2 | Hierarchical DQN: shared rollout, joint training | Coordinated entry+exit |

### Decision rule

| Outcome | Action |
|---|---|
| Both A and B work, C composes cleanly | Integrated production policy: entry-DQN + exit-DQN stack |
| Either A or B works alone | Ship the simpler single-stage version |
| C composition degrades vs A or B alone | Use whichever single-stage is best |

---

## Reduced scope option — Lock A4 first (~1 day)

For users wanting **deployment first, research second**.

### Mini-experiments

| Experiment | Cost | Output |
|---|---|---|
| **Walk-forward of A4** across 6 RL-period folds | ~1 hour | Stability of A4 at maker fee — confirms deployability |
| **Penalty grid at fee=0.0004** (α ∈ {0, 0.0005, 0.001, 0.002, 0.003, 0.005}) | ~30 min | Precise penalty optimum at production fee |
| **Seed sensitivity** — train A4 with 5 random seeds | ~15 min | Variance of result; how robust is +1.72? |

These bound the deployable A4 result. If walk-forward shows ≥4/6 folds positive AND seed std < 1.0, A4 is production-ready.

### Then Path X (production maker execution)

| Step | Effort |
|---|---|
| Implement maker-entry execution layer (limit order + re-quote + fallback) | ~2 days |
| Simulate maker-fill probabilistically (slippage + queue position) | ~1 day |
| Re-validate A4 with realistic maker-fill simulation | ~half day |
| Paper-trade integration (live data + DQN inference) | ~1 day |
| **Total deployment scoping** | **~4–5 days** |

---

## Path X — Maker-only execution (production)

**Why:** Group A4 needs maker fee to be deployable. Without it, Phase 3 baseline (taker fee) returns.

**OKX maker pricing (May 2026):**

| Tier | Maker fee | Taker fee | Round-trip cost |
|---|---|---|---|
| Default (LV1) | 0.020% | 0.050% | 0.070% (taker only) |
| LV3+ (high vol) | 0.014% | 0.040% | 0.054% |
| LV5+ (Pro) | rebate −0.005% | 0.030% | rebate net (effectively pays you) |
| LV7+ (Pro) | rebate −0.010% | 0.020% | strong rebate |

Our A4 simulation assumed 0.04% per side (≈0.08% round-trip). Real maker (0.020% per side, 0.04% round-trip) is **half** that, which would push A4 deployable Sharpe higher than +1.72.

**Production code path:**

| Module to add | Purpose |
|---|---|
| [execution/maker_entry.py](../execution/) (new) | Limit order placement at bid (buy) / ask (sell), re-quote on price drift, fallback to taker after N bars |
| [execution/fill_simulator.py](../execution/) (new) | Probabilistic fill model (queue depth, time-to-fill distribution) |
| [backtest/run_maker.py](../backtest/) (new) | Backtest variant using maker entry + fill simulation |

**Risks to address:**
- Partial fills (size scaled smaller than intended)
- Adverse selection (fills occur preferentially when price about to move against us)
- Cancel-and-replace overhead under fast markets

---

## Alternative: pivot to other alpha sources

Only if Groups B/C don't move the needle and A4 alone isn't enough for production.

| Alpha source | Feasibility | Why interesting |
|---|---|---|
| **Funding-rate / basis arbitrage** | Easy — already have data | OKX funding paid every 8h, basis trading has structural edge |
| **Volatility trading** | Medium — need options data integration | Vol model has Spearman 0.69, strong signal under-utilized |
| **Statistical arbitrage (BTC×ETH×SOL)** | Easy — pipeline reuses | Cross-asset cointegration is a different alpha source |
| **Cross-timeframe ensemble** | Hard — needs per-timeframe model retraining | 5-min for signals + 1-min for execution |

---

## Recommended sequencing (post walk-forward 2026-05-07)

Walk-forward of C2_fix240 across 6 RL folds delivered the decisive verdict: **A2 + rule-based exits dominates A2 + B5 RL exits on 5 of 6 folds (mean Sharpe +9.00 vs +2.59).** The deployable system is now well-defined.

```
Step 1: Path X (~3–5 days) — maker execution scoping
        ├─ Limit-order placement + re-quote logic
        ├─ Probabilistic fill simulation
        └─ Re-validate A2 + rule-based exits with realistic maker-fill model
        ↓
Step 2: Seed variance for A2 (~30 min)
        └─ Train A2 with 5 seeds; std < 1.0 → policy is robust
        ↓
Step 3: Paper-trade (2-4 weeks)
        ├─ Live data ingestion + DQN inference (~1 ms/bar on CPU)
        ├─ Track fill rates, slippage, latency
        └─ Compare paper PnL vs backtest expectation
        ↓
Step 4: Live deploy
        └─ A2 entry DQN + rule-based exits at OKX maker fee tier
        ↓
Optional follow-ups:
        ├─ Joint hierarchical training (C2-true): train B5 on A2's actual
        │    entry distribution. Could let RL exits universally beat rule-based;
        │    revisit if alpha-bound after deployment validates entry signal.
        ├─ B5 as regime-stress fallback: monitor rolling A2+rule Sharpe;
        │    switch to A2+B5 when rule-based degrades (fold-6-style regime).
        ├─ Cross-asset: train A2-style entry on ETH and SOL.
        └─ Alternative exit formulations: dynamic SL placement, signal-driven
            thresholds inside strategies (decoupled from RL).
```

---

## Open questions (research only — not blockers)

1. **Why does the entry DQN concentrate 95-98% on NO_TRADE across all cells?** Mask coverage allows trades on ~30% of bars. The DQN being more selective than the mask is a feature, but the magnitude is striking.

2. **Why does penalty help at fee=0 but hurt at fee=0.0004?** Suggests the optimal selectivity threshold is fee-dependent and roughly compensates for fees. Could formalize as a reward-shaping theory.

3. **Does the prior CUSUM gate +3.13 result (CLAUDE.md) replicate at all?** Walk-forward (D3) showed S4+CUSUM mean −3.13 across folds. Either the original was a single-window artifact or there's a difference in evaluation pipeline.

4. **Where does the residual +28-Sharpe oracle gap live?** A2 gives +7.30 fee-free; oracle gives +35.68 fee-free. Group B confirmed it's *not* in HOLD/EXIT_NOW exit selection (best lift +1.97, well short of +28). Most plausible: intra-bar entry timing — picking the *right second* within a 1-minute bar to enter, which the current architecture cannot resolve.

5. **Why does pooled exit DQN underperform per-strategy exit DQN by a wide margin?** B2 Δ −4.23 (pooled), B4 mean Δ +0.6. Strategies must have heterogeneous exit signatures the shared policy cannot disentangle. Possibly a per-strategy embedding in the state would let one network share parameters while specializing.
