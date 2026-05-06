# Next Steps

Forward-looking plan after Group A breakthrough. Companion to [RESULTS.md](../RESULTS.md) and [experiments_log.md](experiments_log.md).

---

## Decision Tree

```
              ┌─ deployment first ─→ Reduced scope (1 day) → Path X (3-5 days) → paper-trade
              │
Where now? ───┤
              │
              └─ research first ───→ Group B (1.5-2d) → if wins → Group C (3-5d) → deploy best
```

---

## Group B — Exit-timing DQN (~1.5–2 days)

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

## Group C — Stacked entry+exit RL (~3–5 days, conditional)

**Only run if Group A and Group B both win.**

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

## Recommended sequencing

```
Step 1: Group A reduced scope (~1 day)
        ├─ Walk-forward A4
        ├─ Penalty fine-grid at fee=0.0004
        └─ A4 seed variance
        ↓
Step 2: Decision
        ├─ A4 is solid → start Path X scoping in parallel with Group B
        └─ A4 is fragile → drop A4, pivot to Group B exclusively
        ↓
Step 3a: Path X (~4-5 days) — production scoping if A4 solid
Step 3b: Group B (~1.5-2 days) — exit-timing RL
        ↓
Step 4: Decision
        ├─ Both Path X + Group B succeed → deploy Group A4 + plan Group C
        ├─ Path X works, Group B doesn't → deploy A4-only
        ├─ Path X fails (maker fill rate too low) → drop deployment, focus on Group B+C alphas
        └─ Group B works alone → deploy with default entries + RL exits
```

---

## Open questions (research only — not blockers)

1. **Why does the DQN concentrate 95-98% on NO_TRADE across all cells?** Mask coverage allows trades on ~30% of bars. The DQN being more selective than the mask is a feature, but the magnitude is striking.

2. **Why does penalty help at fee=0 but hurt at fee=0.0004?** Suggests the optimal selectivity threshold is fee-dependent and roughly compensates for fees. Could formalize as a reward-shaping theory.

3. **Does the prior CUSUM gate +3.13 result (CLAUDE.md) replicate at all?** Walk-forward (D3) showed S4+CUSUM mean −3.13 across folds. Either the original was a single-window artifact or there's a difference in evaluation pipeline.

4. **What's the ceiling on entry-gating?** A2 gives +7.30 fee-free. Oracle gives +35.68 fee-free. Where is the +28-Sharpe gap from? Likely 90%+ from exits (Group B) and the rest from per-bar entry timing (intra-bar precision, not addressed here).
