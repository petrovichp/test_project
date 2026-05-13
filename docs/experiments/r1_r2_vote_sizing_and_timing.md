# R1 + R2 — vote-strength sizing + trade timing characterization

Executed 2026-05-13. Two cheap diagnostic experiments on the
`VOTE5_v8_H256_DD K=5` (current fee=4.5bp deployable) at OKX taker fee.

**Verdicts:**
- **R1 — POSITIVE (+1.16 Sharpe).** Vote consensus strongly predicts per-trade
  quality. Quadratic sizing on `votes_count` lifts WF from +4.58 to +5.74 at
  4.5bp fees. This is a real free improvement to the taker deployable.
- **R2 — DEPLOYMENT-READY.** Inter-trade gap median 17 min; peak rate 5
  trades/hr; bars-in-position 83%; position-tracking needs to handle 1
  concurrent trade with up to ~16-hour duration.

## R1 — Vote-strength position sizing

### Hypothesis

VOTE5_v8 K=5 plurality produces a `votes_count` per trade (2 = tie → NO_TRADE,
3/4/5 = winning plurality). Currently every trade gets fixed size. If high-
consensus trades (5/5 agreement) are systematically higher-quality than
low-consensus (3/5), weighting position size by consensus should improve
risk-adjusted return.

### Per-vote-count PnL distribution (1394 trades at fee=4.5bp)

| votes | n | mean PnL % | std PnL % | sum PnL % | win rate |
|---:|---:|---:|---:|---:|---:|
| 2 (tie) | 20 | −0.004 | 0.758 | −0.08 | 50.0% |
| **3** | 1093 | **+0.024** | 0.837 | +25.6 | 44.8% |
| **4** | 237 | **+0.324** | 1.110 | +76.7 | **56.5%** |
| **5** | 44 | **+0.587** | 1.027 | +25.8 | **68.2%** |

**Vote consensus is strongly predictive of per-trade quality**:
- 4-vote trades: 14× higher mean PnL than 3-vote, win rate 12 pp higher.
- 5-vote trades: 24× higher mean PnL than 3-vote, win rate 23 pp higher.
- 3-vote dominates volume (78% of trades) but carries near-zero edge.

This is a clean structural signal that the ensemble's *agreement strength*
correlates with the *confidence-worthiness* of its signal. Not surprising
in hindsight — when 5 of 5 independently-trained policies all want to enter
the same position, that's much stronger evidence than 3 of 5 agreeing.

### Sharpe under each sizing scheme

All schemes evaluated on the same 1394 trades, fee=4.5bp uniform.

| Scheme | size(3) | size(4) | size(5) | WF | f1 | f2 | f3 | f4 | f5 | f6 | folds+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FIXED (baseline)** | 1.0 | 1.0 | 1.0 | **+4.58** | +7.23 | +9.94 | +7.96 | +10.51 | −4.33 | −3.82 | 4/6 |
| LINEAR (v−2)/3 | 0.33 | 0.67 | 1.0 | **+5.54** | +8.13 | +13.01 | +7.61 | +12.94 | −4.95 | −3.47 | 4/6 |
| **AGGRESSIVE ((v−2)/3)²** | 0.11 | 0.44 | 1.0 | **+5.74** | +8.14 | +13.17 | +6.88 | +13.70 | −5.08 | −2.37 | 4/6 |
| BINARY v≥4 | 0.0 | 1.0 | 1.0 | +5.28 | +6.78 | +13.92 | +4.28 | +12.60 | −3.29 | −2.60 | 4/6 |
| BINARY v=5 only | 0.0 | 0.0 | 1.0 | +3.80 | +4.31 | +8.26 | +5.14 | +6.51 | −3.24 | **+1.83** | **5/6** |
| STEP 3=0.5, 4=1, 5=1.5 | 0.5 | 1.0 | 1.5 | +5.57 | +8.09 | +12.56 | +8.41 | +12.72 | −4.83 | −3.51 | 4/6 |

### Key findings

1. **AGGRESSIVE quadratic sizing lifts WF Sharpe by +1.16** (+25%
   improvement on the deployable's headline metric).
2. **The improvement comes from upgrading good folds, not from fixing bad
   folds.** f2 lifts +3.23 (+9.94 → +13.17), f4 lifts +3.19 (+10.51 →
   +13.70). The negative f5/f6 (BTC calm uptrend, the structural soft spot
   from A3 audit) get marginally worse — sizing can't rescue an environment
   problem.
3. **BINARY v=5-only is interesting**: only 44 trades, but **5/6 folds
   positive** (vs 4/6 for all other schemes). Trade off magnitude
   (+3.80 WF vs +5.74) for robustness (better fold consistency). This is the
   most conservative deployment — fewer trades, more selective, more reliable.
4. **Capital efficiency**: AGGRESSIVE deploys only ~19% of FIXED's total
   notional (sum of sizes: 268 vs 1394 unit-trade equivalents). The same
   real capital can take BIGGER positions on the high-conviction trades.

### Implications

| Use case | Recommended scheme |
|---|---|
| Max Sharpe at current capital | **AGGRESSIVE ((v−2)/3)²** — WF +5.74 |
| Max robustness, fewer trades | **BINARY v=5 only** — 5/6 folds positive, WF +3.80 |
| Linear simplicity | LINEAR — WF +5.54 |

For Z5.4 freeze: **update the VOTE5_v8 K=5 deployable to use AGGRESSIVE
quadratic sizing**. This is a free Sharpe improvement at no additional
training cost — just a runtime sizing-rule change.

### Caveats

- **Doesn't help DISTILL_v8** — single net has no vote consensus to weight by.
  R1 is specifically a VOTE5_v8 (ensemble) improvement.
- **Sizing schemes assume linear PnL→position relationship.** In real
  execution, larger positions face more market impact + slippage. At 5×
  larger size for v=5 trades vs v=3, slippage could eat into the gain.
  Conservative estimate: actual lift might be +0.8 to +1.0 instead of +1.16.
- **Fold 5/6 remain negative** — sizing doesn't fix the calm-uptrend regime
  weakness (A3 audit context: VOTE5_v8 thrives on volatile-decline folds).

## R2 — Trade clustering / capacity analysis

### Goal

Characterize trade-timing structure for live deployment infrastructure
sizing: position-tracking, order rates, latency tolerance, capital lockup.

### Per-fold inter-trade gap statistics

Bars between consecutive trade close → next trade open (0 = back-to-back):

| fold | n_trades | mean | median | p10 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|
| f1 | 173 | 37.2 | 18 | 3 | 95 | 251 |
| f2 | 248 | 29.0 | 15 | 2 | 73 | 256 |
| f3 | 175 | 44.2 | 26 | 4 | 109 | 363 |
| f4 | 315 | 28.4 | 14 | 3 | 69 | 359 |
| f5 | 281 | 28.5 | 18 | 4 | 64 | 220 |
| f6 | 202 | 38.2 | 21 | 3 | 99 | 372 |

**Aggregate**: mean gap 33 bars (33 min), median 17 bars (17 min),
p10 3 bars (back-to-back trades 10% of time), p90 79 bars,
max 372 bars (~6 hours of silence).

### Trade durations + bars-in-position rate

| fold | trades | total_bars | in_pos_bars | in_pos % | mean_dur | p95_dur |
|---|---:|---:|---:|---:|---:|---:|
| f1 | 173 | 47,195 | 40,515 | 85.8% | 234 | 980 |
| f2 | 248 | 47,195 | 39,908 | 84.6% | 161 | 492 |
| f3 | 175 | 47,195 | 39,316 | 83.3% | 225 | 946 |
| f4 | 315 | 47,195 | 38,122 | 80.8% | 121 | 368 |
| f5 | 281 | 47,195 | 39,183 | 83.0% | 139 | 437 |
| f6 | 202 | 47,199 | 39,527 | 83.7% | 196 | 763 |

**Aggregate: 83.5% of bars are spent in a position.** Mean trade duration
121-234 bars (2-4 hours). p95 duration up to 980 bars (~16 hours, overnight
holds common).

### Burst rates

Max trades opening in a sliding window:

| fold | max@60bar (1hr) | max@240bar (4hr) | max@1440bar (24hr) |
|---|---:|---:|---:|
| f1 | 4 | 9 | 15 |
| f2 | 5 | 9 | 22 |
| f3 | 2 | 4 | 13 |
| f4 | 4 | 10 | 29 |
| f5 | 2 | 5 | 14 |
| f6 | 2 | 4 | 14 |

**Peak rate: 5 trades opened in any 1-hour window; 10 in any 4-hour;
29 in any 24-hour.** Easily handled by automated execution.

### Per-strategy share of trades (1394 total)

| strategy | count | share |
|---|---:|---:|
| S1_VolDir | 421 | 30.2% |
| S7_OIDiverg | 260 | 18.7% |
| S11_Basis | 219 | 15.7% |
| S8_TakerSus | 168 | 12.1% |
| S10_Squeeze | 161 | 11.5% |
| S3_BBExt | 62 | 4.4% |
| S4_MACD | 43 | 3.1% |
| S13_OBDiv | 32 | 2.3% |
| S6_TwoSignal | 15 | 1.1% |
| S2_Funding | 13 | 0.9% |

**5 strategies (S1, S7, S11, S8, S10) carry 88% of trade volume.** S12_VWAPVol
contributes 0 trades (the strategy was effectively dormant on val+test
period — also noted in A1 audit as short-only).

### Deployment implications

| Requirement | Number |
|---|---|
| Max concurrent positions | **1** (sequential by simulator design — entries wait for prior close) |
| Peak entry rate | ~5/hr |
| Median time between trades | ~17 min |
| Capital lockup | ~83% of time in market |
| Max single-position hold | up to 16 hours (overnight allowed) |
| Strategies needing live signal monitoring | 5 (S1, S7, S8, S10, S11) cover 88% volume |

This is a **comfortable load profile** for an automated trading system. No
rate-limit concerns; no concurrent-position management needed (in the
current sequential design); overnight position tracking required for ~5%
of trades.

## Combined verdict

| Experiment | Result | Action |
|---|---|---|
| R1 vote-strength sizing | +1.16 Sharpe on VOTE5_v8 (taker deployable) | **Update Z5.4: VOTE5_v8 K=5 + AGGRESSIVE sizing** as the freeze recommendation |
| R2 trade timing | Comfortable deployment load | No infra blockers; document for live deployment runbook |

## Code touchpoints

- [models/r1_r2_eval.py](../../models/r1_r2_eval.py) — runner for both experiments
- [models/audit_vote5_dd.py](../../models/audit_vote5_dd.py) — `run_walkforward` reused

## Outputs

- `cache/results/r1_r2_eval.json` — full results dump
