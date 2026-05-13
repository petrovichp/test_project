# Post-latency improvements: vote×lag, action cleanup, maker-on-entry

Executed 2026-05-13 as three sequential follow-ups to the
[signal-entry-latency analysis](r1_r2_vote_sizing_and_timing.md).

**Summary of verdicts:**

| Step | Hypothesis | Verdict | Production change |
|---|---|---|---|
| **1 — Vote×lag joint filter** | Lag adds info beyond vote-strength | **NEGATIVE** — lag is redundant with vote | None |
| **2a — Drop S12_VWAPVol** | S12 is dead weight (0 trades) | **CONFIRMED** — zero impact when masked | Drop in next state-pack rebuild |
| **2b — Drop S6_TwoSignal** | S6 has 0.42% hit rate, probably dead | **REFUTED** — S6 contributes +0.79 Sharpe when present | Keep S6 |
| **3 — Maker-on-entry** | Try limit orders before falling back to taker | **POSITIVE +0.79 to +1.23 lift** | Implement in execution layer |

## Combined production impact

Stacking R1 AGGRESSIVE sizing + S12 drop (no-op) + maker-on-entry:

| Configuration | WF Sharpe @ 4.5bp |
|---|---:|
| Current baseline (FIXED sizing, no maker) | +4.58 |
| **+ R1 AGGRESSIVE sizing** (already deployed) | **+5.74** (+1.16) |
| + R1 + maker-on-entry N=1 (conservative) | **+6.22** (+1.64) |
| + R1 + maker-on-entry N=5 | +6.44 (+1.86) |

**Honest production estimate (N=1, discounted 30% for adverse selection)**:
WF ~+6.0-6.3 expected, **+1.5-1.7 lift** over the current frozen baseline.

---

## Step 1 — Vote × lag joint filter

### Hypothesis
R1 found vote_count → quality (24× PnL ratio v=5 vs v=3). Signal-latency
finding showed 56% of trades enter at lag=0. Hypothesis: vote=5 AND lag=0
trades are the highest-conviction setups, deserving the largest position.

### Method
Bucket trades by (vote × lag) and test 7 sizing schemes that combine the
two signals.

### Result table — Mean PnL% per bucket

| | lag=0 | lag 1-3 | lag 4+ |
|---|---:|---:|---:|
| v=3 | +0.010 | +0.032 | +0.050 |
| v=4 | +0.349 | +0.234 | +0.385 |
| v=5 | +0.644 | +0.439 | +0.529 |

### Sizing scheme comparison

| scheme | WF |
|---|---:|
| FIXED baseline | +4.58 |
| **AGGRESSIVE quadratic ((v−2)/3)²** (R1 winner) | **+5.74** |
| Vote-only LINEAR | +5.54 |
| Joint: v=5 lag=0 1.5×, ... | +5.51 |
| Joint: v=5 1.0×, v=4 lag-conditioned | +5.72 |
| Multiplicative vote × lag | +5.46 |
| Conservative: only v=5 lag=0 | +2.47 |

### Verdict
**Lag does NOT add information beyond vote-strength** — closest competitor
joint scheme (+5.72) doesn't beat AGGRESSIVE quadratic (+5.74). The DQN's
lag pattern is correlated with vote consensus; the two signals are partially
redundant. Keep R1 AGGRESSIVE; don't complicate with lag.

---

## Step 2 — Action-space cleanup

### Setup
Test runtime ablation of S12 and/or S6 by masking those actions in
`valid_actions` and re-running walk-forward.

### Results

| Ablation | trades | sized WF | Δ |
|---|---:|---:|---:|
| BASELINE (no mask) | 1394 | **+5.74** | — |
| ablate S12 | 1394 | +5.74 | **0.00** |
| ablate S6 | 1408 | +4.95 | **−0.79** |
| ablate S6 + S12 | 1408 | +4.95 | −0.79 |

### Verdicts

**S12_VWAPVol**: confirmed dead. Zero impact when masked because the DQN
already never selected it (consistent with A1 audit finding S12 is short-only
on val/test and fires only 6 bursts total). Safe to permanently remove in
next state-pack rebuild (state v11 candidate).

**S6_TwoSignal**: my Step-2b hypothesis was WRONG. Despite 0.42% hit rate
(only 15 trades over 3,545 bursts), masking S6 forces the DQN to suboptimal
substitutes. Trade count rises 1394 → 1408 but Sharpe drops 0.79. **S6's
rare selections are high-quality — KEEP it.**

This is a useful negative result. Low hit rate ≠ dead weight when the rare
hits are critical.

---

## Step 3 — Maker-on-entry feasibility

### Hypothesis
Signal-latency showed the DQN's mean entry lag is 2.62 bars (median 0).
S1_VolDir specifically has median lag 3, max 53. For trades the policy
"could afford to wait on," we can try a passive limit order at the entry
price and fall back to taker only if not filled within N bars. Saves 2.5bp
on the maker fills.

### Method — snapshot-based fill proxy
For each trade at decision_bar t with direction d:
- "Filled as maker" within N bars iff:
  - Long (d=+1): `prices[t+k] <= prices[t]` for some k in [1, N] (market came down to our bid)
  - Short (d=−1): `prices[t+k] >= prices[t]` for some k (market came up to our ask)
- If filled: entry_fee = MAKER (2bp) instead of TAKER (4.5bp)
- If not filled: fall back to taker (current behavior)
- All trades execute eventually — this is "try maker, fall back to taker"

### Per-strategy fill rates

| strategy | trades | N=1 | N=3 | N=5 | N=10 |
|---|---:|---:|---:|---:|---:|
| S1_VolDir | 421 | 44.9% | 65.1% | 73.6% | 80.8% |
| S4_MACD | 43 | 67.4% | 74.4% | 81.4% | 88.4% |
| S10_Squeeze | 161 | 59.0% | 75.8% | 80.7% | 88.8% |
| S11_Basis | 219 | 55.7% | 74.0% | 80.4% | 86.3% |
| S8_TakerSus | 168 | 57.1% | 70.8% | 76.8% | 83.3% |
| S7_OIDiverg | 260 | 53.1% | 68.5% | 76.2% | 80.8% |
| (others) | — | 40-67% | 53-77% | 69-77% | 77-84% |

70-80% fill rate at N=5 across all strategies (snapshot proxy).

### Adverse selection check

Filled-subset vs unfilled-subset mean PnL%:

| strategy | filled% | mean filled PnL% | mean unfilled PnL% | adverse? |
|---|---:|---:|---:|---|
| S1_VolDir | 73.6% | +0.102 | +0.437 | YES (4× worse on filled) |
| S4_MACD | 81.4% | +0.598 | +1.438 | YES (2.4× worse) |
| S8_TakerSus | 76.8% | +0.010 | +0.395 | YES |
| S10_Squeeze | 80.7% | −0.036 | +0.179 | YES |
| S6_TwoSignal | 73.3% | +0.382 | +0.763 | YES |
| S2_Funding | 69.2% | −0.000 | +0.103 | YES |
| S13_OBDiv | 68.8% | −0.052 | +0.055 | YES |
| S3_BBExt | 77.4% | −0.121 | −0.092 | no |
| S7_OIDiverg | 76.2% | −0.044 | −0.037 | no |
| S11_Basis | 80.4% | +0.038 | +0.048 | no |

**Adverse selection is real on 7 of 10 strategies**, often substantial
(filled subset 2-4× worse than unfilled). The price coming back to our
limit DOES correlate with our trade signal being less reliable.

**But — fee savings dominate.** The trades still happen (maker fee on
filled, taker fee on missed), so we don't skip the better unfilled trades.
We just save 2.5bp on the filled ones.

### Sharpe under maker-on-entry

| scheme | WF | Δ vs baseline |
|---|---:|---:|
| BASELINE uniform 4.5bp/4.5bp | +4.58 | — |
| + TP-exit maker (P1 revised) | +4.62 | +0.04 |
| **+ maker-on-entry ALL (N=1)** | **+5.37** | **+0.79** |
| + maker-on-entry ALL (N=3) | +5.62 | +1.04 |
| + maker-on-entry ALL (N=5) | +5.72 | +1.14 |
| + maker-on-entry ALL (N=10) | +5.81 | +1.23 |
| + maker-on-entry S1_VolDir only (N=5) | +4.92 | +0.34 |

**S1-only is much weaker than all-strategies** — fee savings spread across
the portfolio, not concentrated in momentum.

### Full stack: AGGRESSIVE sizing + maker-on-entry

| scheme | WF | Δ vs baseline |
|---|---:|---:|
| AGGRESSIVE sizing + maker-on-entry (N=1) | **+6.22** | **+1.64** |
| AGGRESSIVE sizing + maker-on-entry (N=3) | +6.38 | +1.80 |
| AGGRESSIVE sizing + maker-on-entry (N=5) | +6.44 | +1.86 |
| AGGRESSIVE sizing + maker-on-entry (N=10) | +6.48 | +1.90 |

The two improvements stack near-additively (R1 +1.16 + maker +1.14 expected
~+2.30; actual +1.86 — some overlap but mostly additive).

### Caveats

1. **Snapshot-based fill estimation is loose** — real fill rates need
   intra-minute OB. True rates could be 30-50% lower (lots of bars touch
   the price but our limit doesn't actually get filled due to queue
   position).
2. **N=10 patience is overoptimistic** — fill at t+N means trade entry at
   a different price than the simulator's t+1 assumption. Actual PnL
   would differ. **N=1 is the honest conservative estimate.**
3. **Adverse selection** is real but bounded by the "fall back to taker"
   safety net.
4. **Tick-level OB data needed** for production confidence on fill rates.

### Honest production estimate

Conservative (N=1, discount 30% for adverse-selection underestimation):
- R1 AGGRESSIVE sizing: +1.16 (robust, deployed)
- Maker-on-entry @ N=1: +0.79 → effective +0.55 after AS discount
- **Combined: +1.5-1.7 effective Sharpe lift**

Expected production WF at 4.5bp: **+6.0-6.3** (vs current frozen +4.58).

---

## Updated production recommendation

For OKX taker (4.5bp) deployment:

1. **Use VOTE5_v8_H256_DD K=5 ensemble** (frozen baseline)
2. **Apply AGGRESSIVE quadratic sizing** (size = ((votes−2)/3)² × base)
3. **Mask S12_VWAPVol** at runtime (zero impact, slight cleanup)
4. **Try maker-on-entry with N=1 patience** — place passive limit at
   decision-bar price; if not filled within 1 minute, fall back to market
5. **TP-exit maker** (always send TP as limit, not market) — free +0.04

**Expected production WF ~+6.0 vs current frozen +4.58 = +30% improvement.**

## Code touchpoints

- [models/vote_lag_joint.py](../../models/vote_lag_joint.py) — Step 1
- [models/action_ablate_s12_s6.py](../../models/action_ablate_s12_s6.py) — Step 2
- [models/maker_on_entry_eval.py](../../models/maker_on_entry_eval.py) — Step 3

## Outputs

- `cache/results/vote_lag_joint.json`
- `cache/results/action_ablate_s6_s12.json`
- `cache/results/maker_on_entry_eval.json`
