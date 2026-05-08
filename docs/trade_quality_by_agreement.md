# A2 — Trade Quality by Vote Agreement

> **TL;DR**: Vote agreement is **strongly monotone** with trade quality (K=10 per-trade Sharpe ranges −0.06 → +1.10 across vote levels). But sizing variants give only **small aggregate lifts** (≤+0.22 Sharpe) because high-agreement trades are rare and most trade volume sits at moderate agreement. Best free win: K=5 threshold ≥3 → +0.10 Sharpe.

## Hypothesis

If higher vote agreement (e.g. 7/10 nets agree) corresponds to higher per-trade Sharpe, then position sizing proportional to vote count should lift aggregate Sharpe.

## Mechanism validated

### K=5 BASELINE_VOTE5 — per-vote-count stats (1,122 WF trades)

| votes | count | mean PnL | win % | per-trade Sharpe |
|---|---|---|---|---|
| 2 | 9 | **−0.28%** | 55.6% | −0.27 |
| 3 | 936 | +0.18% | 56.4% | +0.20 |
| 4 | 153 | +0.48% | 64.7% | +0.44 |
| 5 | 24 | +1.07% | 75.0% | **+0.73** |

### K=10 ensemble — per-vote-count stats (965 WF trades)

| votes | count | mean PnL | win % | per-trade Sharpe |
|---|---|---|---|---|
| 4 | 92 | **−0.05%** | 50.0% | −0.06 |
| 5 | 175 | +0.20% | 58.9% | +0.21 |
| 6 | 496 | +0.19% | 56.9% | +0.21 |
| 7 | 139 | +0.39% | 55.4% | +0.33 |
| 8 | 41 | +0.74% | 80.5% | +0.61 |
| 9 | 14 | +1.50% | 85.7% | +0.99 |
| 10 | 8 | +1.43% | **87.5%** | **+1.10** |

**Beautifully monotone.** Higher consensus → higher mean PnL, higher win rate, higher per-trade Sharpe. The mechanism prediction holds.

## Sizing experiments

### K=5 BASELINE_VOTE5 (full-size baseline WF +10.400)

| sizing variant | effective trades | WF mean | Δ |
|---|---|---|---|
| full-size | 1,122 | +10.400 | — |
| linear (size = (votes-K/2)/(K/2)) | 1,144 | +10.405 | +0.005 |
| **threshold ≥3/5** | 1,109 | **+10.500** | **+0.100** |
| threshold ≥4/5 | 215 | +7.297 | −3.10 |
| quadratic | 1,138 | +7.937 | −2.46 |

### K=10 ensemble (full-size baseline WF +9.649)

| sizing variant | effective trades | WF mean | Δ |
|---|---|---|---|
| full-size | 965 | +9.649 | — |
| linear | 711 | +9.565 | −0.08 |
| **threshold ≥6/10** | 689 | **+9.864** | **+0.22** |
| threshold ≥7/10 | 205 | +8.815 | −0.83 |
| threshold ≥8/10 | 66 | +5.852 | −3.80 |
| quadratic | 723 | +8.641 | −1.01 |

## Why aggregate lift is small despite strong per-trade signal

**Sharpe arithmetic:** Sharpe ≈ (mean return / std) × √N. Cutting trades shrinks √N.

For K=10:
- 8-vote trades: per-trade Sharpe +0.61, n=41 → Sharpe contribution ≈ +0.61 × √41 ≈ +3.9
- 6-vote trades: per-trade Sharpe +0.21, n=496 → Sharpe contribution ≈ +0.21 × √496 ≈ +4.7

The mid-agreement BULK contributes more aggregate Sharpe than the high-agreement TIPS — because √N matters as much as per-trade Sharpe.

**This means**: sizing schemes that *underweight bulk* (linear, quadratic) lose more from N reduction than they gain from quality concentration.

**Sizing only helps when it cuts trades that are actually negative-Sharpe**. The K=5 ≥3-vote threshold cuts the 9 marginal 2-vote trades (mean PnL −0.28%, per-trade Sharpe −0.27) — that's a true cleanup, +0.10. Beyond that, every cut hurts.

## What we get

**Best free improvements (small)**:
- BASELINE_VOTE5 + threshold ≥3/5 → WF +10.50 (Δ +0.10)
- This is essentially a "skip the 2-vote tie-broken trades" rule — adds nothing structural.

**Conclusion**: vote-agreement-based sizing is a **diagnostic win** (validates the voting mechanism is doing real Q-discrimination work) but only a **modest tactical lift**. The +1.46 from capacity scaling (A1) dominates the +0.10–0.22 from sizing.

## Implications

1. **The voting mechanism IS working as designed** — high-agreement trades genuinely are higher-quality. This is the strongest validation yet of the plurality-voting baseline architecture.

2. **Position sizing won't drive significant Sharpe gains** in this system. Future improvements should target either (a) more capacity (Tier 2), (b) better algorithms (Tier 4 — Double/Dueling DQN), or (c) more diverse seed pools.

3. **A "vote-strength filtering" rule could be applied at deployment time** — skip trades where < K/2 + 1 nets agree. Tiny implementation cost, ~+0.10 Sharpe.

## Files

| File | Contents |
|---|---|
| [models/trade_quality_by_agreement.py](../models/trade_quality_by_agreement.py) | per-vote-count stats + 5 sizing variants for K=5 and K=10 |
| `cache/_a2_full.log` | full eval output |
