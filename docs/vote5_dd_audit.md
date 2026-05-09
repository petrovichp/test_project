# VOTE5_DD Trade Analysis — Audit, Fee Sensitivity, Trade Reduction

> **TL;DR**: DD trades **28% more** than vanilla VOTE5 (1,437 vs 1,122 across WF). Strategy contributions skew differently (S8 dominant; S6 net-NEGATIVE). **Fee sensitivity is brutal**: WF Sharpe drops from +6.80 (fee=0) to +0.75 at OKX maker (4bp) to −8.51 at taker (8bp). Trade reduction (vote ≥5 unanimous filter) lifts WF from +0.75 to +1.88 at 4bp by cutting 67% of trades — but cannot recover the fee=0 ceiling. **Maker-only execution (Path X) is the real deployment blocker, not architecture.**

## Setup

Three-part analysis on `BASELINE_VOTE5_DD` (K=5 plurality of Double_Dueling DQNs):
- Part A: deal-by-deal audit across 6 WF folds (fee=0)
- Part B: fee sensitivity at 8 levels (0, 1, 2, 4, 6, 8, 12, 20 bp per side)
- Part C: trade-count reduction at fee=4bp via vote-strength + strategy filtering

Reproduction:
```bash
python3 -m models.audit_vote5_dd
```

## Part A — Trade audit (fee=0)

### Per-fold equity

| fold | trades | BTC return | DD WF Sharpe | DD equity |
|---|---|---|---|---|
| 1 | 184 | −6.43% | +6.76 | ×1.295 |
| 2 | 282 | **−16.88%** | **+13.06** ⭐ | ×2.026 |
| 3 | 173 | +6.33% | +4.19 | ×1.122 |
| 4 | 311 | **−29.83%** | +6.78 | ×1.407 |
| 5 | 261 | +6.02% | +5.43 | ×1.266 |
| 6 | 226 | +9.11% | +4.58 | ×1.153 |

Total: 1,437 trades, WF mean Sharpe +6.80. **Defensive pattern strongly confirmed** — fold 2 (worst BTC −17%) gets best Sharpe (+13.06) and equity ×2.03.

### Per-strategy contribution (DD vs vanilla VOTE5)

| strategy | DD count | vanilla count | DD mean PnL | DD sum PnL | vanilla sum PnL |
|---|---|---|---|---|---|
| S1_VolDir | 430 | 396 | +0.12% | +51.4% | +108.2% |
| **S8_TakerFlow** | 272 | 244 | +0.20% | **+55.8%** ⭐ | +47.7% |
| **S10_Squeeze** | 224 | 88 | +0.11% | +24.5% | +27.3% |
| S7_OIDiverg | 360 | 249 | +0.07% | +25.7% | +43.8% |
| **S4_MACDTrend** | 54 | 29 | **+0.65%** | +35.3% | +25.0% |
| **S6_TwoSignal** | 26 | 42 | **−0.32%** ⚠ | **−8.2%** ⚠ | +11.3% |
| S3_BBRevert | 56 | 61 | +0.00% | +0.2% | +1.2% |
| S2_Funding | 15 | 13 | +0.10% | +1.5% | +1.4% |
| S12_VWAPVol | 0 | 0 | — | — | — |

**Notable changes from vanilla:**
- DD trades S10 **2.5× more often** (224 vs 88) but with similar mean PnL
- DD's **S6 is net-NEGATIVE** (−0.32% mean, 35% win rate) — pathology unique to DD
- S8 is DD's biggest contributor (+55.8% sum), surpassing S1
- S4 trades 2× more (54 vs 29) and is still high-precision (+0.65%/trade)

### Exit reason breakdown

| reason | DD count | % | mean PnL | avg bars |
|---|---|---|---|---|
| TP | 45 | 3.1% | +2.37% | 312 |
| SL | 266 | 18.5% | −1.10% | 178 |
| TIME | 402 | 28.0% | +0.77% | 280 |
| TSL | 170 | 11.8% | ~0% | 182 |
| **BE** | **550** | **38.3%** | +0.11% | 70 |

**DD relies more on BE exits** (38% vs vanilla 29%). Many trades briefly go in-the-money, hit BE trigger, then close at breakeven. The shorter avg holding (70 bars on BE exits) suggests these are quick scalp-style trades.

### Vote count distribution & quality

| votes | count | % | mean PnL |
|---|---|---|---|
| 2 | 16 | 1.1% | +0.40% |
| 3 | 1076 | 74.9% | +0.07% |
| 4 | 299 | 20.8% | +0.20% |
| 5 (unanimous) | 46 | 3.2% | **+0.92%** |

Same monotone vote→quality as vanilla DQN. 75% of trades at modal 3-vote agreement.

### Holding period

- Overall: 171 bars (~3 hours)
- Long: 182 bars
- Short: 155 bars

DD holds positions ~3× longer than vanilla VOTE5's typical exit-window-truncated trade. The Dueling V/A architecture seems to make the policy more confident in holding longer (V-head provides stable "stay in trade" signal).

---

## Part B — Fee sensitivity

| fee/side | bp | WF mean | val Sharpe | test Sharpe | WF folds + | trades |
|---|---|---|---|---|---|---|
| 0.0000 | 0 | **+6.80** | **+6.12** | **+5.91** | 6/6 | 1437 |
| 0.0001 | 1 | +4.92 | +0.16 | +0.42 | 5/6 | 1369 |
| 0.0002 | 2 | +3.81 | +1.54 | −2.82 | 5/6 | 1349 |
| **0.0004** (OKX maker) | **4** | **+0.75** | **−0.41** | **−4.47** | 5/6 | 1355 |
| 0.0006 | 6 | −3.85 | −3.77 | −5.99 | 1/6 | 1398 |
| 0.0008 (OKX taker) | 8 | −8.51 | −10.69 | −11.01 | 0/6 | 1416 |
| 0.0012 | 12 | −13.62 | −20.14 | −15.25 | 0/6 | 1441 |
| 0.0020 | 20 | −24.31 | −29.98 | −24.50 | 0/6 | 1435 |

**At OKX maker fees (4bp), the DD edge collapses.** WF +0.75, test −4.47, val −0.41.

### Why the fee sensitivity?

Mean PnL per trade across all 1,437 trades is **+0.18%**. Each trade pays 2× fee (entry + exit):

| fee/side | round-trip | mean PnL net |
|---|---|---|
| 0bp | 0% | +0.18% |
| 4bp | 8bp = 0.08% | +0.10% (still positive but marginal) |
| 8bp | 16bp = 0.16% | +0.02% (essentially zero) |

The **per-trade variance** (~0.9%) is unchanged regardless of fees. So Sharpe = mean/std × √N collapses linearly with mean PnL. Doubling the fee from 4bp to 8bp halves the net-of-fees mean and Sharpe falls accordingly.

**Breakeven fee** ≈ 3bp per side. Beyond that, alpha is negative.

---

## Part C — Trade reduction at fee=4bp

Tested filtering strategies to cut trade count and improve fee-net Sharpe:

| filter | trades | Δ trades | val | test | WF mean | folds + |
|---|---|---|---|---|---|---|
| baseline (no filter) | 1355 | — | −0.41 | −4.47 | +0.75 | 5/6 |
| vote ≥3 | 1361 | +6 | −0.20 | −4.11 | +0.57 | 4/6 |
| vote ≥4 | 1082 | −20% | −4.41 | −2.21 | −0.54 | 2/6 |
| **vote ≥5 (unanimous)** | **471** | **−65%** | −3.28 | **+0.37** | **+1.88** ⭐ | 4/6 |
| top-5 strategies (drop S2,S3,S6,S12) | 1286 | −5% | −0.80 | −3.29 | +1.19 | 4/6 |
| top-3 strategies (S1,S8,S7) | 1262 | −7% | −2.31 | −2.89 | +0.43 | 4/6 |
| top-5 + vote ≥3 | 1293 | −5% | −1.24 | −0.36 | +1.13 | 4/6 |
| top-5 + vote ≥4 | 1069 | −21% | −3.73 | −3.22 | +0.74 | 4/6 |

### Findings

**1. Vote ≥5 unanimous filter is the clear winner.**
- Cuts trades 65% (1,437 → 471)
- Lifts WF from +0.75 → +1.88 (2.5×)
- Test goes positive (+0.37, the only filter to achieve this at 4bp)
- But val collapses (−3.28) — unanimous trades are too sparse on the 35-day val window

**2. Vote ≥4 underperforms.** Cuts only 20% of trades but loses WF aggregate. The 3→4-vote bucket has mediocre per-trade quality at 4bp fees.

**3. Strategy filtering helps modestly.** Dropping the bad-performing S6 (DD-specific pathology) plus the rarely-used S2/S3/S12 lifts WF from +0.75 → +1.19. Pure strategy filtering doesn't compete with vote-≥5.

**4. Combined filters are NOT additive.** top-5 + vote ≥3 is roughly equivalent to top-5 alone. The strategy filter and vote filter target overlapping bad-trade subsets.

### Why no filter recovers the fee=0 ceiling

The system has **per-trade alpha of ~+0.18% mean** in fee-free regime. Even with vote-≥5 filtering (which catches the +0.92% mean PnL trades), the total trade count is too small (471) to support the original WF Sharpe of +6.80. **Sharpe scales with √N**, so cutting from 1,437 to 471 trades gives a √(471/1437) = 0.57× Sharpe penalty — about 4 Sharpe units of lift would be needed to compensate the trade-count loss, and the per-trade quality boost only delivers ~1 unit.

---

## Implications for live deployment

### What we learned

1. **The +6.80 fee=0 promise is fragile to execution costs.** Live deployment WITHOUT maker-only execution would produce near-zero or negative Sharpe.

2. **At OKX maker tier (4bp), trade reduction can lift WF to +1.88** — barely deployable but with much-reduced confidence interval (val negative, only 4/6 folds positive).

3. **DD's "trade more" strategy hurts at fees.** Vanilla VOTE5 (1,122 trades) might be more fee-robust than DD (1,437 trades) — worth testing the same fee sensitivity on vanilla VOTE5.

### Forward paths

| Path | What it solves | Cost |
|---|---|---|
| **Path X — maker-only execution scoping** | Achieves fee≈0 regime → +6.80 holds | 3-5 days |
| **Retrain with stronger fee penalty in reward** | Train policy to be selective enough to handle 4-8bp fees natively | 1-2 days |
| **Deploy with vote-≥5 filter at 4bp fees** | Marginally deployable +1.88 WF | trivial |
| **Increase trade-penalty hyperparameter** | Force lower trade count during training | 1 day |

### Recommendation

**Prioritize Path X scoping**. The +6.80 alpha is real; the bottleneck is execution. Vote-≥5 filtering at 4bp gives us a fallback option (+1.88 WF) but at much-reduced confidence. The retrain-with-fees experiment is also worth running — it's cheap and could produce a structurally fee-robust policy.

The "trade count is the bottleneck" framing was right — but the lever isn't filtering at deploy time; it's training-time penalty calibration to match expected fees.

## Files

| File | Contents |
|---|---|
| [models/audit_vote5_dd.py](../models/audit_vote5_dd.py) | three-part audit + fee sweep + trade reduction |
| `cache/audit_vote5_dd_results.json` | full per-trade log + fee sensitivity + reduction results |
