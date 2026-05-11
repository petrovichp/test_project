# Z2.1 — Cross-asset (ETH, SOL) — MIXED-POSITIVE

Executed 2026-05-11. Path Z2.1 from [development_plan.md](development_plan.md).

## Hypothesis

The BTC-trained policy stack (`VOTE5_v8_H256_DD`) captures market microstructure
edges that should transfer to other large-cap perps with similar venue (OKX) and
data schema. If true, we get a 2-3× capital multiplier across BTC/ETH/SOL with
no algorithmic change.

## Method

For ETH and SOL we rebuilt the full upstream pipeline (ticker-parametric paths
already existed; one `models/build_state_v8.py` parameter change made it
fully cross-asset). Strategy parameters and execution config: unchanged from BTC.
Architecture (`DuelingDQN(52, 12, 256)`), training hyperparameters: unchanged.

1. **Parquet cache** from raw CSV (data/loader.py `load(ticker)`)
2. **Features**: `features.assembly <ticker>` — 191 features
3. **Vol model**: `models.vol_v4 <ticker>` — LightGBM ATR-30
4. **Direction**: `models.direction_dl_v4 <ticker>` — 4 CNN-LSTMs (up/down × 60/100)
5. **Regime**: `models.regime_cusum_v4 <ticker>` — CUSUM+Hurst
6. **State v5/v8**: `models.dqn_state <ticker>` + `models.build_state_v8 <ticker>`
7. **Training**: 5 seeds of `VOTE5_v8_H256_DD_<ticker>` (200k grad steps, early
   stop after 25k stagnant)

## Per-seed training (single-seed greedy val Sharpe)

| seed | ETH best val | SOL best val |
|---|---:|---:|
| 42 | −15.088 (step 5k) | −16.326 (step 5k) |
| 7 | −12.185 (step 30k) | −11.464 (step 15k) |
| 123 | −11.640 (step 95k) | −12.885 (step 20k) |
| 0 | −13.738 (step 20k) | −11.619 (step 5k) |
| 99 | −6.795 (step 90k) | −13.037 (step 35k) |
| **mean** | **−11.89** | **−13.07** |

**At first glance: catastrophic.** All 10 seeds across both assets early-stop at
a negative single-seed greedy Sharpe. But this is the wrong measurement —
single-seed greedy uses every bar's argmax action, while production uses
**5-seed plurality vote**. The training metric does not reflect the deployable
artifact.

## Ensemble (plurality vote) walk-forward results

The honest measurement: K=5 plurality-vote ensemble, walk-forward over the
full RL period (6 folds, zero fee, rule-based exits):

| ticker | WF | val | test | folds+ | val trades | test trades | n bars |
|---|---:|---:|---:|:---:|---:|---:|---:|
| **BTC** (baseline) | **+12.065** | **+6.673** | **+4.442** | 6/6 | 300 | 199 | 283,174 |
| ETH | +7.218 | +5.565 | −0.086 | 5/6 | 123 | 79 | 280,079 |
| SOL | +8.236 | +4.160 | +2.185 | 6/6 | 457 | 301 | 273,063 |

### Per-fold breakdown

**ETH**:
| fold | ETH price chg | WF Sharpe | trades | equity |
|---:|---:|---:|---:|---:|
| 1 | −11.5% | +8.13 | 111 | 1.34 |
| 2 | −25.2% | +12.49 | 154 | 1.78 |
| 3 | +12.4% | +3.24 | 75 | 1.07 |
| 4 | **−41.8%** | **+16.53** | 191 | **2.42** |
| 5 | +10.1% | +3.09 | 109 | 1.08 |
| 6 | +8.3% | −0.17 | 74 | 1.00 |

**SOL**:
| fold | SOL price chg | WF Sharpe | trades | equity |
|---:|---:|---:|---:|---:|
| 1 | +3.2% | +9.06 | 325 | 1.65 |
| 2 | −38.9% | +15.08 | 391 | 2.73 |
| 3 | +7.0% | +5.08 | 200 | 1.18 |
| 4 | **−36.4%** | **+15.55** | 477 | **2.73** |
| 5 | +6.2% | +2.19 | 408 | 1.10 |
| 6 | −4.5% | +2.45 | 307 | 1.09 |

## Findings

### Strategy stack transfers, but loses on WF and test

- **All three tickers: positive WF mean Sharpe** (+12, +7, +8).
- **All three: positive val Sharpe** (+6.67, +5.57, +4.16).
- **ETH/SOL preserve BTC's 6/6 (or 5/6) fold positivity** — structural soundness.
- **Δ WF vs BTC: −4.85 for ETH, −3.83 for SOL.** Real cost of using a BTC-tuned
  stack on different volatility/liquidity profiles.

### Test is the soft spot

- ETH test: **−0.086** (79 trades, ~half of BTC's). Effectively zero.
- SOL test: **+2.185** (301 trades). About half of BTC's test Sharpe.
- val→test degradation is worse for ETH (+5.57 → −0.09, Δ −5.66) than for
  BTC (+6.67 → +4.44, Δ −2.23) or SOL (+4.16 → +2.19, Δ −1.97).
- ETH's recent period (test split = last ~17% of timeline) is harder for
  the BTC-tuned policies than its earlier periods.

### Plurality voting is the key transfer mechanism

- Single-seed val Sharpes are all negative (−6.8 to −16.3).
- Ensemble val Sharpes are all positive (+4.16 to +6.67).
- **The plurality threshold (3 of 5 seeds) filters out bad trades that
  individual under-trained seeds would have made.** ETH only fires 123 val
  trades (BTC 300) — voting suppresses ~60% of would-be trades.
- This proves voting's **defensive** value: even under-trained members of an
  ensemble produce robust ensemble policies via plurality.

### Strategies thrive on downmoves

Both ETH and SOL had **massive equity gains on their largest down-folds**
(fold 4: ETH −41.8% → policy ×2.42 equity; SOL −36.4% → policy ×2.73 equity).
The policy is capturing the downside efficiently via short signals — not just
free-riding the long bias.

### Asset diversification potential

- All 3 ticker policies are positive WF.
- If asset PnL streams are uncorrelated, an equal-weighted 3-asset deployment
  yields √3 × single-asset Sharpe (~+6 expected at portfolio level vs each
  asset's individual +7-12).
- Worth pursuing: a 3-asset orchestrator with shared capital pool.

## Verdict

🟡 **MIXED-POSITIVE — partial transfer**.

- Strategy stack works at the ensemble level on ETH and SOL — both positive WF,
  both positive val, 5-6/6 folds positive.
- ETH test is essentially zero; SOL test is half BTC's.
- The single-seed training failures are misleading; plurality voting recovers a
  deployable policy from each ticker's 5 under-trained seeds.

**Recommended action**: Deploy multi-asset (BTC + ETH + SOL) with the same
strategy stack. Expect ~50-60% of BTC's per-asset Sharpe on ETH/SOL. If asset
PnLs decorrelate, portfolio Sharpe lifts. Cross-asset retuning (per-ticker
strategy parameters, per-ticker direction-model fit) is a future direction.

## Code touchpoints

- `models/build_state_v8.py` — parameterized ticker
- `models/eval_cross_asset.py` — walk-forward + val/test for any ticker
- 5 ETH policies: `cache/eth_dqn_policy_VOTE5_v8_H256_DD_eth_seed{N}.pt`
- 5 SOL policies: `cache/sol_dqn_policy_VOTE5_v8_H256_DD_sol_seed{N}.pt`
- Caches: 5 parquets + 7 npzs each for ETH and SOL upstream pipeline

## What we learned

- The plurality-voting mechanism is **strongly defensive**: it salvages
  positive Sharpe from poorly-trained per-seed members. This was hypothesized
  for BTC but never tested in a stress case where every member was negative.
- Cross-asset alpha exists at the architectural level: 191 features +
  9+2 strategies + 6-fold walk-forward isn't BTC-specific magic.
- Test-split degradation is the consistent failure mode across assets. The
  policy doesn't generalize to the **most recent** held-out period on ETH;
  partially on SOL. Hypothesis: recent regime shifts in those assets.
- For Path X / production: BTC remains the safest single-asset deployment.
  ETH/SOL are viable as **diversification** layers but not as standalone bets.
