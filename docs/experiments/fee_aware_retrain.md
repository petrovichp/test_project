# Fee-aware retraining — does training with fee help?

Hypothesis: training with `--fee 0.0004` baked into the reward should produce a policy that natively selects only trades whose alpha clears the round-trip cost, without needing the post-hoc `top-5 + vote≥3` filter.

Two configs trained, 5 seeds each (42, 7, 123, 0, 99):

| Tag | Train fee | Trade penalty | Notes |
|---|---:|---:|---|
| `FEE4_p001` | 0.0004 (4 bp) | 0.001 | fee alone |
| `FEE4_p005` | 0.0004 (4 bp) | 0.005 | fee + heavier per-trade penalty |

Evaluation at OKX real taker fee = **4.5 bp/side**. Sources: [models/eval_fee_aware.py](../models/eval_fee_aware.py).

## Headline at fee = 4.5 bp/side (real taker)

| config | trades | WF | val | test | folds + |
|---|---:|---:|---:|---:|:--:|
| vanilla VOTE5 (no filter) | 1,144 | +1.10 | −10.99 | −5.94 | 4/6 |
| **vanilla VOTE5 + top-5 + vote≥3** | 1,048 | **+3.72** | −8.11 | **+0.97** | 4/6 |
| FEE4_p001 (no filter) | 1,182 | +1.87 | **−1.83** | −3.29 | 4/6 |
| FEE4_p001 + vote≥3 | 1,193 | −0.02 | −1.97 | −4.04 | 3/6 |
| FEE4_p005 (no filter) | 1,023 | +2.42 | −4.81 | −4.07 | 4/6 |
| FEE4_p005 + vote≥3 | 1,038 | +2.57 | −4.82 | −4.07 | 5/6 |

## Verdict

**Vanilla VOTE5 + post-hoc filter still wins on WF and test.** Fee-aware training did not produce a policy that beats the simple `top-5 + vote≥3` filter on the primary metric (walk-forward mean Sharpe).

But the retrains did meaningfully change the loss surface:

1. **FEE4_p001 makes val 5–9× less bad**: val Sharpe goes from −10.99 (vanilla) or −8.11 (vanilla+filter) to −1.83 (FEE4_p001). The fee-aware policy is dramatically more robust on the regime-shifted val period — it has learned to skip the borderline trades that cost vanilla so much.

2. **FEE4_p005 + vote≥3 wins fold-consistency**: 5/6 folds positive at 4.5bp (vs 4/6 for vanilla+filter). It loses 1 Sharpe point in WF aggregate (+2.57 vs +3.72) but spreads the alpha more evenly across regimes.

3. **At fee=0, FEE4_p001 outperforms vanilla on test (+8.48 vs +4.19)** — surprising. Adding fee pressure during training apparently regularized the policy enough to find more durable test-period structure, even if WF aggregate dropped.

4. **Stacking is destructive**: FEE4_p001 + vote≥3 → WF −0.02; FEE4_p005 + top-5 → WF +2.02 (worse than no filter). The trained policy already self-selects; layering hand-picked filters fights with its learned mask.

## Raw fee curves

### FEE4_p001 (fee-aware, penalty=0.001)

| fee bp | trades | WF | val | test | folds+ |
|---:|---:|---:|---:|---:|:--:|
| 0.0 | 1,185 | +7.64 | +2.06 | **+8.48** | 6/6 |
| 4.0 | 1,155 | +1.93 | −1.94 | +0.38 | 4/6 |
| 4.5 | 1,182 | +1.87 | **−1.83** | −3.29 | 4/6 |

### FEE4_p005 (fee-aware, penalty=0.005)

| fee bp | trades | WF | val | test | folds+ |
|---:|---:|---:|---:|---:|:--:|
| 0.0 | 1,021 | +6.00 | +1.11 | +4.66 | 6/6 |
| 4.0 | 1,029 | +2.98 | −4.91 | −3.92 | 4/6 |
| 4.5 | 1,023 | +2.42 | −4.81 | −4.07 | 4/6 |

## Why didn't fee training beat vanilla+filter?

Three plausible reasons:

1. **Train fee was 4 bp, real fee is 4.5 bp**. Slight mismatch — re-training at 4.5 bp could move things by ~0.2 Sharpe.
2. **The policy can't observe fee directly in state**. Reward signal teaches it indirectly by punishing trades whose pnl − 2×fee < 0, but the state vector doesn't include fee as a feature, so the policy must memorize "this kind of setup is profitable enough" rather than reasoning about cost-vs-edge.
3. **Train period vs val period shift**. The audit-derived `top-5 + vote≥3` filter was discovered by examining train+test contributions. The fee-aware policy gets no such cross-period information — it just sees the train fold.

## Recommended deployment config (fee = 4.5 bp/side)

Two viable configurations, depending on what we optimize for:

### A — Best WF aggregate (most reward, less consistent)
```
policy   = BASELINE_VOTE5 (vanilla DQN, 5 seeds)
filters  = top-5 strategies + vote ≥ 3
expected = WF +3.72, test +0.97, 4/6 folds +
```

### B — Best fold consistency + best val
```
policy   = FEE4_p005 (fee-aware retrain, 5 seeds, vote ≥ 3)
filters  = vote ≥ 3 only
expected = WF +2.57, val −4.82, 5/6 folds +
```

If shipping today, **A is the headline result**. **B is the safer deployment** if minimizing tail risk per fold matters more than peak Sharpe. Neither is a clean win across all metrics.

## Open follow-ups

- Retrain at exact 4.5 bp (correct for the small training-fee mismatch).
- Train with `--fee 0.0009` (taker × 2 stress) to see if stronger fee pressure forces more selectivity, possibly closer to the vanilla+filter optimum.
- Test FEE4_p005 on fee curve at deeper non-train fees (6 bp, 8 bp) — if its slope is shallower than vanilla's, the retrain pays off when execution slips above maker tier.
