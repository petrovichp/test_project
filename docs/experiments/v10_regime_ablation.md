# v10 regime-ablation experiment — does the regime one-hot earn its place?

Executed 2026-05-13 as follow-up to the A2 audit ([regime_classifier_quality.md](../audits/regime_classifier_quality.md))
and its reanalysis (regime is "informative but largely redundant" with
vol_pred / atr_pred_norm / bb_width already in state).

**Verdict: NUANCED. Regime is NOT pure shortcut — it carries real value at
the ENSEMBLE level via seed-disagreement diversity. At the SINGLE-SEED level,
removing it forces more accurate individual policies, but distillation
doesn't transmit the gain. Z5.4 freeze remains the right call.**

## Hypothesis

A2's reanalysis showed:
- Regime one-hot at `state[:, 2:7]` is largely predictable from
  vol_pred (F=22,218), atr_pred_norm (F=13,645), bb_width (F=10,521).
- P(positive return | regime) is stable across splits (no real sign-flip).
- The "weak/non-stationary" verdict was overstated.

Three possible outcomes pre-experiment:

| Outcome | Interpretation |
|---|---|
| WF within ±0.5 | regime is pure shortcut (redundant); drop it |
| WF drops >1.0 | regime carries real value beyond shortcut |
| WF improves >0.5 | regime is distraction; removing it helps |

## Setup

Built v10 state-pack ([models/build_state_v10_regimeoff.py](../../models/build_state_v10_regimeoff.py))
by copying v8_s11s13 cached state files and zeroing dims [2:7]. Identical
otherwise (same 52 dims, same 12 actions, same signals/prices/atr).

Trained 5 seeds of `VOTE5_v10_H256_DD` with same recipe as v8:
- `--algo double_dueling --hidden 256 --state-version v10_regimeoff`
- fee=0.0, trade_penalty=0.001
- 5 seeds {42, 7, 123, 0, 99}, ~2 min/seed wall

Then trained `DISTILL_v10_seed42` on the v10 K=5 plurality labels
(same recipe as `DISTILL_v8_seed42`).

## Per-seed training-best val Sharpes

| seed | v10 val (training-best) | v8 val (training-best, ref) |
|---|---:|---:|
| 42 | +6.71 | +7.30 |
| 7 | +4.41 | +5.71 |
| 123 | +6.71 | +5.55 |
| 0 | +5.56 | +5.21 |
| 99 | +7.83 | +7.78 |
| **mean** | **+6.24** | +6.31 |

Comparable training-time vals. The interesting differences emerge at the
walk-forward eval.

## Walk-forward results

| policy | WF | val | test | folds+ | per-fold |
|---|---:|---:|---:|---:|---|
| VOTE5_v10 single s=42 | +9.91 | +6.71 | **+6.81** | 6/6 | [14.85, 12.43, 6.32, 10.61, 8.70, 6.56] |
| VOTE5_v10 single s=7 | +10.81 | +4.41 | +7.10 | 6/6 | [13.10, 16.75, 9.77, 13.73, 4.77, 6.75] |
| VOTE5_v10 single s=123 | +8.38 | +6.71 | **+8.90** | 6/6 | [7.30, 15.21, 7.21, 5.45, 6.51, 8.61] |
| VOTE5_v10 single s=0 | +8.98 | +5.56 | +4.73 | 6/6 | [11.49, 15.82, 7.73, 12.03, 4.60, 2.24] |
| VOTE5_v10 single s=99 | +11.09 | +7.83 | +5.94 | 6/6 | [11.62, 18.87, 9.88, 12.71, 7.85, 5.57] |
| **VOTE5_v10 single MEAN** | **+9.83** | +6.24 | **+6.69** | 6/6 | — |
| **VOTE5_v10 K=5 plurality** | +10.75 | **+0.85** | +5.55 | **5/6** | [15.59, 15.24, 9.64, 19.08, **−0.50**, 5.46] |
| BASELINE VOTE5_v8 single MEAN (Z5.2) | +8.33 | +6.31 | +4.39 | 6/6 | — |
| BASELINE VOTE5_v8_H256_DD K=5 | **+12.07** | +6.67 | +4.44 | **6/6** | [11.14, 19.44, 13.08, 18.01, 6.29, 4.44] |
| **DISTILL_v10_seed42** | +8.71 | +7.88 | +6.14 | 6/6 | [11.53, 12.51, 8.29, 6.14, 9.04, 4.77] |
| DISTILL_v8_seed42 (reference) | **+9.99** | **+10.41** | **+9.35** | 6/6 | [11.31, 16.11, 7.57, 10.86, 5.61, 8.50] |

## Findings

### 1. Single-seed v10 policies are MEASURABLY BETTER than v8 single seeds

| metric | v10 single mean | v8 single mean | Δ |
|---|---:|---:|---:|
| WF | **+9.83** | +8.33 | **+1.50** |
| val | +6.24 | +6.31 | ≈ 0 |
| test | **+6.69** | +4.39 | **+2.30** |

Removing the redundant regime shortcut forces each seed to learn the
underlying vol/atr/bb_width patterns properly rather than relying on the
categorical bin. Individual policies become more capable.

### 2. v10 ensemble val Sharpe COLLAPSES from +6.67 to +0.85

The K=5 plurality vote on v10 loses the val cushion:
- v8 ensemble: WF +12.07, val +6.67, test +4.44, 6/6 folds
- v10 ensemble: WF +10.75 (Δ −1.31), val **+0.85** (Δ −5.82), test +5.55 (Δ +1.11), **5/6 folds** (fold-5 turns negative)

Mechanism (bias-variance shift): without regime as a context-bucket that
seeds use differently, seeds become more correlated → plurality vote loses
diversity benefit → tie inflation → less protection on tricky periods.
Fold-5 (which was +6.29 for v8 baseline) flipped to −0.50 — seeds
disagreed less, made similar wrong calls.

### 3. DISTILL_v10 is WORSE than DISTILL_v8 across the board

| metric | DISTILL_v10_seed42 | DISTILL_v8_seed42 | Δ |
|---|---:|---:|---:|
| WF | +8.71 | +9.99 | −1.28 |
| val | +7.88 | +10.41 | −2.53 |
| test | +6.14 | +9.35 | −3.21 |

The single v10 seeds are individually better than v8 seeds, but the
distillation chain didn't transmit that gain. The bottleneck is the
**label quality**: v10's K=5 plurality vote has val +0.85 (vs v8's +6.67),
so the labels themselves are weaker. Student trained on weaker labels
inherits weakness.

### 4. v10 K=5 ensemble has higher TEST Sharpe than v8 ensemble

| | v8 K=5 | v10 K=5 |
|---|---:|---:|
| test | +4.44 | **+5.55** |

Modest but real test-side improvement on the ensemble. Doesn't compensate
for val collapse, but worth noting — the v10 ensemble overfits val LESS.

## Interpretation

The hypothesis was binary: regime is either shortcut or substantive. The
real answer is: **regime serves a function the binary framing missed —
ensemble diversity injection.**

- At the single-seed level, regime IS shortcut (Δ +1.50 / +2.30 in WF/test
  when removed).
- At the ensemble level, regime PROVIDES seed-disagreement (Δ −1.31 / −5.82
  in WF/val when removed).
- These work in opposite directions; the net depends on whether you deploy
  single net or ensemble.

This is a clean **bias-variance tradeoff**:
- regime increases per-seed bias (shortcut) but contributes ensemble variance reduction
- removing regime reduces per-seed bias but increases ensemble correlation (less variance reduction)

## Implications for Z5.4 freeze

| Deployable use case | Decision unchanged? |
|---|---|
| fee=0 (maker-only) — DISTILL_v8_seed42 | **YES.** DISTILL_v10 is worse on all metrics. v10 single seeds are better but not deployed as single seeds (we deploy one trained net). |
| fee=4.5bp (taker) — VOTE5_v8_H256_DD K=5 | **YES.** v10 K=5 ensemble underperforms (WF -1.31, val collapse). |
| cross-asset — VOTE5_v8_H256_DD_{eth,sol} | YES — not affected. |

## Open questions surfaced

1. **Could we deploy a v10 single seed directly?** The best single (s=99 WF +11.09) is competitive with the v8 ensemble (+12.07) at 5× cheaper inference. Worth exploring if 1× inference is critical and we can tolerate single-seed variance.
2. **Could we distill from a single v10 seed instead of the K=5 plurality?** Skip the lossy plurality step. Student would be trained on cleaner labels from one strong teacher. ~10 min experiment.
3. **Hybrid state v11**: keep regime, add explicit interactions (e.g., vol × regime)? Probably not — A2 showed regime is already derivable from vol/atr.

## Code touchpoints

- `models/build_state_v10_regimeoff.py` — state-pack builder (zeros [2:7])
- `models/dqn_selector.py` — added `v10_regimeoff` to state-version choices
- `models/distill_targets_v10.py` — v10 plurality label generator
- `models/distill_vote5.py` — added `--state-suffix` flag

## Outputs

- `cache/state/btc_dqn_state_{train,val,test}_v10_regimeoff.npz`
- `cache/policies/btc_dqn_policy_VOTE5_v10_H256_DD_seed{N}.pt` (×5)
- `cache/policies/btc_dqn_policy_DISTILL_v10_seed42.pt`
- `cache/distill/btc_distill_targets_{train,val,test}_v10.npz`
- `cache/results/v10_regimeoff_eval.json`
- `cache/results/distill_v10_eval.json`
