# docs/ — categorized index

All experiment docs live in this flat folder so cross-links work without subfolder prefixes. This README groups them by purpose; the files themselves are next to it.

## 🎯 Active reference (read these first)

| doc | purpose |
|---|---|
| [baselines.md](baselines.md) | Current primary + alternative baselines, reproduction commands, performance tables |
| [development_plan.md](development_plan.md) | Live forward plan — Path Z (zero-fee, active) + Path F (non-zero-fee, parked) |
| [data_splits.md](data_splits.md) | Canonical split boundaries (vol-train, dir-train, dir-holdout, DQN-train/val/test) |
| [experiments_log.md](experiments_log.md) | Running log of completed experiments + decisions |

## 🆕 Latest experiments (2026-05-11)

| doc | verdict |
|---|---|
| [distill_vote5.md](distill_vote5.md) | **C2** — single-net distillation MIXED-WIN. Test +9.35 (project record), 1× inference. Disjoint-pool reproduces. |
| [cross_asset.md](cross_asset.md) | **Z2.1** — ETH/SOL MIXED-POSITIVE. WF +7.22 / +8.24 at ensemble level; per-seed greedy negative. |
| [path_a_c1_results.md](path_a_c1_results.md) | **Path A + C1** — A3 disjoint validation shows v8 partly seed-luck. C1 curriculum NEGATIVE. |
| [z2_z3_results.md](z2_z3_results.md) | **Z2/Z3 Steps 2-5** — v7_pa negative, v7_basis partial, **v8 promoted** (Step 4), v9 doesn't compose. |
| [z3_step1_killed_strategies.md](z3_step1_killed_strategies.md) | **Z3 Step 1** — diagnostic of "killed" strategies; kept S11/S13 for Step 4 |
| [z3_data_feasibility.md](z3_data_feasibility.md) | **Z3** — data feasibility check for proposed new strategies |

## 📊 Phase Z baselines (Z1)

| doc | verdict |
|---|---|
| [z1_results.md](z1_results.md) | **Z1** — H256_DD promoted (WF +11.05). H128 exposed as seed-luck. K=10 collapses. |

## 🔬 Architecture / algorithm experiments

| doc | verdict |
|---|---|
| [algo_test.md](algo_test.md) | DQN vs Double / Dueling / Double_Dueling — DD wins |
| [capacity_test.md](capacity_test.md) | h=64 / 128 / 256 sweep — H256 lifts WF |
| [state_v6_test.md](state_v6_test.md) | v6 state (direction probs) — NEGATIVE |
| [voting_ensemble.md](voting_ensemble.md) | K=5 plurality validation — structurally beneficial |
| [ensemble_baseline.md](ensemble_baseline.md) | Q-averaging vs plurality vote — plurality wins (no third-action drift) |
| [seed_variance.md](seed_variance.md) | Per-seed variance characterization (single-seed ±2.17) |
| [trade_quality_by_agreement.md](trade_quality_by_agreement.md) | Vote-strength filtering analysis |

## 🔍 Audit / diagnostic

| doc | purpose |
|---|---|
| [baseline_vote5_audit.md](baseline_vote5_audit.md) | VOTE5 (h=64 vanilla) trade-level audit |
| [vote5_dd_audit.md](vote5_dd_audit.md) | VOTE5_DD trade-level audit |
| [a2_rule_audit.md](a2_rule_audit.md) | A2 + rule-based exits deal-by-deal audit |
| [audit_followup_tests.md](audit_followup_tests.md) | 5 follow-up tests from a2_rule_audit (ablations, TP tightening, long-bias) |

## 💸 Fee experiments

| doc | verdict |
|---|---|
| [fee_sensitivity_vote5.md](fee_sensitivity_vote5.md) | DIAGNOSTIC — strategies die at 4.5 bp taker; breakeven ~3-5 bp |
| [fee_aware_retrain.md](fee_aware_retrain.md) | MIXED — `FEE4_p001` cuts val damage; `FEE4_p005` + vote≥3 wins fold consistency |
| [fee_improvement_proposals.md](fee_improvement_proposals.md) | PARKED — 7 prioritized ideas (maker-only, vote sizing, CVaR, etc.) |

## 📚 Historical / superseded

| doc | status |
|---|---|
| [next_steps.md](next_steps.md) | Superseded by [development_plan.md](development_plan.md). Kept for cross-link reference. |

## Doc-writing conventions

From [.claude/rules/experiments.md](../.claude/rules/experiments.md):

- Every meaningful experiment gets a `docs/{experiment_name}.md` with: what was tested, command(s), per-fold + aggregate table, and verdict.
- Cross-link new docs from [CLAUDE.md](../CLAUDE.md) and from this README.
- Always update [RESULTS.md](../RESULTS.md) status block in the same commit.
- One experiment = one commit (per [.claude/rules/git.md](../.claude/rules/git.md)).
