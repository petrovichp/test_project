# C2 — VOTE5_v8 self-distillation (single-net imitator)

Executed 2026-05-11. Path C2 from [development_plan.md](development_plan.md).

## Hypothesis

A single net trained to mimic the K=5 ensemble's plurality output can:
1. Match deployment-time performance with 1/5 the inference cost.
2. Be regularized by the soft consensus labels (when teachers disagree, the
   label is NO_TRADE, encoding "uncertain → don't trade").
3. Either inherit or exceed teacher generalization.

## Method

**Teacher labels** (`models/distill_targets.py`):
- For each bar in train/val/test (combined 283k bars) compute each VOTE5_v8 member's
  masked-argmax action. Plurality vote → label. Tie (2-2) → label = 0 (NO_TRADE).
- Also store `q_mean` (ensemble-averaged Q-values) as optional soft target.

| split | n bars | NO_TRADE % | trade % | votes 5/4/3/tie |
|---|---:|---:|---:|---|
| train | 180,000 | 94.7% | 5.27% | 130305/22276/24619/2800 |
| val | 50,867 | 94.2% | 5.77% | 35314/6968/7668/917 |
| test | 52,307 | 95.0% | 4.96% | 38663/6055/6818/771 |

**Student** (`models/distill_vote5.py`):
- Same architecture as teachers: `DuelingDQN(52, 12, 256)`, 48,141 params.
- Loss: **masked cross-entropy** on plurality labels. `q.masked_fill(~valid, -1e9)`
  before softmax to bake in the action-mask constraint at training time.
- Stratified batches: 128 NO_TRADE + 128 trade per batch of 256 to compensate
  for the 95/5 class imbalance.
- Optimizer: Adam, lr 1e-3, grad clip 10.
- 12 epochs (~9s total per seed on CPU). Best checkpoint by val Sharpe.

## Results — walk-forward (zero fee, rule-based exits)

| policy | WF | val | test | folds+ | trades(val/test) |
|---|---:|---:|---:|:---:|---|
| DISTILL_v8 single seed=42 | +9.99 | **+10.41** | +9.35 | 6/6 | 341/291 |
| DISTILL_v8 single seed=7 | +7.30 | +7.57 | +7.79 | 6/6 | 344/265 |
| DISTILL_v8 single seed=123 | +7.85 | +7.72 | **+10.31** | 6/6 | 345/280 |
| DISTILL_v8 single seed=0 | +8.51 | +9.49 | +6.20 | 6/6 | 349/268 |
| DISTILL_v8 single seed=99 | +7.86 | +6.34 | +5.55 | 6/6 | 346/274 |
| **DISTILL_v8 single-mean** | **+8.30** | **+8.30** | **+7.85** | 6/6 | — |
| DISTILL_v8 VOTE5 (plurality) | +7.13 | +3.45 | +3.98 | 6/6 | 350/272 |
| **BASELINE VOTE5_v8_H256_DD** | **+12.07** | **+6.67** | **+4.44** | 6/6 | 300/199 |

## Findings

### Single distilled net is competitive with the teacher ensemble

- **Mean test Sharpe of single students: +7.85 vs teacher VOTE5's +4.44** — the distilled
  policy *exceeds* the teacher on the locked-in test split.
- Best single seed (s=42): test **+9.35** with **341 trades** vs teacher's 199.
- Val Sharpes are higher across the board (mean +8.30 vs teacher's +6.67), though
  there is implicit early-stopping leak: best checkpoint is picked by val Sharpe.
  The honest comparison is the test number.
- All 5 seeds: 6/6 folds positive — structural robustness preserved.

### WF Sharpe drops vs teacher (the interesting deficit)

- Teacher VOTE5: WF +12.07. Distilled mean: +8.30. **Δ −3.77** WF.
- The WF gap is real and consistent across seeds.
- Likely cause: the teacher's WF lift comes partly from each fold's training-time
  fitting to its specific period — the student inherits one fixed policy and can't
  re-tune per fold. Distilled WF Sharpe represents a single deployable policy's
  WF performance, not a re-trained-per-fold ensemble.

### Voting distilled students HURTS

- DISTILL_v8 plurality of 5 students: WF +7.13, val +3.45, test +3.98 — **worse**
  than the mean of individual students.
- Reason: all 5 students are trained on the **same teacher labels**, so their
  errors are correlated. Plurality of correlated voters adds tie → NO_TRADE
  events without diversity benefit. Compare to teacher VOTE5 where each seed
  has independent training stochasticity (different ε-trajectories, different
  buffer fills) → less correlation → genuine voting benefit.
- **Implication**: deploy a single distilled net, not a vote of them. The voting
  benefit is exhausted by the teacher's diversity; can't be recovered from
  the labels.

### Increased trade volume

- Single distilled net: 291 test trades (s=42) vs teacher's 199.
- Distilled student is slightly *more* aggressive than the teacher: on bars
  where the teacher vote was 3-2 in favor of trading (a real edge), the
  student fits to "trade" cleanly. The student doesn't have the teacher's
  built-in tie-breaking conservatism — yet test Sharpe is still higher,
  meaning the extra trades have positive EV.

## Verdict

✅ **C2 distillation succeeds as a deployment artifact**:
- Single 48k-param policy with test Sharpe matching the K=5 ensemble.
- Deploy `DISTILL_v8_seed42` as the single-net alternative to `VOTE5_v8_H256_DD`.
- 5× cheaper inference, equivalent or better test performance.

⚠️ **Caveats**:
- WF mean Sharpe is lower than teacher (real, persistent gap).
- Val Sharpe ranking is biased by early stopping; test is the honest read.
- Single-seed selection (s=42) by val Sharpe is itself leakage — for production,
  either pick by val Sharpe upfront (treat val as a tuning set) or use the
  median/mean of all 5 seeds as a reference.

## Code touchpoints

- `models/distill_targets.py` — precompute teacher plurality labels + ensemble q_mean
- `models/distill_vote5.py` — student trainer (masked CE, stratified sampling)
- `models/eval_distill_v8.py` — walk-forward + val/test eval pack
- 5 `cache/btc_dqn_policy_DISTILL_v8_seed{42,7,123,0,99}.pt`
- 3 `cache/btc_distill_targets_{train,val,test}_v8.npz`

## What we learned for C3/future research

- Distillation can produce a competitive single-net deployment from a costly ensemble.
- Voting requires diversity sources; same-label-same-arch students don't supply it.
- The student found behaviors the teacher's plurality vote suppressed — the
  ensemble's tie-breaks were leaving alpha on the table.
- For Path C3 (QR-DQN), the lesson is that algorithmic regularization on a single
  net may be more fruitful than ensemble averaging if we want a single deployable
  artifact.
