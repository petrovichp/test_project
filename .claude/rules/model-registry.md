# Model registry rules

Every trained model must be registered in [`model_registry.json`](../../model_registry.json) and documented before the training session is considered complete.

## When this rule fires

- **Every** new policy / network weight saved to `cache/btc_dqn_policy_*.pt` or `cache/*_lgbm_*.txt` or any other model artifact.
- This includes one-off seeds inside an ensemble (each seed gets its own entry — they are separate model instances).
- Re-training with new hyperparameters under the same tag → overwrite the existing entry.

## What to update

### 1. `model_registry.json`

Add one entry per model file. Schema differs by `model_type`:

**DQN / Dueling / Double-Dueling**
```json
"BASELINE_FULL": {
  "ticker": "btc",
  "model_type": "dqn",
  "tag": "BASELINE_FULL",
  "seed": 42,
  "algo": "dqn",
  "hidden": 64,
  "state_version": "v5",
  "fee_train": 0.0,
  "trade_penalty": 0.001,
  "ablate_actions": [],
  "ensemble_role": "BASELINE_VOTE5_seed42",
  "val_sharpe": 7.30,
  "test_sharpe": 4.19,
  "wf_mean_sharpe": 9.03,
  "wf_folds_positive": 6,
  "n_trades_test": 174,
  "model_file": "btc_dqn_policy_BASELINE_FULL.pt",
  "history_file": "btc_dqn_train_history_BASELINE_FULL.json",
  "trained_at": "2026-05-07",
  "notes": "Single-seed reference; member of BASELINE_VOTE5"
}
```

**LGBM (existing schema, keep as-is for those entries):**
```json
"btc_lgbm_atr_30": {
  "ticker": "btc",
  "model_type": "lgbm",
  "target": "atr",
  "horizon": 30,
  "val_spearman": 0.627,
  "test_spearman": 0.801,
  "val_dir_acc": 0.756,
  "test_dir_acc": 0.824,
  "model_file": "btc_lgbm_atr_30.txt"
}
```

**Voting ensembles** (registered as a virtual entry pointing to its members):
```json
"BASELINE_VOTE5": {
  "ticker": "btc",
  "model_type": "ensemble",
  "ensemble_mode": "plurality",
  "members": ["BASELINE_FULL", "BASELINE_FULL_seed7", "BASELINE_FULL_seed123",
              "BASELINE_FULL_seed0", "BASELINE_FULL_seed99"],
  "val_sharpe": 3.53,
  "test_sharpe": 4.19,
  "wf_mean_sharpe": 10.40,
  "wf_folds_positive": 6,
  "n_trades_test": 174,
  "trained_at": "2026-05-08",
  "notes": "K=5 plurality ensemble; tie → NO_TRADE"
}
```

Use the metrics at the same fee setting as the model was trained at (typically fee=0). If results at additional fee levels are reported (e.g. fee=0.00045), put them under `extra_evals`:
```json
"extra_evals": {
  "fee_4.5bp": { "val_sharpe": -10.99, "test_sharpe": -5.94, "wf_mean_sharpe": 1.10 }
}
```

### 2. Documentation

- If the model is part of an existing experiment family (e.g. another seed for an existing baseline), add a row to that experiment's existing doc (`docs/experiments/voting_ensemble.md`, `docs/experiments/seed_variance.md`, etc.).
- If the model is a new family / new hyperparameter / new architecture, create a fresh `docs/{experiment_name}.md` per the [experiments rule](experiments.md).
- Cross-link the new doc from `CLAUDE.md`'s top-of-file doc list.

### 3. Commit

Bundle the model registry update + the doc + any code changes in **one commit**. Do not commit the cache `.pt` file (it's gitignored) — only the registry entry and doc.

Commit message body must include the new entry's headline metrics:
```
WF +9.03 / val +7.30 / test +4.19 / 6/6 folds positive
```

## What if the model is a one-off / throwaway?

There are no throwaway trained models. If it was worth training, it's worth registering — even negative results. Future sessions need to find them when proposing similar experiments.

If a model is genuinely temporary (e.g. a debugging run), delete the `.pt` file before ending the session and don't register.

## Backfill

The registry currently contains only LGBM entries. **The DQN baselines and recent fee-aware retrains are not yet registered.** When this rule is first applied, backfill:

| Tag family | Models to register | Source for metrics |
|---|---|---|
| `BASELINE_FULL` + seeds | 5 entries (42, 7, 123, 0, 99) | [docs/reference/baselines.md](../../docs/reference/baselines.md), [docs/experiments/seed_variance.md](../../docs/experiments/seed_variance.md) |
| `BASELINE_LEAN` | 1 entry | [docs/reference/baselines.md](../../docs/reference/baselines.md) |
| `BASELINE_VOTE5` (ensemble) | 1 virtual entry | [docs/experiments/voting_ensemble.md](../../docs/experiments/voting_ensemble.md) |
| `BASELINE_VOTE5_DISJOINT` (ensemble) + 5 members | 6 entries | [docs/experiments/voting_ensemble.md](../../docs/experiments/voting_ensemble.md) |
| `VOTE5_DOUBLE`, `VOTE5_DUELING`, `VOTE5_DD` (+ seeds) | 18 entries | [docs/experiments/algo_test.md](../../docs/experiments/algo_test.md) |
| `FEE4_p001`, `FEE4_p005` (+ seeds) | 10 entries | [docs/experiments/fee_aware_retrain.md](../../docs/experiments/fee_aware_retrain.md) |
| `BASELINE_FULL_h128`, `_h256` (+ seeds) | several | [docs/experiments/capacity_test.md](../../docs/experiments/capacity_test.md) |

Backfill can be a single dedicated commit titled `Backfill model_registry.json with all trained DQN policies`.
