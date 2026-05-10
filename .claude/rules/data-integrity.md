# Data integrity rules

These are non-negotiable. Violating any of them invalidates research results.

## No leakage

- Features may only use bars `[t − lookback, t]`. Never look forward.
- `shift(-n)` is only allowed when constructing **labels** (the prediction target), never features.
- Cross-asset features must lag the predictor asset by ≥1 bar.

## No random splits

- Time-series data is never shuffled. Sequential or walk-forward splits only.
- No `train_test_split` from sklearn. No `shuffle=True` in any DataLoader on time-ordered data.

## Embargo

- Leave a gap of `label_length` bars between train end and val/test start. This prevents label-window overlap from leaking train-set targets into the val period.

## Normalization

- Fit any scaler / mean / std on the train split only. Transform val/test using the train-fit statistics.
- Same rule applies to ATR median, vol percentiles, and any threshold derived from data.

## Test split is locked

- The test split is only touched **after** val tuning is frozen.
- Hyperparameter selection by test Sharpe is forbidden.
- This includes ensemble composition: do not pick which seeds go into VOTE5 by test Sharpe.

## Rolling windows

- Use `min_periods=full_window`. Early bars are NaN by design; exclude them from training samples.
- Don't backfill or forward-fill rolling-window NaNs.

## Cache rule

- Any computation > 10s must save to `cache/*.npz` or `cache/*.parquet`.
- Always check the cache path before recomputing.
- `cache/` is gitignored — never commit binary artifacts there.

## Raw data

- Never touch raw CSVs. Use the preprocessed parquets in `cache/`.
- Loader: `data/loader.py` — `load_meta(ticker)` or `load(ticker, include_ob=True)`.
