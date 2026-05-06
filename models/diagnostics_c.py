"""
Path 1c — 5-minute timeframe diagnostic.

Tests whether higher-timescale signals + slower trade frequency reduce fee
drag enough to flip strategies positive.

Approach (approximate but cheap):
  1. Subsample features parquet by every 5th row → 5-min cadence (~76,634 bars).
  2. Recompute ATR-30 target at 5-min cadence on 5-min prices.
  3. Retrain vol LightGBM on 5-min features.
  4. Reuse direction predictions (subsampled to every 5th value) — quick approx.
  5. Build strategy DataFrame, run walk-forward at 5-min cadence using 5-min
     prices for both signals AND trade exits (i.e. TP/SL touched at 5-min bar
     close). This is conservative; real 5-min trading would touch TP/SL at any
     1-min bar within the 5-min window.
  6. Compare to 1-min walk-forward baseline.

Run: python3 -m models.diagnostics_c [ticker]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from numba import njit
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader        import load_meta
from strategy.agent     import STRATEGIES
from models.grid_search import _exit_arrays, _sharpe
from models.walk_forward import _build_default_full_params, _fmt
from models.diagnostics_ab import _simulate_sequential_fee
from backtest.costs     import TAKER_FEE

CACHE        = ROOT / "cache"
WARMUP_5MIN  = 1440 // 5            # warmup in 5-min bars (288)
N_FOLDS      = 6


# ── retrain vol on 5-min ─────────────────────────────────────────────────────

def _retrain_vol_5min(X_5: np.ndarray, prices_5: np.ndarray,
                       train_start: int, train_end: int) -> dict:
    """Train LightGBM ATR-30 on 5-min cadence. Returns predictions for full series + median."""
    from numpy.lib.stride_tricks import sliding_window_view

    # ATR-30 at 5-min cadence: mean(|diff(price)|) over next 30 5-min bars
    n = len(prices_5)
    H = 30
    y = np.full(n, np.nan)
    if n > H:
        wins = sliding_window_view(prices_5[1:], H)
        y[: n - H] = np.mean(np.abs(np.diff(wins, axis=1)), axis=1)

    # train slice (with 5% holdout for early-stopping)
    tr_idx = np.arange(train_start, train_end)
    ok_tr  = ~np.isnan(y[tr_idx])
    X_tr   = X_5[tr_idx][ok_tr]
    y_tr   = y[tr_idx][ok_tr]

    n_es  = max(50, int(len(X_tr) * 0.05))
    X_fit = X_tr[:-n_es]; y_fit = y_tr[:-n_es]
    X_es  = X_tr[-n_es:]; y_es  = y_tr[-n_es:]

    sc = StandardScaler(); sc.fit(X_fit)
    Xfs  = sc.transform(X_fit)
    Xes  = sc.transform(X_es)
    Xall = sc.transform(X_5)

    params = {
        "objective":        "regression",
        "metric":           "rmse",
        "boosting_type":    "gbdt",
        "num_leaves":       64,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "learning_rate":    0.05,
        "verbosity":        -1,
    }
    ds_fit = lgb.Dataset(Xfs, label=y_fit)
    ds_es  = lgb.Dataset(Xes, label=y_es, reference=ds_fit)
    model  = lgb.train(params, ds_fit, num_boost_round=500, valid_sets=[ds_es],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                    lgb.log_evaluation(-1)])

    pred_full = model.predict(Xall).astype(np.float32)
    pred_tr   = model.predict(sc.transform(X_tr)).astype(np.float32)

    # OOS spearman: rest of series outside train
    oos_mask = np.zeros(n, dtype=bool)
    oos_mask[train_end:] = True
    oos_mask &= ~np.isnan(y)
    sp_oos = spearmanr(pred_full[oos_mask], y[oos_mask]).statistic if oos_mask.sum() > 100 else float("nan")
    sp_tr  = spearmanr(pred_tr, y_tr).statistic

    sorted_tr = np.sort(pred_tr)
    rank_full = np.clip(np.searchsorted(sorted_tr, pred_full) / len(sorted_tr), 0, 1).astype(np.float32)
    median_tr = float(np.median(pred_tr))

    return dict(pred=pred_full, rank=rank_full, median=median_tr,
                spearman_in=sp_tr, spearman_oos=sp_oos)


# ── strategy DataFrame builder (5-min) ──────────────────────────────────────

_STRAT_COLS = [
    "bb_pct_b", "bb_width", "rsi_6", "rsi_14", "macd_hist",
    "ofi_perp_10_r15", "ofi_perp_10", "taker_imb_5", "taker_net_15",
    "fund_rate", "fund_mom_480", "ret_sma_200", "vwap_dev_1440",
    "sma_50", "sma_200",
    "oi_price_div_15", "taker_net_30", "taker_net_60",
    "taker_imb_30", "ret_15", "vwap_dev_240",
    "vol_z_spot_60", "spot_imbalance", "perp_imbalance",
    "spot_large_bid_count", "spot_large_ask_count",
    "perp_large_bid_count", "perp_large_ask_count",
    "diff_price",
]


def _build_df_5min(pq_5: pd.DataFrame, meta_5: pd.DataFrame,
                    price_5, atr_5, rank_5, dir_preds_5: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "price":    price_5,
        "atr_pred": atr_5,
        "vol_pred": rank_5,
    })
    for c in _STRAT_COLS:
        if c in pq_5.columns:
            df[c] = pq_5[c].values
        elif c in meta_5.columns:
            df[c] = meta_5[c].values
        else:
            df[c] = 0.0
    df["p_up_60"]   = dir_preds_5["up_60"]
    df["p_dn_60"]   = dir_preds_5["down_60"]
    df["p_up_100"]  = dir_preds_5["up_100"]
    df["p_dn_100"]  = dir_preds_5["down_100"]
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*78}\n  PATH 1c — 5-MIN TIMEFRAME DIAGNOSTIC  ({ticker.upper()})\n{'='*78}")

    # ── load 1-min source ────────────────────────────────────────────────────
    pq_1   = pd.read_parquet(CACHE / f"{ticker}_features_assembled.parquet")
    meta_1 = load_meta(ticker)
    assert (pq_1["timestamp"].values == meta_1["timestamp"].values).all()
    print(f"  1-min bars: {len(pq_1):,}")

    # subsample by 5
    pq_5   = pq_1.iloc[::5].reset_index(drop=True)
    meta_5 = meta_1.iloc[::5].reset_index(drop=True)
    price_5 = meta_5["perp_ask_price"].values.astype(np.float64)
    n_5    = len(pq_5)
    print(f"  5-min bars (every 5th): {n_5:,}")
    print(f"    span: {_fmt(pq_5['timestamp'].iloc[0])} → {_fmt(pq_5['timestamp'].iloc[-1])}")

    # ── train slices in 5-min space ──────────────────────────────────────────
    # Mirror 1-min structure: warmup 288 (=1440/5), vol-train next 20k bars,
    # RL period the rest.
    WARMUP_E   = WARMUP_5MIN                          # 288
    VOL_TR_E   = WARMUP_E + 100_000 // 5               # 20,288
    DQN_TR_E   = WARMUP_E + 280_000 // 5               # 56,288
    DQN_VAL_E  = WARMUP_E + 330_867 // 5               # 66,461
    print(f"  splits (5-min): warmup→{WARMUP_E}, vol-train→{VOL_TR_E}, "
          f"DQN-train→{DQN_TR_E}, DQN-val→{DQN_VAL_E}, end={n_5}")

    # ── retrain vol on 5-min ────────────────────────────────────────────────
    print(f"\n  Retraining vol LightGBM on 5-min ...")
    feat_cols_5 = [c for c in pq_5.columns if c != "timestamp"]
    X_5 = pq_5[feat_cols_5].values.astype(np.float32)
    # NaN handling: fill with 0 (warmup rows)
    X_5 = np.nan_to_num(X_5, nan=0.0)

    t1 = time.perf_counter()
    vol_5 = _retrain_vol_5min(X_5, price_5, train_start=WARMUP_E, train_end=VOL_TR_E)
    print(f"    fit done in {time.perf_counter()-t1:.1f}s  "
          f"Spearman in-sample={vol_5['spearman_in']:+.3f}  OOS={vol_5['spearman_oos']:+.3f}")

    atr_5  = vol_5["pred"]
    rank_5 = vol_5["rank"]
    atr_med_5 = vol_5["median"]

    # ── subsample direction preds (approximate) ──────────────────────────────
    print(f"\n  Subsampling direction preds (every 5th value) — approximate ...")
    dir_5 = {}
    for col in ["up_60", "down_60", "up_100", "down_100"]:
        d_full = np.load(CACHE / f"{ticker}_pred_dir_{col}_v4.npz")["preds"]   # length 383174 (post-warmup)
        # full 1-min array length = len(pq_1) - 1440
        # we need 5-min predictions aligned with pq_5 (which spans full series w/ subsample)
        # pq_5 row i corresponds to pq_1 row i*5. dir preds are aligned to pq_1 from row 1440 onward.
        # So dir_5[i] = dir_full[i*5 - 1440] for i*5 >= 1440 (else 0.5).
        d5 = np.full(n_5, 0.5, dtype=np.float32)
        for i in range(n_5):
            j = i * 5 - 1440
            if 0 <= j < len(d_full):
                d5[i] = d_full[j]
        dir_5[col] = d5

    # ── build strategy DataFrame ─────────────────────────────────────────────
    df_full_5 = _build_df_5min(pq_5, meta_5, price_5, atr_5, rank_5, dir_5)

    # ── fold boundaries (within RL period: WARMUP_E + 100k/5 → end) ─────────
    rl_start = VOL_TR_E
    rl_end   = n_5
    fold_size = (rl_end - rl_start) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        a = rl_start + i * fold_size
        b = rl_start + (i + 1) * fold_size if i < N_FOLDS - 1 else rl_end
        folds.append((a, b))

    print(f"\n  RL folds (5-min):")
    ts_5 = pq_5["timestamp"].values
    for i, (a, b) in enumerate(folds):
        print(f"    fold {i+1}: bars [{a:>5,}, {b:>5,})  "
              f"{_fmt(ts_5[a])} → {_fmt(ts_5[b-1])}  ({b-a:,} 5-min bars)")

    # ── jit warmup ───────────────────────────────────────────────────────────
    _ = _simulate_sequential_fee(
        np.zeros(20, dtype=np.int8), price_5[:20],
        np.full(20, 0.02, dtype=np.float32), np.full(20, 0.005, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.float32),
        np.zeros(20, dtype=np.float32), np.zeros(20, dtype=np.int32),
        TAKER_FEE,
    )

    # ── walk-forward (default params, two fee modes) ─────────────────────────
    strats = ["S1_VolDir", "S4_MACDTrend", "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow"]
    rows = []
    print(f"\n  Running walk-forward at 5-min cadence ...")
    for strat_key in strats:
        fn, _ = STRATEGIES[strat_key]
        params = _build_default_full_params(strat_key)
        for fee_label, fee_val in [("with_fee", TAKER_FEE), ("fee_free", 0.0)]:
            for i, (a, b) in enumerate(folds):
                df_fold    = df_full_5.iloc[a:b].reset_index(drop=True)
                price_fold = price_5[a:b]
                atr_fold   = atr_5[a:b]
                n_fold     = b - a

                sigs, _, _ = fn(df_fold, params)
                sigs       = np.asarray(sigs, dtype=np.int8)

                # NB: time_stop_bars and breakeven_pct stay numeric, but
                # interpretation now is "in 5-min bars" (e.g. ts=60 → 5h)
                tp, sl, tr, tab, be, ts_bars = _exit_arrays(
                    atr_fold,
                    params["base_tp_pct"], params["base_sl_pct"], atr_med_5,
                    params["breakeven_pct"], params["time_stop_bars"],
                    params["trail_after_breakeven"],
                )
                pnls, _ = _simulate_sequential_fee(
                    sigs, price_fold, tp, sl, tr, tab, be, ts_bars, fee_val)

                # Sharpe annualization for 5-min bars: 525,960/5 = 105,192 bars/yr
                if len(pnls) == 0:
                    sharpe = 0.0
                else:
                    rets = np.zeros(n_fold, dtype=np.float64)
                    rets[:min(len(pnls), n_fold)] = pnls[:n_fold]
                    if rets.std() < 1e-12:
                        sharpe = 0.0
                    else:
                        sharpe = float(rets.mean() / rets.std() * np.sqrt(105_192))
                rows.append(dict(
                    strategy=strat_key, fee_mode=fee_label, fold=i + 1,
                    n_trades=len(pnls), sharpe=sharpe,
                    win_rate=float((pnls > 0).mean()) if len(pnls) else 0.0,
                    total_pnl=float(pnls.sum()) if len(pnls) else 0.0,
                ))

    df = pd.DataFrame(rows)
    df.to_parquet(CACHE / f"{ticker}_diag_1c_5min.parquet", index=False)

    # ── summary tables ──────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  5-MIN WALK-FORWARD — Sharpe per fold\n{'='*78}")
    for fee_mode in ["with_fee", "fee_free"]:
        print(f"\n  ── {fee_mode} ──")
        print(f"  {'strategy':<14} " + " ".join(f"{f'fold{i+1}':>8}" for i in range(N_FOLDS))
              + f"  {'mean':>7}  {'std':>5}  pos/{N_FOLDS}  {'mean tr':>8}")
        print("  " + "─" * 96)
        for strat_key in strats:
            sub = df[(df["strategy"] == strat_key) & (df["fee_mode"] == fee_mode)].sort_values("fold")
            if len(sub) == 0:
                continue
            sharps = sub["sharpe"].values
            mean   = float(sharps.mean())
            std    = float(sharps.std())
            n_pos  = int((sharps > 0).sum())
            n_tr_mean = sub["n_trades"].mean()
            stable = "★" if (n_pos >= N_FOLDS // 2 + 1 and mean > 0) else " "
            line   = f"  {strat_key:<14} " + " ".join(f"{s:>+8.2f}" for s in sharps)
            line  += f"  {mean:>+6.2f}  {std:>5.2f}  {n_pos}/{N_FOLDS} {stable}  {n_tr_mean:>7.0f}"
            print(line)

    # ── headline comparison vs 1-min ─────────────────────────────────────────
    grand_5min_with_fee = df[df["fee_mode"] == "with_fee"]["sharpe"].mean()
    grand_5min_no_fee   = df[df["fee_mode"] == "fee_free"]["sharpe"].mean()
    print(f"\n  Grand-mean 5-min Sharpe:")
    print(f"    with-fee  = {grand_5min_with_fee:+.2f}    (1-min baseline: -10.09)")
    print(f"    fee-free  = {grand_5min_no_fee:+.2f}      (1-min baseline:  +2.31)")

    if grand_5min_with_fee > 0.5:
        print(f"\n  → STRONG SIGNAL: 5-min cadence flips strategies positive even WITH fees.")
        print(f"     Higher timeframe is the answer.")
    elif grand_5min_with_fee > grand_5min_with_fee + 5:  # always false; placeholder
        pass
    else:
        delta = grand_5min_with_fee - (-10.09)
        print(f"\n  → 5-min improvement vs 1-min: {delta:+.2f} Sharpe. "
              f"{'meaningful' if delta > 3 else 'modest' if delta > 1 else 'minimal'}.")

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
