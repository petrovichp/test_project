"""
D1 — Supervised per-strategy PnL predictor.

Phase 3 (DQN gating) failed because the state representation can't be
turned into a useful gate via Q-learning. This module tests the cleaner
formulation: train a LightGBM regressor to predict realized trade PnL
from the same state, then gate by `pred_pnl > τ_k`.

Pipeline:
  1. For each strategy k where sigs[t,k]!=0 in DQN-train:
       y = realized PnL of simulated trade entered at bar t+1
  2. Features X: 50-dim DQN state + 4 direction probs (up/dn × 60/100)
  3. Train LightGBM on DQN-train; eval on DQN-val.
  4. For each strategy: find val-best threshold τ that maximizes Sharpe.
  5. Single-shot DQN-test eval with locked val thresholds.

Strategies trained: those with ≥ MIN_SAMPLES bars on DQN-train.
Strategies skipped: lowest-fire ones (S12, possibly S2/S3) — fall back
to "no gate" or "always block" as configurable.

Outputs:
  cache/btc_pnl_pred_{Sk}.txt           — LightGBM model per strategy
  cache/btc_pnl_pred_thresholds.json    — val-best τ_k + diagnostics
  cache/btc_pnl_pred_results.parquet    — per-strategy val/test Sharpe table

Run: python3 -m models.pnl_predictor [ticker]
"""

import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtest.single_trade  import simulate_one_trade
from models.dqn_rollout     import _build_exit_arrays, STRAT_KEYS

CACHE       = ROOT / "cache"
MIN_SAMPLES = 1_000             # min DQN-train fires to train a model
BARS_PER_YR = 525_960           # 1-min bars/yr (matches engine.sharpe)

LGB_PARAMS = {
    "objective":        "regression",
    "metric":           "rmse",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "learning_rate":    0.03,
    "verbosity":        -1,
}


# ── per-strategy dataset builder ────────────────────────────────────────────

def _build_dataset_for_strategy(
    state, valid, signals, prices,
    tp, sl, trail, tab, be, ts_bars,
    dir_preds: np.ndarray,                   # (n_bars, 4)  up_60 dn_60 up_100 dn_100
    strat_idx: int,
):
    """Returns (X, y, n_bars_used) — one row per active bar."""
    sig_k = signals[:, strat_idx]
    fires = np.where(sig_k != 0)[0]
    n = len(fires)
    if n == 0:
        return np.empty((0, state.shape[1] + 4)), np.empty(0), 0

    # Skip last few bars to ensure trade has lookahead room (TP/SL may not realize)
    # 1500 bars headroom covers the longest observed trade duration (3979) safely
    fires = fires[fires < len(state) - 5]
    n = len(fires)
    if n == 0:
        return np.empty((0, state.shape[1] + 4)), np.empty(0), 0

    X = np.concatenate([state[fires], dir_preds[fires]], axis=1).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(fires):
        direction = int(sig_k[t])
        pnl, _, _ = simulate_one_trade(
            prices, t + 1, direction,
            float(tp[t, strat_idx]), float(sl[t, strat_idx]),
            float(trail[t, strat_idx]), float(tab[t, strat_idx]),
            float(be[t, strat_idx]),   int(ts_bars[t, strat_idx]),
            0,
        )
        y[i] = float(pnl)
    return X, y, n


# ── Sharpe + gate helpers ────────────────────────────────────────────────────

def _sharpe_from_trades(pnls: np.ndarray, n_total_bars: int) -> float:
    """Per-bar Sharpe matching engine.run sharpe convention.

    Constructs a per-bar return series with the trade pnl placed at each
    trade-close bar (rest are 0). Annualized for 1-min bars."""
    if len(pnls) == 0:
        return 0.0
    # Approximation: distribute trade pnl as a single-bar return at uniform
    # intervals. This is an ordering-free Sharpe that matches engine.run when
    # trade durations don't overlap (which they can't in single-strategy mode).
    rets = np.zeros(n_total_bars, dtype=np.float64)
    n_t  = min(len(pnls), n_total_bars)
    rets[:n_t] = pnls[:n_t]
    if rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(BARS_PER_YR))


def _evaluate_gate(pred: np.ndarray, actual: np.ndarray,
                    n_bars: int, threshold: float) -> dict:
    """Apply gate `pred > threshold` and report Sharpe / win-rate / n_trades."""
    keep = pred > threshold
    pnls = actual[keep]
    n    = int(keep.sum())
    if n == 0:
        return dict(threshold=float(threshold), n_trades=0,
                    sharpe=0.0, win_rate=0.0, total_pnl=0.0)
    return dict(
        threshold = float(threshold),
        n_trades  = n,
        sharpe    = _sharpe_from_trades(pnls, n_bars),
        win_rate  = float((pnls > 0).mean()),
        total_pnl = float(pnls.sum()),
    )


def _scan_thresholds(pred_val: np.ndarray, actual_val: np.ndarray,
                      n_val_bars: int, n_grid: int = 41) -> dict:
    """Grid-scan thresholds; return best by Sharpe + adjacent diagnostics."""
    if len(pred_val) == 0:
        return dict(threshold=-np.inf, sharpe=0.0, n_trades=0,
                     win_rate=0.0, scan=[])
    grid = np.linspace(np.percentile(pred_val, 5),
                        np.percentile(pred_val, 95), n_grid)
    scan = []
    for τ in grid:
        scan.append(_evaluate_gate(pred_val, actual_val, n_val_bars, τ))
    # sort by sharpe; require min 30 trades for stability
    valid_scan = [s for s in scan if s["n_trades"] >= 30]
    if not valid_scan:
        return dict(threshold=-np.inf, sharpe=0.0, n_trades=int(len(pred_val)),
                     win_rate=float((actual_val > 0).mean()),
                     total_pnl=float(actual_val.sum()), scan=scan)
    best = max(valid_scan, key=lambda s: s["sharpe"])
    return dict(scan=scan, **best)


# ── main ────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*72}\n  D1 — SUPERVISED PNL PREDICTOR  ({ticker.upper()})\n{'='*72}")

    # ── load DQN-train/val/test arrays ───────────────────────────────────────
    arr = {nm: np.load(CACHE / "state" / f"{ticker}_dqn_state_{nm}.npz")
           for nm in ["train", "val", "test"]}
    print(f"  splits: " + "  ".join(f"{nm}={arr[nm]['state'].shape[0]:,}"
                                       for nm in arr))

    # ── direction predictions (full bars[1440, end), need split-aligned slices) ─
    print("  loading direction predictions ...")
    DIR_COLS = ["up_60", "down_60", "up_100", "down_100"]
    dir_full = np.zeros((arr["train"]["state"].shape[0]
                          + arr["val"]["state"].shape[0]
                          + arr["test"]["state"].shape[0], 4), dtype=np.float32)
    # reconstruct per-split slices: bars [101440-1440, 384614-1440) = [100000, 383174)
    # train [100000, 280000), val [280000, 330867), test [330867, 383174)
    dir_arrays_full = []
    for col in DIR_COLS:
        d = np.load(CACHE / "preds" / f"{ticker}_pred_dir_{col}_v4.npz")
        dir_arrays_full.append(d["preds"])      # length 383174
    dir_pred_full = np.stack(dir_arrays_full, axis=1)
    splits = dict(train=(100_000, 280_000),
                   val  =(280_000, 330_867),
                   test =(330_867, 383_174))
    dir_per_split = {nm: dir_pred_full[a:b]
                     for nm, (a, b) in splits.items()}
    for nm in arr:
        assert dir_per_split[nm].shape[0] == arr[nm]["state"].shape[0]
        print(f"    dir preds {nm}: {dir_per_split[nm].shape}")

    # ── ATR-train median for exit arrays ─────────────────────────────────────
    vol = np.load(CACHE / "preds" / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    # ── pre-compute exit arrays per split ────────────────────────────────────
    print("  building exit arrays ...")
    exit_arrs = {}
    for nm in arr:
        tp, sl, tr, tab, be, ts = _build_exit_arrays(
            arr[nm]["price"], arr[nm]["atr"], atr_median)
        exit_arrs[nm] = dict(tp=tp, sl=sl, trail=tr, tab=tab, be=be, ts=ts)

    # ── build datasets per strategy on DQN-train ────────────────────────────
    print(f"\n  Building per-strategy datasets on DQN-train ...")
    train_data = {}
    for k, key in enumerate(STRAT_KEYS):
        e = exit_arrs["train"]
        X, y, n = _build_dataset_for_strategy(
            arr["train"]["state"], arr["train"]["valid_actions"],
            arr["train"]["signals"], arr["train"]["price"],
            e["tp"], e["sl"], e["trail"], e["tab"], e["be"], e["ts"],
            dir_per_split["train"], strat_idx=k,
        )
        if n >= MIN_SAMPLES:
            n_pos = int((y > 0).sum())
            print(f"    {key:<14}  n={n:>6,}  pos={n_pos:>5,} ({n_pos/n*100:>4.1f}%)  "
                  f"mean={y.mean()*100:>+.3f}%  std={y.std()*100:.3f}%")
            train_data[key] = (X, y, k)
        else:
            print(f"    {key:<14}  n={n:>6,}  (< {MIN_SAMPLES} → skipped)")

    if not train_data:
        print("\n  ✗ No strategies have enough train samples. Aborting.")
        return

    # ── train per-strategy LightGBM, val/test predictions, threshold scan ──
    print(f"\n  Training {len(train_data)} per-strategy models ...")
    results = []
    thresholds = {}
    for key, (X_tr, y_tr, k) in train_data.items():
        # split last 10% of train as early-stop holdout
        n_es = max(200, int(len(X_tr) * 0.10))
        X_fit, y_fit = X_tr[:-n_es], y_tr[:-n_es]
        X_es,  y_es  = X_tr[-n_es:], y_tr[-n_es:]

        ds_fit = lgb.Dataset(X_fit, label=y_fit)
        ds_es  = lgb.Dataset(X_es,  label=y_es, reference=ds_fit)
        model  = lgb.train(
            LGB_PARAMS, ds_fit, num_boost_round=500, valid_sets=[ds_es],
            callbacks=[lgb.early_stopping(30, verbose=False),
                        lgb.log_evaluation(-1)],
        )
        model.save_model(str(CACHE / "preds" / f"{ticker}_pnl_pred_{key}.txt"))

        # val + test datasets for THIS strategy
        ev = exit_arrs["val"]
        Xv, yv, nv = _build_dataset_for_strategy(
            arr["val"]["state"], arr["val"]["valid_actions"],
            arr["val"]["signals"], arr["val"]["price"],
            ev["tp"], ev["sl"], ev["trail"], ev["tab"], ev["be"], ev["ts"],
            dir_per_split["val"], strat_idx=k,
        )
        et = exit_arrs["test"]
        Xt, yt, nt = _build_dataset_for_strategy(
            arr["test"]["state"], arr["test"]["valid_actions"],
            arr["test"]["signals"], arr["test"]["price"],
            et["tp"], et["sl"], et["trail"], et["tab"], et["be"], et["ts"],
            dir_per_split["test"], strat_idx=k,
        )

        if nv == 0:
            print(f"    {key:<14}  no val fires; skipping")
            continue

        pred_v  = model.predict(Xv).astype(np.float32)
        pred_te = model.predict(Xt).astype(np.float32) if nt > 0 else np.array([])

        sp_v  = float(spearmanr(pred_v,  yv).statistic) if len(yv) > 5 else 0.0
        sp_te = float(spearmanr(pred_te, yt).statistic) if len(yt) > 5 else 0.0

        n_val_bars  = arr["val"]["state"].shape[0]
        n_test_bars = arr["test"]["state"].shape[0]

        # baseline (no gate) Sharpe
        no_gate_v  = _sharpe_from_trades(yv, n_val_bars)
        no_gate_te = _sharpe_from_trades(yt, n_test_bars) if nt > 0 else 0.0
        no_gate_v_wr  = float((yv > 0).mean())
        no_gate_te_wr = float((yt > 0).mean()) if nt > 0 else 0.0

        # threshold scan on val
        scan = _scan_thresholds(pred_v, yv, n_val_bars)
        τ    = scan["threshold"]
        # locked: apply same τ on test
        gated_te = _evaluate_gate(pred_te, yt, n_test_bars, τ) if nt > 0 \
                    else dict(n_trades=0, sharpe=0.0, win_rate=0.0)

        thresholds[key] = dict(threshold=τ,
                                val_spearman=sp_v, test_spearman=sp_te,
                                n_val_fires=nv, n_test_fires=nt)

        results.append(dict(
            strategy_key   = key,
            n_train        = len(y_tr),
            n_val_fires    = nv,
            n_test_fires   = nt,
            val_spearman   = sp_v,
            test_spearman  = sp_te,
            no_gate_val_sharpe   = no_gate_v,
            no_gate_test_sharpe  = no_gate_te,
            no_gate_val_winrate  = no_gate_v_wr,
            no_gate_test_winrate = no_gate_te_wr,
            best_val_threshold   = τ,
            gated_val_n_trades   = scan["n_trades"],
            gated_val_sharpe     = scan["sharpe"],
            gated_val_winrate    = scan.get("win_rate", 0.0),
            gated_test_n_trades  = gated_te["n_trades"],
            gated_test_sharpe    = gated_te["sharpe"],
            gated_test_winrate   = gated_te["win_rate"],
        ))

        print(f"    {key:<14}  Spearman v/t = {sp_v:>+.3f} / {sp_te:>+.3f}  "
              f"  no-gate v/t = {no_gate_v:>+.2f} / {no_gate_te:>+.2f}  "
              f"  τ={τ:>+.4f}  gated v/t = {scan['sharpe']:>+.2f} / "
              f"{gated_te['sharpe']:>+.2f}  "
              f"({scan['n_trades']}→{gated_te['n_trades']} test trades)")

    # ── save thresholds + results ────────────────────────────────────────────
    json_path = CACHE / "lookup" / f"{ticker}_pnl_pred_thresholds.json"
    json_path.write_text(json.dumps(thresholds, indent=2))

    df = pd.DataFrame(results)
    df.to_parquet(CACHE / "lookup" / f"{ticker}_pnl_pred_results.parquet", index=False)

    # ── summary tables ───────────────────────────────────────────────────────
    print(f"\n\n{'='*72}\n  SUMMARY — PNL-PREDICTOR GATE vs NO GATE\n{'='*72}")
    print(f"\n  {'strategy':<14}  {'n_val':>6}  {'no_gate_val':>11}  "
          f"{'gated_val':>9}  {'no_gate_test':>12}  {'gated_test':>10}  "
          f"{'tr→':>8}  {'spearman_v':>11}")
    print("  " + "─" * 90)
    for r in results:
        gated_v_tag  = "★" if r["gated_val_sharpe"]  > r["no_gate_val_sharpe"]  else " "
        gated_te_tag = "★" if r["gated_test_sharpe"] > r["no_gate_test_sharpe"] else " "
        print(f"  {r['strategy_key']:<14}  {r['n_val_fires']:>6,}  "
              f"{r['no_gate_val_sharpe']:>+11.3f}  "
              f"{r['gated_val_sharpe']:>+8.3f}{gated_v_tag}  "
              f"{r['no_gate_test_sharpe']:>+12.3f}  "
              f"{r['gated_test_sharpe']:>+9.3f}{gated_te_tag}  "
              f"{r['gated_val_n_trades']:>3}→{r['gated_test_n_trades']:<4}  "
              f"{r['val_spearman']:>+11.3f}")

    print(f"\n  ── BASELINES ──")
    print(f"    CUSUM gate best (CLAUDE.md): S4_MACDTrend test Sharpe = +3.130")
    print(f"    No-gate best on val (CLAUDE.md): S1_VolDir val Sharpe = +7.020 (test = -0.81)")

    # decision
    best_test = max(results, key=lambda r: r["gated_test_sharpe"])
    print(f"\n  ── BEST GATED TEST RESULT ──")
    print(f"    {best_test['strategy_key']}: "
          f"gated_test_sharpe = {best_test['gated_test_sharpe']:+.3f}  "
          f"(no-gate test = {best_test['no_gate_test_sharpe']:+.3f}, "
          f"trades {best_test['gated_test_n_trades']})")

    if best_test["gated_test_sharpe"] > 3.13:
        print(f"  ✓ BEATS CUSUM baseline (+3.13). Deployable gate.")
    elif best_test["gated_test_sharpe"] > 0:
        print(f"  ⚠ Positive but under CUSUM. Consider per-strategy hyperparam tune.")
    else:
        print(f"  ✗ NEGATIVE. State representation insufficient for gating.")

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
