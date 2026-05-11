"""
Build v8 state arrays — adds S11_Basis + S13_OBDiv to the action space.

Differences from v5:
  - signals shape:        (n, 9)  → (n, 11)   [S11, S13 appended]
  - valid_actions shape:  (n, 10) → (n, 12)   [actions 10, 11 added]
  - state shape:          (n, 50) → (n, 52)   [strategy flags for S11, S13 appended at [50:52]]

Approach: load v5 state files (which already have all 9 existing strategy signals),
compute S11 + S13 signals on the same data, append, save as v8.

Run: python3 -m models.build_state_v8 [ticker]
"""
import json, pathlib, sys, time
import numpy as np
import pandas as pd

from data.loader import load_meta
from strategy.agent import strategy_11, strategy_13, DEFAULT_PARAMS

CACHE = pathlib.Path("cache")


def _build_strategy_df(price, atr, pq_use, meta_use):
    """Same shape as models/dqn_state.py:_build_strategy_df, minus the dir_preds
    (S11+S13 don't need direction probabilities)."""
    df = pd.DataFrame({
        "price":    price,
        "atr_pred": atr,
    })
    # columns needed by strategy_11 + strategy_13
    needed = ["diff_price", "fund_mom_480", "spot_imbalance", "perp_imbalance",
              "taker_imb_5"]
    for c in needed:
        if c in pq_use.columns:
            df[c] = pq_use[c].values
        elif c in meta_use.columns:
            df[c] = meta_use[c].values
        else:
            df[c] = 0.0
    return df


def main(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Build v8 state — append S11_Basis + S13_OBDiv  ({ticker.upper()})\n{'='*70}")

    # Load existing v5 state for all 3 splits
    splits = ["train", "val", "test"]
    data = {nm: np.load(CACHE / "state" / f"{ticker}_dqn_state_{nm}.npz") for nm in splits}
    sizes = [data[nm]["state"].shape[0] for nm in splits]
    print(f"  v5 sizes: train={sizes[0]:,}  val={sizes[1]:,}  test={sizes[2]:,}")

    # Load features parquet + meta — needed for S11/S13 signal computation
    print(f"  Loading features parquet + meta ...")
    pq   = pd.read_parquet(CACHE / "features" / f"{ticker}_features_assembled.parquet")
    meta = load_meta(ticker)
    ts_pq   = pq["timestamp"].values
    ts_meta = meta["timestamp"].values
    assert (ts_pq == ts_meta).all(), "timestamp misalignment"

    # Map each split's ts to indices in pq (which is full-data, no warmup applied)
    ts_to_idx = pd.Series(np.arange(len(ts_pq)), index=ts_pq).to_dict()

    # Compute S11+S13 signals over the full pq (so rolling windows have history)
    print(f"  Computing S11 + S13 signals over full {len(pq):,} bars ...")
    price_full = ((meta["spot_ask_price"] + meta["spot_bid_price"]) / 2).values.astype(np.float64)
    atr_full   = np.zeros(len(pq), dtype=np.float64)   # not needed by S11/S13 logic

    df_full = _build_strategy_df(price_full, atr_full, pq, meta)
    s11_signal, _, _ = strategy_11(df_full, DEFAULT_PARAMS["S11_Basis"])
    s13_signal, _, _ = strategy_13(df_full, DEFAULT_PARAMS["S13_OBDiv"])
    s11_signal = s11_signal.astype(np.int8)
    s13_signal = s13_signal.astype(np.int8)

    print(f"    S11_Basis  long={(s11_signal > 0).sum():,}  short={(s11_signal < 0).sum():,}  "
          f"total={(s11_signal != 0).sum():,}  ({(s11_signal != 0).mean()*100:.2f}%)")
    print(f"    S13_OBDiv  long={(s13_signal > 0).sum():,}  short={(s13_signal < 0).sum():,}  "
          f"total={(s13_signal != 0).sum():,}  ({(s13_signal != 0).mean()*100:.2f}%)")

    # Build each split
    for nm in splits:
        d = data[nm]
        ts_split = d["ts"]
        idx = np.array([ts_to_idx[t] for t in ts_split])

        # signals: 9 → 11 cols
        s11_split = s11_signal[idx]
        s13_split = s13_signal[idx]
        new_signals = np.concatenate([
            d["signals"],
            s11_split[:, None].astype(np.int8),
            s13_split[:, None].astype(np.int8),
        ], axis=1)

        # valid_actions: 10 → 12 cols
        s11_valid = (s11_split != 0)
        s13_valid = (s13_split != 0)
        new_valid = np.concatenate([
            d["valid_actions"],
            s11_valid[:, None],
            s13_valid[:, None],
        ], axis=1)

        # state: 50 → 52 dims (append S11+S13 flags at the end)
        new_state_extra = np.stack([
            s11_split.astype(np.float32),
            s13_split.astype(np.float32),
        ], axis=1)
        new_state = np.concatenate([d["state"], new_state_extra], axis=1)

        assert new_signals.shape == (sizes[splits.index(nm)], 11), new_signals.shape
        assert new_valid.shape   == (sizes[splits.index(nm)], 12), new_valid.shape
        assert new_state.shape   == (sizes[splits.index(nm)], 52), new_state.shape

        out = CACHE / "state" / f"{ticker}_dqn_state_{nm}_v8_s11s13.npz"
        np.savez(
            out,
            state         = new_state,
            valid_actions = new_valid,
            ts            = d["ts"],
            price         = d["price"],
            signals       = new_signals,
            atr           = d["atr"],
            rank          = d["rank"],
            regime_id     = d["regime_id"],
        )
        n_active = new_valid[:, 1:].any(axis=1).mean() * 100
        n_active_s11 = s11_valid.mean() * 100
        n_active_s13 = s13_valid.mean() * 100
        print(f"  {nm:<6}  state={new_state.shape}  signals={new_signals.shape}  "
              f"valid={new_valid.shape}  "
              f"any-strategy-active={n_active:.1f}%  S11={n_active_s11:.2f}%  S13={n_active_s13:.2f}%  "
              f"→ {out.name}")

    print(f"\n  Total time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "btc")
