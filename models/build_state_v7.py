"""
Build v7 state arrays by appending new features to existing v5 state.

Variants:
  pa:    +4 dims (price-action context: drawdown_60, runup_60, realized_vol_60, vol_ratio_30_60)
  basis: +5 dims (basis_z_60, basis_change_1bar, funding_apr, funding_z_120, oi_change_60)

Approach: load existing v5 train/val/test state files (which are temporally
contiguous), compute new features on the concatenated timeline (so rolling
windows respect causality across split boundaries), standardize using
train-slice statistics only, append to state, save as v7_{variant}.

Run: python3 -m models.build_state_v7 --variant pa
     python3 -m models.build_state_v7 --variant basis
"""
import argparse, json, pathlib, time
import numpy as np
import pandas as pd

from data.loader import load_meta

CACHE = pathlib.Path("cache")


def _med_iqr(x: np.ndarray) -> tuple[float, float]:
    """median + IQR for standardization. Returns (median, max(iqr, eps))."""
    m = float(np.nanmedian(x))
    q1, q3 = np.nanpercentile(x, [25, 75])
    return m, float(max(q3 - q1, 1e-12))


def _std(x: np.ndarray, m: float, iqr: float, clip: float = 10.0) -> np.ndarray:
    """median/IQR normalization with clip."""
    z = (x - m) / iqr
    return np.clip(z, -clip, clip).astype(np.float32)


def build_pa_features(price: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    """4 price-action context features. All derived from price + atr arrays."""
    n = len(price)
    s_price = pd.Series(price.astype(np.float64))
    s_atr   = pd.Series(atr.astype(np.float64))

    # drawdown_60 = 1 - now / max(price[-60:])
    max60 = s_price.rolling(60, min_periods=60).max()
    drawdown_60 = (1.0 - price / max60.values).astype(np.float32)
    drawdown_60 = np.nan_to_num(drawdown_60, nan=0.0)

    # runup_60 = now / min(price[-60:]) - 1
    min60 = s_price.rolling(60, min_periods=60).min()
    runup_60 = (price / min60.values - 1.0).astype(np.float32)
    runup_60 = np.nan_to_num(runup_60, nan=0.0)

    # realized_vol_60 = std(returns over last 60 bars) * sqrt(60)
    rets = np.diff(price, prepend=price[0]) / price
    realized_vol_60 = pd.Series(rets).rolling(60, min_periods=60).std().values * np.sqrt(60.0)
    realized_vol_60 = np.nan_to_num(realized_vol_60, nan=0.0).astype(np.float32)

    # vol_ratio_30_60 = atr_30_mean / median(atr_60)
    atr_30 = s_atr.rolling(30, min_periods=30).mean()
    atr_60_med = s_atr.rolling(60, min_periods=60).median()
    vol_ratio = (atr_30 / atr_60_med.replace(0, np.nan)).values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0, posinf=10.0, neginf=0.0).astype(np.float32)

    return {
        "drawdown_60":     drawdown_60,
        "runup_60":        runup_60,
        "realized_vol_60": realized_vol_60,
        "vol_ratio_30_60": vol_ratio,
    }


def build_basis_features(ts: np.ndarray) -> dict[str, np.ndarray]:
    """5 perp basis + funding features. Loaded from meta parquet via timestamps."""
    meta = load_meta("btc")
    ts_meta = meta["timestamp"].values

    # Index meta to align with the state ts array
    ts_to_idx = pd.Series(np.arange(len(ts_meta)), index=ts_meta).to_dict()
    idx = np.array([ts_to_idx[t] for t in ts])

    fund_rate = meta["fund_rate"].values[idx].astype(np.float64)
    oi_usd    = meta["oi_usd"].values[idx].astype(np.float64)
    spot_mid  = ((meta["spot_ask_price"] + meta["spot_bid_price"]) / 2).values[idx].astype(np.float64)
    perp_mid  = ((meta["perp_ask_price"] + meta["perp_bid_price"]) / 2).values[idx].astype(np.float64)

    # 1. basis_z_60: (basis - mean(basis_60)) / std(basis_60)
    basis_bps = (perp_mid - spot_mid) / spot_mid * 1e4
    s_basis = pd.Series(basis_bps)
    basis_mean = s_basis.rolling(60, min_periods=60).mean()
    basis_std  = s_basis.rolling(60, min_periods=60).std()
    basis_z = ((basis_bps - basis_mean.values) /
                np.where(basis_std.values > 1e-6, basis_std.values, 1e-6))
    basis_z = np.nan_to_num(basis_z, nan=0.0).astype(np.float32)

    # 2. basis_change_1bar: bps delta (basis[t] - basis[t-1])
    basis_change = np.diff(basis_bps, prepend=basis_bps[0]).astype(np.float32)

    # 3. funding_apr (annualized %)
    funding_apr = (fund_rate * (365 * 24 / 8) * 100).astype(np.float32)

    # 4. funding_z_120
    s_fund = pd.Series(fund_rate)
    fund_mean = s_fund.rolling(120, min_periods=120).mean()
    fund_std  = s_fund.rolling(120, min_periods=120).std()
    funding_z = ((fund_rate - fund_mean.values) /
                  np.where(fund_std.values > 1e-12, fund_std.values, 1e-12))
    funding_z = np.nan_to_num(funding_z, nan=0.0).astype(np.float32)

    # 5. oi_change_60: (oi[t] - oi[t-60]) / oi[t-60]
    s_oi = pd.Series(oi_usd)
    oi_lag60 = s_oi.shift(60).values
    oi_change = (oi_usd - oi_lag60) / np.where(oi_lag60 > 0, oi_lag60, 1.0)
    oi_change = np.nan_to_num(oi_change, nan=0.0).astype(np.float32)

    return {
        "basis_z_60":        basis_z,
        "basis_change_1bar": basis_change,
        "funding_apr":       funding_apr,
        "funding_z_120":     funding_z,
        "oi_change_60":      oi_change,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["pa", "basis"])
    args = ap.parse_args()
    t0 = time.perf_counter()

    print(f"\n{'='*70}\n  Build v7 state — variant={args.variant}\n{'='*70}")

    # ── load + concatenate v5 state ─────────────────────────────────────────
    splits = ["train", "val", "test"]
    data = {nm: np.load(CACHE / f"btc_dqn_state_{nm}.npz") for nm in splits}
    sizes = [data[nm]["state"].shape[0] for nm in splits]
    boundaries = np.cumsum([0] + sizes)   # e.g. [0, 180000, 230867, 283174]

    print(f"  split sizes: train={sizes[0]:,}  val={sizes[1]:,}  test={sizes[2]:,}")
    print(f"  boundaries:  {list(boundaries)}")

    ts_full     = np.concatenate([data[nm]["ts"]    for nm in splits])
    price_full  = np.concatenate([data[nm]["price"] for nm in splits])
    atr_full    = np.concatenate([data[nm]["atr"]   for nm in splits])

    # ── build new features ──────────────────────────────────────────────────
    print(f"  Computing {args.variant} features over {len(ts_full):,} contiguous bars ...")
    if args.variant == "pa":
        feats = build_pa_features(price_full, atr_full)
    else:
        feats = build_basis_features(ts_full)

    n_new = len(feats)
    feat_names = list(feats.keys())
    print(f"  built {n_new} new features: {feat_names}")

    # ── standardize using train-only slice ──────────────────────────────────
    train_end = boundaries[1]
    stats = {}
    for k, a in feats.items():
        # use first half of train (skip warmup bars where rolling windows are NaN-filled with 0)
        train_slice = a[1000:train_end]   # skip first 1000 bars (warmup safety)
        m, q = _med_iqr(train_slice)
        stats[k] = {"median": m, "iqr": q}
        print(f"    {k:<20}  median={m:>+10.4f}  iqr={q:>+10.4f}  "
              f"range=[{a.min():+.4f},{a.max():+.4f}]")

    feats_std = {k: _std(a, stats[k]["median"], stats[k]["iqr"]) for k, a in feats.items()}

    # ── save stats ──────────────────────────────────────────────────────────
    out_json = CACHE / f"btc_dqn_standardize_v7_{args.variant}.json"
    out_json.write_text(json.dumps(stats, indent=2))
    print(f"  → {out_json.name}")

    # ── extend state arrays + save per-split ────────────────────────────────
    print(f"\n  Extending state by +{n_new} dims (50 → {50 + n_new}) ...")
    extra_full = np.stack([feats_std[k] for k in feat_names], axis=1).astype(np.float32)
    assert extra_full.shape == (len(ts_full), n_new), extra_full.shape

    for i, nm in enumerate(splits):
        a = boundaries[i]; b = boundaries[i + 1]
        old_state = data[nm]["state"]
        new_state = np.concatenate([old_state, extra_full[a:b]], axis=1)
        assert new_state.shape == (sizes[i], 50 + n_new), new_state.shape

        out = CACHE / f"btc_dqn_state_{nm}_v7_{args.variant}.npz"
        np.savez(
            out,
            state         = new_state,
            valid_actions = data[nm]["valid_actions"],
            ts            = data[nm]["ts"],
            price         = data[nm]["price"],
            signals       = data[nm]["signals"],
            atr           = data[nm]["atr"],
            rank          = data[nm]["rank"],
            regime_id     = data[nm]["regime_id"],
            extra_feat_names = np.array(feat_names),
        )
        print(f"    {nm:<6}  state shape: {new_state.shape}  → {out.name}")

    # sanity: extra-feat stats on each split after standardization
    print(f"\n  Standardized feature stats per split (should be roughly mean~0, std~1 on train):")
    for i, nm in enumerate(splits):
        a = boundaries[i]; b = boundaries[i + 1]
        for j, k in enumerate(feat_names):
            arr = extra_full[a:b, j]
            print(f"    {nm:<6} {k:<20}  mean={arr.mean():>+.2f}  std={arr.std():.2f}  "
                  f"min={arr.min():>+.1f}  max={arr.max():>+.1f}")

    print(f"\n  Total time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
