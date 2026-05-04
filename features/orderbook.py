"""
Orderbook features from the 800-column OB snapshot.

Feature groups:
  1. Bucket amounts      : spot/perp × bids/asks × [0-50, 50-100, 100-200]
  2. Bucket imbalances   : (bid-ask)/(bid+ask) per bucket per instrument
  3. Bucket velocities   : 1-bar diff of bucket amounts
  4. Span scalars        : span_spot_price, span_perp_price (= bid-ask spread)
  5. True OFI            : Δbids[0:N] - Δasks[0:N], N=5,10,20 — order flow imbalance
  6. Rolling OFI         : cumsum of OFI over 5, 15, 30 bars
  7. Depth band features : imbalance and concentration at fine-grained bands
                           [0-5], [5-20], [20-50], [50-100] bins (proxy for price levels)
  8. Wall detection      : bin index of max single-bin amount near mid (top 50 bins)
  9. Book shape          : near-mid concentration = top-10-bin share of total liquidity

Note on price-level features: span_spot/perp = bid-ask spread, not OB depth range.
Exact price-distance conversion (±0.5% from mid) requires adding OB depth span to
the data collection pipeline. Current features use bin-band proxies instead.

Output: cache/{ticker}_features_ob.parquet
Run   : python3 -m features.orderbook [ticker]
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loader import load, load_meta

CACHE_DIR   = Path(__file__).parent.parent / "cache"
BUCKETS     = [(0, 50), (50, 100), (100, 200)]
DEPTH_BANDS = [(0, 5), (5, 20), (20, 50), (50, 100)]   # fine-grained near-mid bands
OFI_DEPTHS  = [5, 10, 20]                               # bins used for True OFI
INSTRUMENTS = ["spot", "perp"]
SIDES       = ["bids", "asks"]


def _cols(inst: str, side: str, lo: int, hi: int) -> list[str]:
    return [f"{inst}_{side}_amount_{i}" for i in range(lo, hi)]


def compute(ticker: str, force: bool = False) -> pd.DataFrame:
    out = CACHE_DIR / f"{ticker}_features_ob.parquet"
    if out.exists() and not force:
        print(f"Loading OB features from cache: {out.name}")
        return pd.read_parquet(out)

    print(f"Computing OB features for {ticker} ...")
    meta, ob = load(ticker, include_ob=True)

    feats = pd.DataFrame(index=meta.index)
    feats["timestamp"] = meta["timestamp"].values

    # ── 1. bucket amounts ─────────────────────────────────────────────────────
    buckets: dict[str, pd.Series] = {}
    for inst in INSTRUMENTS:
        for side in SIDES:
            for lo, hi in BUCKETS:
                key = f"{inst}_{side}_{lo}_{hi}"
                buckets[key] = ob[_cols(inst, side, lo, hi)].sum(axis=1)
                feats[f"ob_amt_{key}"] = buckets[key]

    # ── 2. per-bucket imbalance ───────────────────────────────────────────────
    for inst in INSTRUMENTS:
        for lo, hi in BUCKETS:
            bid = buckets[f"{inst}_bids_{lo}_{hi}"]
            ask = buckets[f"{inst}_asks_{lo}_{hi}"]
            feats[f"ob_imb_{inst}_{lo}_{hi}"] = (bid - ask) / (bid + ask).replace(0, np.nan)

    # ── 3. bucket velocities ──────────────────────────────────────────────────
    for inst in INSTRUMENTS:
        for side in SIDES:
            for lo, hi in BUCKETS:
                key = f"ob_amt_{inst}_{side}_{lo}_{hi}"
                feats[f"ob_vel_{inst}_{side}_{lo}_{hi}"] = feats[key].diff(1)

    # ── 4. span scalars ───────────────────────────────────────────────────────
    feats["span_spot"] = meta["span_spot_price"].values   # bid-ask spread (%)
    feats["span_perp"] = meta["span_perp_price"].values

    # ── 4b. OB depth span (price-level features) ──────────────────────────────
    # ob_depth_span_spot/perp = dollar range covered by 200 bins.
    # Added to collection pipeline — only present in data collected after the fix.
    # When available: bin_price_offset = (bin_idx / 200) * ob_depth_span
    # Enables "liquidity within ±0.5% of mid" and exact wall price features.
    for col, name in [("ob_depth_span_spot", "depth_span_spot"),
                      ("ob_depth_span_perp", "depth_span_perp")]:
        if col in meta.columns:
            feats[name] = meta[col].values
            # Price-level features only computable when depth span is available
            price_col = "spot_ask_price" if "spot" in col else "perp_ask_price"
            mid = meta[price_col].values
            span_dollar = meta[col].values
            for inst, side_cols in [("spot", "spot"), ("perp", "perp")]:
                if inst not in col:
                    continue
                # Liquidity within ±0.5% and ±1% of mid (bin cutoffs from span)
                for pct in [0.005, 0.01]:
                    cutoff = np.clip(
                        (pct * mid / (span_dollar + 1e-12) * 200).astype(int),
                        1, 199
                    )
                    bid_near = np.array([
                        ob[[f"{inst}_bids_amount_{i}" for i in range(int(c))]].iloc[row].sum()
                        for row, c in enumerate(cutoff)
                    ]) if len(cutoff) > 0 else np.zeros(len(meta))
                    feats[f"liq_bid_{inst}_{int(pct*1000)}bps"] = bid_near

    # ── 5 & 6. True OFI and rolling OFI ──────────────────────────────────────
    # OFI = Δ(near-bid-qty) - Δ(near-ask-qty)
    # Positive = bids building or asks pulling → buying pressure
    # Negative = asks building or bids pulling → selling pressure
    for inst in INSTRUMENTS:
        for n in OFI_DEPTHS:
            bid_near = ob[_cols(inst, "bids", 0, n)].sum(axis=1)
            ask_near = ob[_cols(inst, "asks", 0, n)].sum(axis=1)
            ofi      = bid_near.diff(1) - ask_near.diff(1)
            feats[f"ofi_{inst}_{n}"] = ofi
            for w in [5, 15, 30]:
                feats[f"ofi_{inst}_{n}_r{w}"] = ofi.rolling(w, min_periods=w).sum()

    # ── 7. depth band features (price-level proxy) ────────────────────────────
    # Fine-grained bands near mid: [0-5], [5-20], [20-50], [50-100] bins.
    # Band 0-5   ≈ very near mid (within ~2-3% of full depth range)
    # Band 5-20  ≈ near mid
    # Band 20-50 ≈ mid range
    # Band 50-100≈ far from mid
    for inst in INSTRUMENTS:
        total_bid = ob[_cols(inst, "bids", 0, 200)].sum(axis=1)
        total_ask = ob[_cols(inst, "asks", 0, 200)].sum(axis=1)

        for lo, hi in DEPTH_BANDS:
            band_bid = ob[_cols(inst, "bids", lo, hi)].sum(axis=1)
            band_ask = ob[_cols(inst, "asks", lo, hi)].sum(axis=1)

            # imbalance within this band
            denom = (band_bid + band_ask).replace(0, np.nan)
            feats[f"band_imb_{inst}_{lo}_{hi}"] = (band_bid - band_ask) / denom

            # share of total liquidity in this band (concentration)
            feats[f"band_bid_share_{inst}_{lo}_{hi}"] = band_bid / total_bid.replace(0, np.nan)
            feats[f"band_ask_share_{inst}_{lo}_{hi}"] = band_ask / total_ask.replace(0, np.nan)

        # spot vs perp near-mid imbalance divergence — when they disagree → informed flow
        if inst == "perp":
            perp_near = feats["band_imb_perp_0_5"]
            spot_near = feats["band_imb_spot_0_5"]
            feats["ob_spot_perp_imb_div"] = perp_near - spot_near

    # ── 8. wall detection ─────────────────────────────────────────────────────
    # Location (bin index) of the largest single order cluster within top 50 bins.
    # Low index = wall is close to mid (strong near-term support/resistance).
    for inst in INSTRUMENTS:
        bid50 = ob[_cols(inst, "bids", 0, 50)].values
        ask50 = ob[_cols(inst, "asks", 0, 50)].values
        feats[f"wall_bid_idx_{inst}"]  = bid50.argmax(axis=1).astype(np.float32)
        feats[f"wall_ask_idx_{inst}"]  = ask50.argmax(axis=1).astype(np.float32)
        feats[f"wall_bid_size_{inst}"] = bid50.max(axis=1)
        feats[f"wall_ask_size_{inst}"] = ask50.max(axis=1)

    # ── 9. book shape — near-mid concentration ────────────────────────────────
    # What fraction of total bid/ask liquidity sits within the top 10 bins?
    # High value = thin book (liquidity concentrated near mid, large moves possible)
    # Low value  = deep book (spread across many levels, more stable)
    for inst in INSTRUMENTS:
        bid10  = ob[_cols(inst, "bids", 0, 10)].sum(axis=1)
        ask10  = ob[_cols(inst, "asks", 0, 10)].sum(axis=1)
        bid200 = ob[_cols(inst, "bids", 0, 200)].sum(axis=1)
        ask200 = ob[_cols(inst, "asks", 0, 200)].sum(axis=1)
        feats[f"near_conc_bid_{inst}"] = bid10 / bid200.replace(0, np.nan)
        feats[f"near_conc_ask_{inst}"] = ask10 / ask200.replace(0, np.nan)

    CACHE_DIR.mkdir(exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"  Saved {out.name}  ({out.stat().st_size // 1024:,} KB)  shape={feats.shape}")
    return feats


if __name__ == "__main__":
    force  = "--force" in sys.argv
    ticker = next((a for a in sys.argv[1:] if not a.startswith("-")), "btc")
    df = compute(ticker, force=force)
    print(f"\nShape: {df.shape}")
    groups = {
        "bucket_amounts":   [c for c in df.columns if c.startswith("ob_amt_")],
        "bucket_imbalance": [c for c in df.columns if c.startswith("ob_imb_")],
        "bucket_velocity":  [c for c in df.columns if c.startswith("ob_vel_")],
        "true_ofi":         [c for c in df.columns if c.startswith("ofi_") and "_r" not in c],
        "rolling_ofi":      [c for c in df.columns if c.startswith("ofi_") and "_r" in c],
        "depth_bands":      [c for c in df.columns if c.startswith("band_")],
        "wall_detection":   [c for c in df.columns if c.startswith("wall_")],
        "book_shape":       [c for c in df.columns if c.startswith("near_conc_")],
        "misc":             ["span_spot", "span_perp", "ob_spot_perp_imb_div"],
    }
    for g, cols in groups.items():
        print(f"  {g:<20} {len(cols):>3} features")
    print(f"  {'TOTAL':<20} {sum(len(v) for v in groups.values()):>3} features")
