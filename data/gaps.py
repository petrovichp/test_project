"""
Gap detection and gap-aware windowing.

A gap is any consecutive timestamp difference > GAP_THRESHOLD_S (allowing 30s
jitter on 60s bars). Rows whose lookback window crosses a gap are flagged so
feature engineering can skip or mark them.
"""

import numpy as np
import pandas as pd

GAP_THRESHOLD_S = 90  # seconds; >90s between consecutive bars = true gap


def find_gaps(timestamps: pd.Series, threshold_s: int = GAP_THRESHOLD_S) -> pd.DataFrame:
    """
    Find all gaps in a timestamp series.

    Returns DataFrame with columns:
      start_idx    — last row before the gap
      end_idx      — first row after the gap
      gap_seconds  — size of the gap in seconds
    """
    diffs = timestamps.diff()
    gap_idx = np.where(diffs > threshold_s)[0]

    if len(gap_idx) == 0:
        return pd.DataFrame(columns=["start_idx", "end_idx", "gap_seconds"])

    return pd.DataFrame({
        "start_idx": gap_idx - 1,
        "end_idx": gap_idx,
        "gap_seconds": diffs.iloc[gap_idx].values,
    }).reset_index(drop=True)


def clean_mask(timestamps: pd.Series, max_lookback: int, threshold_s: int = GAP_THRESHOLD_S) -> np.ndarray:
    """
    Boolean mask — True means the row's lookback window is gap-free.

    Row i is contaminated when any gap falls inside its lookback window
    [i - max_lookback, i], i.e. the first max_lookback rows after each gap
    are marked False.
    """
    gaps = find_gaps(timestamps, threshold_s)
    n = len(timestamps)
    contaminated = np.zeros(n, dtype=bool)

    for _, row in gaps.iterrows():
        gap_end = int(row["end_idx"])
        contaminated[gap_end : min(n, gap_end + max_lookback)] = True

    return ~contaminated


def summary(timestamps: pd.Series, threshold_s: int = GAP_THRESHOLD_S) -> None:
    """Print a human-readable gap summary for a timestamp series."""
    gaps = find_gaps(timestamps, threshold_s)
    total = len(timestamps)
    duration_days = (timestamps.iloc[-1] - timestamps.iloc[0]) / 86400

    print(f"Rows          : {total:,}")
    print(f"Duration      : {duration_days:.1f} days")
    print(f"Gaps (>{threshold_s}s)  : {len(gaps)}")

    if len(gaps) == 0:
        print("No gaps found.")
        return

    print(f"Total missing : {int(gaps['gap_seconds'].sum() / 60):,} bars")
    print(f"Largest gap   : {gaps['gap_seconds'].max() / 3600:.2f} hours  "
          f"(row {int(gaps.loc[gaps['gap_seconds'].idxmax(), 'start_idx'])})")
    print("\nTop 5 gaps:")
    top = gaps.nlargest(5, "gap_seconds")[["start_idx", "end_idx", "gap_seconds"]].copy()
    top["gap_hours"] = (top["gap_seconds"] / 3600).round(2)
    print(top.to_string(index=False))
