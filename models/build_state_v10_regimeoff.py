"""
Build v10 state — v8 layout with regime one-hot at dims [2:7] zeroed out.

Tests the A2 audit finding that regime info is largely redundant with
vol_pred / atr_pred / bb_width already in state. If retrained policies
match v8 baseline within ±0.5 WF Sharpe, regime is shortcut.

Input:  cache/state/btc_dqn_state_{split}_v8_s11s13.npz
Output: cache/state/btc_dqn_state_{split}_v10_regimeoff.npz

Run: python3 -m models.build_state_v10_regimeoff [ticker]
"""
import sys, time
from pathlib import Path
import numpy as np

CACHE = Path("cache")


def main(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Build v10 — zero out regime dims [2:7]  ({ticker.upper()})\n{'='*70}")
    for split in ("train", "val", "test"):
        src = CACHE / "state" / f"{ticker}_dqn_state_{split}_v8_s11s13.npz"
        dst = CACHE / "state" / f"{ticker}_dqn_state_{split}_v10_regimeoff.npz"
        d = dict(np.load(src))
        new_state = d["state"].copy()
        new_state[:, 2:7] = 0.0
        d["state"] = new_state
        np.savez(dst, **d)
        nz = int((d["state"][:, 2:7] != 0).sum())
        print(f"  {split:<6} {d['state'].shape}  non-zero in [2:7]={nz}  → {dst.name}")
    print(f"\n  total [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "btc")
