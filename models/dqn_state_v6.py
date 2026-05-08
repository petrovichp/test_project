"""
v6 state: extends v5 (50-dim) with 4 direction-prediction features (54-dim total).

New dims (50–53):
    50  p_up_60  centered:  2 * (P_up_60   − 0.5)   → ∈ [-1, 1]
    51  p_dn_60  centered:  2 * (P_dn_60   − 0.5)
    52  p_up_100 centered:  2 * (P_up_100  − 0.5)
    53  p_dn_100 centered:  2 * (P_dn_100  − 0.5)

All other dims (0–49) are identical to v5. The 4 CNN-LSTM probabilities (AUC
0.64–0.70 OOS per docs/experiments_log.md) are computed and cached but never
exposed to the DQN in v5; this version exposes them directly.

Output: cache/btc_dqn_state_{train,val,test}_v6.npz — same schema as v5 but
state has shape (N, 54).

Run: python3 -m models.dqn_state_v6 [ticker]
"""
import sys, time
import numpy as np
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "cache"


def run(ticker: str = "btc"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  v6 state builder ({ticker.upper()}) — 50 → 54 dims\n{'='*70}")

    # ── load v5 cached state files ───────────────────────────────────────────
    splits = {}
    for split in ("train", "val", "test"):
        p = CACHE / f"{ticker}_dqn_state_{split}.npz"
        if not p.exists():
            print(f"  ! missing {p.name} — run models.dqn_state first"); return
        splits[split] = dict(np.load(p))
        print(f"  loaded {p.name}  state shape {splits[split]['state'].shape}")

    # ── load full direction predictions (full 383k bars; v5 stripped them) ──
    print(f"\n  loading direction predictions (full bars) ...")
    dir_full = {}
    for col in ("up_60", "down_60", "up_100", "down_100"):
        d = np.load(CACHE / f"{ticker}_pred_dir_{col}_v4.npz")
        a = d["preds"].astype(np.float32)
        if np.isnan(a).any():
            import pandas as pd
            a = pd.Series(a).bfill().fillna(0.5).values.astype(np.float32)
        dir_full[col] = a
        print(f"    {col:<10}  {len(a):,} bars  AUC RL = {float(d['auc_rl']):.4f}")

    # the v5 state arrays have these split bar counts (after WARMUP=1440 trim):
    #   train: bars [101440-1440, 281440-1440)
    #   val:   bars [281440-1440, 332307-1440)
    #   test:  bars [332307-1440, 384614-1440)
    # but we should not reconstruct from constants — just align by length & start.
    WARMUP = 1440
    # cached direction prediction arrays are POST-WARMUP (length 383,174).
    # v5 split arrays are also post-WARMUP slices of the same length.
    # post-WARMUP start indices: subtract WARMUP from raw boundaries.
    starts = dict(train=101_440 - WARMUP, val=281_440 - WARMUP, test=332_307 - WARMUP)

    # ── extend each split's state to 54 dims ─────────────────────────────────
    for split, sp in splits.items():
        n = len(sp["state"])
        a = starts[split]
        b = a + n
        # extract aligned direction probs
        slice_d = {col: dir_full[col][a:b] for col in dir_full}
        # sanity
        for col, arr in slice_d.items():
            assert len(arr) == n, f"{split}/{col} len mismatch {len(arr)} vs {n}"

        # center to [-1, 1]
        new_dims = np.stack([
            2.0 * (slice_d["up_60"]    - 0.5),
            2.0 * (slice_d["down_60"]  - 0.5),
            2.0 * (slice_d["up_100"]   - 0.5),
            2.0 * (slice_d["down_100"] - 0.5),
        ], axis=1).astype(np.float32)            # shape (n, 4)

        new_state = np.concatenate([sp["state"], new_dims], axis=1)   # (n, 54)
        assert new_state.shape == (n, 54)

        out_path = CACHE / f"{ticker}_dqn_state_{split}_v6.npz"
        np.savez(
            out_path,
            state         = new_state,
            valid_actions = sp["valid_actions"],
            ts            = sp["ts"],
            price         = sp["price"],
            signals       = sp["signals"],
            atr           = sp["atr"],
            rank          = sp["rank"],
            regime_id     = sp["regime_id"],
        )
        # report stats on the new dims
        for i, name in enumerate(["p_up_60", "p_dn_60", "p_up_100", "p_dn_100"]):
            col = new_dims[:, i]
            print(f"    {split:<6}  dim {50+i} ({name:<8})  mean={col.mean():>+.3f}  "
                  f"std={col.std():.3f}  min={col.min():>+.2f}  max={col.max():>+.2f}")
        print(f"  → {out_path.name}  state shape {new_state.shape}\n")

    print(f"\n  Total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
