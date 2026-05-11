"""
Build v9 state — combines v7_basis (basis+funding state features)
and v8_s11s13 (S11+S13 action expansion).

Shapes:
  state:         50 (orig) + 5 (basis) + 2 (S11/S13 flags) = 57 dims
  signals:       (n, 11)  — same as v8
  valid_actions: (n, 12)  — same as v8

Hypothesis: Step 3 (basis state) lifted val by +2.84 but hurt test.
Step 4 (S11+S13 action) lifted WF +1.02. Combining should capture
both lifts — val resilience AND larger action space — without the
test regression Step 3 alone showed.

Run: python3 -m models.build_state_v9
"""
import pathlib, time
import numpy as np

CACHE = pathlib.Path("cache")


def main():
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Build v9 state — combine v7_basis + v8_s11s13\n{'='*70}")

    for split in ("train", "val", "test"):
        d_basis = np.load(CACHE / f"btc_dqn_state_{split}_v7_basis.npz")
        d_v8    = np.load(CACHE / f"btc_dqn_state_{split}_v8_s11s13.npz")

        assert d_basis["state"].shape[0] == d_v8["state"].shape[0]
        n = d_basis["state"].shape[0]
        assert d_basis["state"].shape[1] == 55, d_basis["state"].shape  # 50 + 5 basis
        assert d_v8["state"].shape[1]    == 52, d_v8["state"].shape     # 50 + 2 s11/s13 flags

        # state v9: 55 (basis) + 2 (S11/S13 flags from v8 cols 50:52) = 57 dims
        s11s13_flags = d_v8["state"][:, 50:52]
        new_state    = np.concatenate([d_basis["state"], s11s13_flags], axis=1)
        assert new_state.shape == (n, 57), new_state.shape

        # signals + valid_actions come from v8 (12-action space)
        new_signals = d_v8["signals"]          # (n, 11)
        new_valid   = d_v8["valid_actions"]    # (n, 12)

        out = CACHE / f"btc_dqn_state_{split}_v9_basis_s11s13.npz"
        np.savez(
            out,
            state         = new_state.astype(np.float32),
            valid_actions = new_valid,
            ts            = d_v8["ts"],
            price         = d_v8["price"],
            signals       = new_signals,
            atr           = d_v8["atr"],
            rank          = d_v8["rank"],
            regime_id     = d_v8["regime_id"],
        )
        print(f"  {split:<6}  state={new_state.shape}  signals={new_signals.shape}  "
              f"valid={new_valid.shape}  → {out.name}")

    print(f"\n  Total time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
