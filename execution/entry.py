"""
Entry strategies — pre-process signal arrays before passing to engine.

Each strategy exposes:
  .apply(signals) → np.ndarray  (processed signal array)
"""

import numpy as np


class MarketEntry:
    """Enter at next bar close — current baseline. No preprocessing."""

    def apply(self, signals: np.ndarray) -> np.ndarray:
        return signals


class ConfirmEntry:
    """Require k consecutive same-direction bars before firing signal.

    Prevents entering on single-bar noise.
    Signal fires on bar k, engine enters at bar k+1 (1-bar lag preserved).

    Example k=2: bar[5]=+1, bar[6]=+1 → confirmed signal fires at bar[6].
    """

    def __init__(self, k: int = 2):
        self.k = k

    def apply(self, signals: np.ndarray) -> np.ndarray:
        out = np.zeros_like(signals)
        k   = self.k
        for i in range(k - 1, len(signals)):
            window = signals[i - k + 1: i + 1]
            if window[0] != 0 and np.all(window == window[0]):
                out[i] = window[0]
        return out


class SpreadEntry:
    """Enter only when bid-ask spread is below threshold.

    spread_arr must be supplied at apply() time.
    """

    def __init__(self, max_spread_bps: float = 5.0):
        self.max_spread = max_spread_bps

    def apply(self, signals: np.ndarray, spread_arr: np.ndarray = None) -> np.ndarray:
        if spread_arr is None:
            return signals
        mask = (spread_arr < self.max_spread).astype(int)
        return signals * mask
