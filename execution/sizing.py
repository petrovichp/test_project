"""
Position sizing strategies.

Each strategy exposes:
  .size(equity, atr_pred, price, sl_pct) → float  (fraction of equity)
"""

import numpy as np


class FixedFraction:
    """Fixed fraction of equity per trade — current baseline."""

    def __init__(self, fraction=0.10):
        self.fraction = fraction

    def size(self, equity=None, atr_pred=None, price=None, sl_pct=None) -> float:
        return self.fraction


class VolScaledSizer:
    """Constant dollar risk per trade.

    position_size = target_risk_pct / sl_pct
    → each trade risks the same dollar amount regardless of SL width.

    Example: target_risk=1%, sl=0.5%  → position=20% equity
             target_risk=1%, sl=2.0%  → position=5%  equity
    """

    def __init__(self, target_risk_pct=0.010, max_size=0.25, min_size=0.02):
        self.target_risk = target_risk_pct
        self.max_size    = max_size
        self.min_size    = min_size

    def size(self, equity=None, atr_pred=None, price=None, sl_pct=None) -> float:
        if sl_pct is None or sl_pct < 1e-6:
            return self.min_size
        return float(np.clip(self.target_risk / sl_pct, self.min_size, self.max_size))
