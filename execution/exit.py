"""
Exit strategies — determine TP, SL, and exit conditions per trade.

ATR-dynamic scaling:
  TP = base_tp × (atr_bar / atr_train_median)
  SL = base_sl × (atr_bar / atr_train_median)
  → At median volatility: TP=base_tp, SL=base_sl (same as fixed)
  → At 2× median vol:    TP=2×base_tp, SL=2×base_sl (wider, survives noise)
  → At 0.5× median vol:  TP=0.5×base_tp, SL=0.5×base_sl (tighter, exits faster)

tab_pct (trail-after-breakeven):
  Trailing SL that only activates AFTER breakeven is triggered.
  Distinct from trail_pct (immediate trailing).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ExitPlan:
    tp_pct:         float          # take-profit fraction of entry
    sl_pct:         float          # initial stop-loss fraction of entry
    trail_pct:      float = 0.0    # immediate trailing SL (0 = fixed)
    tab_pct:        float = 0.0    # trailing SL enabled only AFTER breakeven (0 = off)
    breakeven_pct:  float = 0.0    # move SL to entry once unrealised PnL ≥ this (0 = off)
    time_stop_bars: int   = 0      # force-close after N bars (0 = off)


def _scale(base: float, atr_arr: np.ndarray, median: float,
           lo: float, hi: float) -> np.ndarray:
    """Vectorised relative ATR scaling with bounds."""
    if median > 0:
        s = np.clip(atr_arr / median, 0.2, 5.0)   # cap scaling at 5× in either direction
    else:
        s = np.ones(len(atr_arr))
    return np.clip(base * s, lo, hi).astype(np.float32)


class FixedExit:
    """Static TP/SL — simple baseline."""

    def __init__(self, tp_pct=0.020, sl_pct=0.007, trail_pct=0.0, tab_pct=0.0,
                 breakeven_pct=0.0, time_stop_bars=0):
        self._p = dict(tp_pct=tp_pct, sl_pct=sl_pct, trail_pct=trail_pct,
                       tab_pct=tab_pct, breakeven_pct=breakeven_pct,
                       time_stop_bars=time_stop_bars)

    def plan(self, atr_pred=None, price=None, atr_train_median=None) -> ExitPlan:
        return ExitPlan(**self._p)

    def arrays(self, atr_arr, price_arr, atr_train_median=None):
        n = len(atr_arr)
        return (np.full(n, self._p["tp_pct"], dtype=np.float32),
                np.full(n, self._p["sl_pct"], dtype=np.float32))


class ATRDynamicExit:
    """TP and SL scale with predicted ATR relative to training median.

    At median vol:  TP=base_tp,  SL=base_sl
    At 2× median:   TP=2×base_tp, SL=2×base_sl
    """

    def __init__(self, base_tp_pct=0.020, base_sl_pct=0.007,
                 trail_pct=0.0, tab_pct=0.0,
                 breakeven_pct=0.0, time_stop_bars=0,
                 min_tp=0.005, max_tp=0.060, min_sl=0.002, max_sl=0.025):
        self.base_tp = base_tp_pct
        self.base_sl = base_sl_pct
        self._tr  = trail_pct
        self._tab = tab_pct
        self._be  = breakeven_pct
        self._ts  = time_stop_bars
        self.min_tp, self.max_tp = min_tp, max_tp
        self.min_sl, self.max_sl = min_sl, max_sl

    def plan(self, atr_pred: float, price: float, atr_train_median: float = None) -> ExitPlan:
        median = atr_train_median or atr_pred   # fallback: no scaling
        scale  = float(np.clip(atr_pred / median, 0.2, 5.0)) if median > 0 else 1.0
        tp = float(np.clip(self.base_tp * scale, self.min_tp, self.max_tp))
        sl = float(np.clip(self.base_sl * scale, self.min_sl, self.max_sl))
        return ExitPlan(tp_pct=tp, sl_pct=sl, trail_pct=self._tr, tab_pct=self._tab,
                        breakeven_pct=self._be, time_stop_bars=self._ts)

    def arrays(self, atr_arr: np.ndarray, price_arr: np.ndarray,
               atr_train_median: float = None) -> tuple:
        median = atr_train_median or float(np.median(atr_arr))
        tp_arr = _scale(self.base_tp, atr_arr, median, self.min_tp, self.max_tp)
        sl_arr = _scale(self.base_sl, atr_arr, median, self.min_sl, self.max_sl)
        return tp_arr, sl_arr


class ComboExit:
    """ATR-scaled TP/SL + breakeven stop + time stop.

    trail_after_breakeven: enable trailing SL (at sl_pct) once breakeven triggers.
    This prevents immediate trailing noise while still locking in gains on runners.
    """

    def __init__(self, base_tp_pct=0.020, base_sl_pct=0.007,
                 trail_after_breakeven=False,
                 breakeven_pct=0.005, time_stop_bars=60,
                 min_tp=0.005, max_tp=0.060, min_sl=0.002, max_sl=0.025):
        self.base_tp  = base_tp_pct
        self.base_sl  = base_sl_pct
        self._tab_en  = trail_after_breakeven   # enable trail after BE?
        self._be      = breakeven_pct
        self._ts      = time_stop_bars
        self.min_tp, self.max_tp = min_tp, max_tp
        self.min_sl, self.max_sl = min_sl, max_sl

    def plan(self, atr_pred: float, price: float, atr_train_median: float = None) -> ExitPlan:
        median = atr_train_median or atr_pred
        scale  = float(np.clip(atr_pred / median, 0.2, 5.0)) if median > 0 else 1.0
        tp  = float(np.clip(self.base_tp * scale, self.min_tp, self.max_tp))
        sl  = float(np.clip(self.base_sl * scale, self.min_sl, self.max_sl))
        tab = sl if self._tab_en else 0.0        # trail amount = current sl
        return ExitPlan(tp_pct=tp, sl_pct=sl,
                        trail_pct=0.0, tab_pct=tab,
                        breakeven_pct=self._be, time_stop_bars=self._ts)

    def arrays(self, atr_arr: np.ndarray, price_arr: np.ndarray,
               atr_train_median: float = None) -> tuple:
        median = atr_train_median or float(np.median(atr_arr))
        tp_arr = _scale(self.base_tp, atr_arr, median, self.min_tp, self.max_tp)
        sl_arr = _scale(self.base_sl, atr_arr, median, self.min_sl, self.max_sl)
        return tp_arr, sl_arr
