"""
Bar-by-bar backtest engine for OKX perp trading.

Assumptions:
  - Signal at bar T close → execute at bar T+1 close (1-bar lag)
  - Position sizing: per-bar array or fixed fraction of equity
  - Max 1 position at a time (no pyramiding)
  - Long and short via perp (symmetric fees)
  - OKX taker fee 0.08% per side

Exit mechanisms (all optional, stacked):
  - Fixed TP / SL
  - Trailing SL (ratchets toward running price peak)
  - Breakeven stop (move SL to entry once profit ≥ breakeven_pct)
  - Time stop (force-close after N bars)
  - Force-exit array (external close signal)

Metrics: total return, annualised Sharpe, Calmar, max drawdown,
         win rate, profit factor, avg trade, n_trades.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from backtest.costs import TAKER_FEE


@dataclass
class Trade:
    bar_in:         int
    direction:      int       # +1 long, -1 short
    entry:          float
    tp:             float
    sl:             float
    trail_pct:      float = 0.0   # immediate trailing SL (0 = fixed)
    tab_pct:        float = 0.0   # trail-after-breakeven: activates once BE triggers
    breakeven_pct:  float = 0.0   # 0 = disabled
    time_stop_bars: int   = 0     # 0 = disabled
    position_size:  float = 0.10  # fraction of equity
    bar_out:        int   = -1
    exit:           float = 0.0
    pnl_pct:        float = 0.0
    exit_reason:    str   = ""
    _be_done:       bool  = field(default=False, repr=False)  # breakeven triggered


@dataclass
class EngineResult:
    equity:     np.ndarray
    trades:     list
    timestamps: np.ndarray

    @property
    def returns(self) -> np.ndarray:
        return np.diff(self.equity) / self.equity[:-1]

    @property
    def total_return(self) -> float:
        return self.equity[-1] / self.equity[0] - 1

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_pct > 0) / len(self.trades)

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity)
        dd   = (self.equity - peak) / peak
        return float(dd.min())

    @property
    def sharpe(self) -> float:
        r = self.returns
        if r.std() < 1e-10:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(525_960))

    @property
    def calmar(self) -> float:
        mdd = abs(self.max_drawdown)
        if mdd < 1e-10:
            return 0.0
        n_bars = len(self.equity)
        annual_return = (1 + self.total_return) ** (525_960 / n_bars) - 1
        return float(annual_return / mdd)

    @property
    def profit_factor(self) -> float:
        wins   = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        losses = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        return float(wins / losses) if losses > 1e-10 else float("inf")

    def summary(self) -> dict:
        return {
            "total_return":  round(self.total_return * 100, 2),
            "sharpe":        round(self.sharpe, 3),
            "calmar":        round(self.calmar, 3),
            "max_drawdown":  round(self.max_drawdown * 100, 2),
            "n_trades":      self.n_trades,
            "win_rate":      round(self.win_rate * 100, 1),
            "profit_factor": round(self.profit_factor, 3),
        }


def run(
    signals:            np.ndarray,           # (n,) +1 long / -1 short / 0 flat
    prices:             np.ndarray,           # (n,) bar close prices
    tp_pct_arr:         np.ndarray,           # (n,) take-profit fraction of entry
    sl_pct_arr:         np.ndarray,           # (n,) initial stop-loss fraction
    timestamps:         np.ndarray,           # (n,) unix timestamps
    trail_pct_arr:      np.ndarray = None,    # (n,) immediate trailing SL pct
    tab_pct_arr:        np.ndarray = None,    # (n,) trail-after-breakeven pct
    breakeven_pct_arr:  np.ndarray = None,    # (n,) breakeven trigger pct (0 = off)
    time_stop_arr:      np.ndarray = None,    # (n,) time stop in bars (0 = off)
    position_size_arr:  np.ndarray = None,    # (n,) per-bar position size (equity fraction)
    force_exit_arr:     np.ndarray = None,    # (n,) 1 = close open position this bar
    initial_capital:    float = 10_000,
    position_size:      float = 0.10,         # fallback if position_size_arr is None
) -> EngineResult:
    """
    Bar-by-bar simulation with 1-bar execution lag.

    Signal at bar T → enter at bar T+1 close.
    All exit mechanisms are checked in order each bar:
      1. Force exit (external close signal)
      2. Time stop
      3. Breakeven stop ratchet
      4. Trailing SL ratchet
      5. TP / SL check
    """
    n         = len(signals)
    equity    = np.full(n + 1, initial_capital, dtype=np.float64)
    trades:   list[Trade] = []
    position: Trade | None = None

    for i in range(1, n):
        price = prices[i]
        eq    = equity[i]

        if position is not None:

            # ── 1. Force exit (external close signal) ─────────────────────────
            if force_exit_arr is not None and force_exit_arr[i]:
                raw_pnl = position.direction * (price / position.entry - 1)
                net_pnl = raw_pnl - 2 * TAKER_FEE
                position.bar_out     = i
                position.exit        = price
                position.pnl_pct     = net_pnl
                position.exit_reason = "FORCE"
                trades.append(position)
                equity[i + 1] = eq * (1 + net_pnl * position.position_size)
                position = None
                continue

            # ── 2. Time stop ──────────────────────────────────────────────────
            if position.time_stop_bars > 0:
                if (i - position.bar_in) >= position.time_stop_bars:
                    raw_pnl = position.direction * (price / position.entry - 1)
                    net_pnl = raw_pnl - 2 * TAKER_FEE
                    position.bar_out     = i
                    position.exit        = price
                    position.pnl_pct     = net_pnl
                    position.exit_reason = "TIME"
                    trades.append(position)
                    equity[i + 1] = eq * (1 + net_pnl * position.position_size)
                    position = None
                    continue

            # ── 3. Breakeven stop ratchet ─────────────────────────────────────
            if position.breakeven_pct > 0 and not position._be_done:
                unrealised = position.direction * (price / position.entry - 1)
                if unrealised >= position.breakeven_pct:
                    position.sl       = position.entry
                    position._be_done = True
                    if position.tab_pct > 0:        # enable trail-after-breakeven now
                        position.trail_pct = position.tab_pct

            # ── 4. Trailing SL ratchet ────────────────────────────────────────
            if position.trail_pct > 0:
                if position.direction == 1:
                    position.sl = max(position.sl, price * (1 - position.trail_pct))
                else:
                    position.sl = min(position.sl, price * (1 + position.trail_pct))

            # ── 5. TP / SL check ──────────────────────────────────────────────
            hit_tp = ((position.direction ==  1 and price >= position.tp) or
                      (position.direction == -1 and price <= position.tp))
            hit_sl = ((position.direction ==  1 and price <= position.sl) or
                      (position.direction == -1 and price >= position.sl))

            if hit_tp or hit_sl:
                exit_price = position.tp if hit_tp else position.sl
                raw_pnl    = position.direction * (exit_price / position.entry - 1)
                net_pnl    = raw_pnl - 2 * TAKER_FEE

                reason = "TP" if hit_tp else ("TSL" if position.trail_pct > 0 else "SL")
                if not hit_tp and position._be_done:
                    reason = "BE"   # breakeven stop triggered

                position.bar_out     = i
                position.exit        = exit_price
                position.pnl_pct     = net_pnl
                position.exit_reason = reason
                trades.append(position)
                equity[i + 1] = eq * (1 + net_pnl * position.position_size)
                position = None
                continue

        equity[i + 1] = equity[i]

        # ── New signal (1-bar lag) ────────────────────────────────────────────
        if position is None and signals[i - 1] != 0:
            direction = int(signals[i - 1])
            entry     = price * (1 + direction * TAKER_FEE)
            tp_pct    = float(tp_pct_arr[i - 1])
            sl_pct    = float(sl_pct_arr[i - 1])
            tp        = entry * (1 + direction * tp_pct)
            sl        = entry * (1 - direction * sl_pct)

            trail    = float(trail_pct_arr[i - 1])     if trail_pct_arr     is not None else 0.0
            tab      = float(tab_pct_arr[i - 1])       if tab_pct_arr       is not None else 0.0
            be_pct   = float(breakeven_pct_arr[i - 1]) if breakeven_pct_arr is not None else 0.0
            ts_bars  = int(time_stop_arr[i - 1])       if time_stop_arr     is not None else 0
            pos_size = float(position_size_arr[i - 1]) if position_size_arr is not None else position_size

            valid = ((direction ==  1 and tp > entry and sl < entry) or
                     (direction == -1 and tp < entry and sl > entry))
            if valid:
                position = Trade(
                    bar_in=i, direction=direction, entry=entry, tp=tp, sl=sl,
                    trail_pct=trail, tab_pct=tab, breakeven_pct=be_pct,
                    time_stop_bars=ts_bars, position_size=pos_size,
                )

    # ── Force-close any open position at last bar ─────────────────────────────
    if position is not None:
        price   = prices[-1]
        raw_pnl = position.direction * (price / position.entry - 1)
        net_pnl = raw_pnl - 2 * TAKER_FEE
        position.bar_out     = n - 1
        position.exit        = price
        position.pnl_pct     = net_pnl
        position.exit_reason = "EOD"
        trades.append(position)

    return EngineResult(equity=equity[:n], trades=trades, timestamps=timestamps)
