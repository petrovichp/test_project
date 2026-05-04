"""
Bar-by-bar backtest engine for OKX perp trading.

Assumptions:
  - Signal at bar T close → execute at bar T+1 close (1-bar lag)
  - Position sizing: fixed risk per trade = 1% of equity / SL distance
  - Max 1 position at a time (no pyramiding)
  - Long and short via perp (symmetric fees)
  - OKX taker fee 0.08% per side

Metrics computed: total return, annualised Sharpe, Calmar ratio, max drawdown,
                  win rate, profit factor, avg trade, n_trades.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from backtest.costs import TAKER_FEE


@dataclass
class Trade:
    bar_in:    int
    direction: int        # +1 long, -1 short
    entry:     float
    tp:        float
    sl:        float
    bar_out:   int   = -1
    exit:      float = 0.0
    pnl_pct:   float = 0.0
    exit_reason: str = ""


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
        # 1-min bars → 525,960 bars/year
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
        wins  = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
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
    signals:         np.ndarray,  # (n,): +1 long, -1 short, 0 flat
    prices:          np.ndarray,  # (n,): bar close prices
    tp_pct_arr:      np.ndarray,  # (n,): take-profit as fraction of entry (e.g. 0.008)
    sl_pct_arr:      np.ndarray,  # (n,): stop-loss  as fraction of entry (e.g. 0.003)
    timestamps:      np.ndarray,  # (n,): unix timestamps
    initial_capital: float = 10_000,
    position_size:   float = 0.10,   # fraction of equity per trade (10%)
) -> EngineResult:
    """
    Bar-by-bar simulation with 1-bar execution lag.

    Signal at bar T → enter at bar T+1 close.
    TP/SL are percentages of entry price — checked every bar after entry.
    Position size is a fixed fraction of current equity.
    """
    n      = len(signals)
    equity = np.full(n + 1, initial_capital)
    trades: list[Trade] = []
    position: Trade | None = None

    for i in range(1, n):
        price = prices[i]
        eq    = equity[i]

        # ── manage open position ──────────────────────────────────────────
        if position is not None:
            hit_tp = (position.direction ==  1 and price >= position.tp) or \
                     (position.direction == -1 and price <= position.tp)
            hit_sl = (position.direction ==  1 and price <= position.sl) or \
                     (position.direction == -1 and price >= position.sl)

            if hit_tp or hit_sl:
                exit_price   = position.tp if hit_tp else position.sl
                raw_pnl      = position.direction * (exit_price / position.entry - 1)
                net_pnl      = raw_pnl - 2 * TAKER_FEE
                position.bar_out     = i
                position.exit        = exit_price
                position.pnl_pct     = net_pnl
                position.exit_reason = "TP" if hit_tp else "SL"
                trades.append(position)
                equity[i + 1] = eq * (1 + net_pnl * position_size)
                position = None
                continue

        equity[i + 1] = equity[i]

        # ── new signal (1-bar lag) ────────────────────────────────────────
        if position is None and signals[i - 1] != 0:
            direction = int(signals[i - 1])
            entry     = price * (1 + direction * TAKER_FEE)
            tp_pct    = float(tp_pct_arr[i - 1])
            sl_pct    = float(sl_pct_arr[i - 1])
            tp        = entry * (1 + direction * tp_pct)
            sl        = entry * (1 - direction * sl_pct)

            if direction ==  1 and tp > entry and sl < entry:
                position = Trade(i, direction, entry, tp, sl)
            elif direction == -1 and tp < entry and sl > entry:
                position = Trade(i, direction, entry, tp, sl)

    # force-close at last bar
    if position is not None:
        price    = prices[-1]
        raw_pnl  = position.direction * (price / position.entry - 1)
        net_pnl  = raw_pnl - 2 * TAKER_FEE
        position.bar_out     = n - 1
        position.exit        = price
        position.pnl_pct     = net_pnl
        position.exit_reason = "EOD"
        trades.append(position)

    return EngineResult(equity=equity[:n], trades=trades, timestamps=timestamps)
