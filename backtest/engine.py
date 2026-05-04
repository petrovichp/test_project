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
    signals:    np.ndarray,   # shape (n,): +1 long, -1 short, 0 flat
    prices:     np.ndarray,   # shape (n,): bar close prices
    tp_arr:     np.ndarray,   # shape (n,): take-profit price per bar
    sl_arr:     np.ndarray,   # shape (n,): stop-loss price per bar
    timestamps: np.ndarray,   # shape (n,): unix timestamps
    initial_capital: float = 10_000,
    risk_per_trade:  float = 0.01,    # 1% of equity risked per trade
) -> EngineResult:
    """
    Simulate bar-by-bar execution with 1-bar signal lag.

    Signal at bar T → enter at bar T+1 close.
    TP/SL checked every bar after entry.
    """
    n      = len(signals)
    equity = np.full(n + 1, initial_capital)
    trades: list[Trade] = []

    position: Trade | None = None

    for i in range(1, n):
        price = prices[i]
        eq    = equity[i]

        # ── manage open position ───────────────────────────────────────────
        if position is not None:
            hit_tp = (position.direction ==  1 and price >= position.tp) or \
                     (position.direction == -1 and price <= position.tp)
            hit_sl = (position.direction ==  1 and price <= position.sl) or \
                     (position.direction == -1 and price >= position.sl)

            if hit_tp or hit_sl:
                exit_price  = position.tp if hit_tp else position.sl
                exit_reason = "TP" if hit_tp else "SL"
                raw_pnl     = position.direction * (exit_price / position.entry - 1)
                net_pnl     = raw_pnl - 2 * TAKER_FEE
                position.bar_out    = i
                position.exit       = exit_price
                position.pnl_pct    = net_pnl
                position.exit_reason = exit_reason
                trades.append(position)
                equity[i + 1] = eq * (1 + net_pnl * risk_per_trade /
                                      abs(position.sl / position.entry - 1))
                position = None
                continue

        equity[i + 1] = equity[i]   # carry forward if no exit

        # ── check for new signal (1-bar lag: signal from bar i-1) ─────────
        if position is None and signals[i - 1] != 0:
            direction = int(signals[i - 1])
            entry     = price * (1 + direction * TAKER_FEE)   # fee on entry
            tp        = tp_arr[i - 1]
            sl        = sl_arr[i - 1]

            # sanity: tp/sl must be on correct sides
            if direction == 1  and tp > entry and sl < entry:
                position = Trade(i, direction, entry, tp, sl)
            elif direction == -1 and tp < entry and sl > entry:
                position = Trade(i, direction, entry, tp, sl)

    # force-close any open position at last bar
    if position is not None:
        price = prices[-1]
        raw_pnl = position.direction * (price / position.entry - 1)
        net_pnl = raw_pnl - 2 * TAKER_FEE
        position.bar_out    = n - 1
        position.exit       = price
        position.pnl_pct    = net_pnl
        position.exit_reason = "EOD"
        trades.append(position)

    return EngineResult(equity=equity[:n], trades=trades, timestamps=timestamps)
