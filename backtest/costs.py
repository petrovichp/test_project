"""
OKX fee model.

Taker: 0.08% per side (market orders — used for entries/exits in backtest)
Maker: 0.02% per side (limit orders — not modelled here)
Round-trip taker cost: 0.16%
"""

TAKER_FEE = 0.0008   # 0.08%
MAKER_FEE = 0.0002   # 0.02%


def round_trip_cost(use_maker: bool = False) -> float:
    fee = MAKER_FEE if use_maker else TAKER_FEE
    return 2 * fee
