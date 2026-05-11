"""
Z3 Step 1 — Standalone validation of S5, S9, S11, S13.

These strategies exist in strategy/agent.py but are commented as "Killed
(Sharpe < -20 on both splits, no recoverable signal)" and not registered
in STRATEGIES dict. This script monkey-patches them back in temporarily
and runs the standard backtest pipeline to get their current numbers.

Decision: even confirmed-dead strategies might add value as DQN actions
(the DQN learns when NOT to take them). But if standalone Sharpe is
catastrophic (< -10), the action-space expansion is probably not worth
the retrain cost.
"""
import sys, json, pathlib
from datetime import datetime

# Monkey-patch BEFORE backtest imports STRATEGIES
from strategy import agent as _agent

_KILLED = {
    "S5_OFISpike":  (_agent.strategy_5,  "Killed: OFI spike scalp"),
    "S9_LargeOrd":  (_agent.strategy_9,  "Killed: large-order imbalance"),
    "S11_Basis":    (_agent.strategy_11, "Killed: basis momentum"),
    "S13_OBDiv":    (_agent.strategy_13, "Killed: spot/perp OB disagreement"),
}
_DEFAULTS = {
    "S5_OFISpike":  {"ofi_sigma": 2.0,   "vol_floor": 0.40,
                       "tp_pct": 0.010, "sl_pct": 0.005, "trail_pct": 0.0},
    "S9_LargeOrd":  {"imb_sigma": 1.5,   "vol_floor": 0.45,
                       "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.0},
    "S11_Basis":    {"basis_sigma": 1.5,
                       "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.0},
    "S13_OBDiv":    {"imb_thresh": 0.10,
                       "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.0},
}
_agent.STRATEGIES   = {**_agent.STRATEGIES,   **_KILLED}
_agent.DEFAULT_PARAMS = {**_agent.DEFAULT_PARAMS, **_DEFAULTS}

# Also patch REGIME_GATES in backtest.run BEFORE it runs main()
from backtest import run as _backtest_run

for regime_model in _backtest_run.REGIME_GATES:
    for name in _KILLED:
        # Allow all regimes for the killed strategies
        _backtest_run.REGIME_GATES[regime_model][name] = {
            "trend_up", "trend_down", "chop", "ranging", "calm",
            "trend_bull", "trend_bear", "high_vol_chop", "fund_long", "fund_short",
        }

# Run the standard backtest
if __name__ == "__main__":
    print(f"\n{'='*100}\n  Z3 Step 1 — Standalone validation of killed strategies S5, S9, S11, S13\n{'='*100}\n")
    print(f"  Registered strategies (after monkey-patch): {len(_agent.STRATEGIES)}")
    for k in _agent.STRATEGIES:
        marker = " (killed-rewired)" if k in _KILLED else ""
        print(f"    {k}{marker}")
    print(f"\n  Running backtest.run.run('btc') ...\n")
    _backtest_run.run("btc")
