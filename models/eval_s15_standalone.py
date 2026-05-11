"""
Z3 Step 6 — Standalone validation of S15_VolBreakout (strategy_14).

Per the plan: if standalone Sharpe ≥ -10 on both val and test (similar
to currently-deployed strategies like S10/S2/S7), proceed to retrain
the v8 baseline with S15 added (action space 12 → 13).
"""
import sys
from backtest import run as _backtest_run

# Allow S15 in all regimes (give it a fair test)
for regime_model in _backtest_run.REGIME_GATES:
    _backtest_run.REGIME_GATES[regime_model]["S15_VolBreakout"] = {
        "trend_up", "trend_down", "chop", "ranging", "calm",
        "trend_bull", "trend_bear", "high_vol_chop", "fund_long", "fund_short",
    }

print(f"\n{'='*100}\n  Z3 Step 6 — Standalone validation of S15_VolBreakout\n{'='*100}\n")
_backtest_run.run("btc")
