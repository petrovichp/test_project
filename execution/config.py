"""
ExecutionConfig — binds entry, exit, and sizing strategies per signal strategy.
EXECUTION_CONFIG maps strategy key → ExecutionConfig.

ATR-dynamic TP/SL:
  At median vol → TP=base_tp, SL=base_sl (same as previous fixed params)
  Scales proportionally with predicted ATR vs training median.
"""

from dataclasses import dataclass
from execution.exit   import FixedExit, ATRDynamicExit, ComboExit
from execution.sizing import FixedFraction, VolScaledSizer
from execution.entry  import MarketEntry, ConfirmEntry, SpreadEntry


@dataclass
class ExecutionConfig:
    entry:  object
    exit:   object
    sizing: object


EXECUTION_CONFIG = {

    # ── S1: Momentum — no time stop (let TP/SL decide), trail after BE ────────
    "S1_VolDir": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.020, base_sl_pct=0.007,
            breakeven_pct=0.005, time_stop_bars=0,   # no time stop for momentum
            trail_after_breakeven=True,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.008),
    ),

    # ── S2: Funding mean-reversion — tight TP (reversion is modest), 60-bar TS
    "S2_Funding": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.008, base_sl_pct=0.005,   # mean-reversion: small TP
            breakeven_pct=0.003, time_stop_bars=60,
            trail_after_breakeven=False,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.006),
    ),

    # ── S3: BB reversion — tight TP, 30-bar TS ───────────────────────────────
    "S3_BBRevert": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.008, base_sl_pct=0.004,   # mean-reversion: small TP
            breakeven_pct=0.002, time_stop_bars=30,
            trail_after_breakeven=False,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.006),
    ),

    # ── S4: MACD trend — no time stop, trail after BE ────────────────────────
    "S4_MACDTrend": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.025, base_sl_pct=0.008,
            breakeven_pct=0.006, time_stop_bars=0,   # no time stop
            trail_after_breakeven=True,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.008),
    ),

    # ── S6: Two-signal — no time stop, trail after BE ────────────────────────
    "S6_TwoSignal": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.025, base_sl_pct=0.008,
            breakeven_pct=0.005, time_stop_bars=0,
            trail_after_breakeven=True,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.008),
    ),

    # ── S7: OI divergence — mean-rev, tight TP, 45-bar TS ───────────────────
    "S7_OIDiverg": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.010, base_sl_pct=0.005,   # mean-reversion: tight TP
            breakeven_pct=0.003, time_stop_bars=45,
            trail_after_breakeven=False,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.007),
    ),

    # ── S8: Sustained taker — no time stop, trail after BE ───────────────────
    "S8_TakerFlow": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.015, base_sl_pct=0.006,
            breakeven_pct=0.004, time_stop_bars=0,
            trail_after_breakeven=True,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.007),
    ),

    # ── S10: Vol squeeze — wide TP, 120-bar TS ───────────────────────────────
    "S10_Squeeze": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.030, base_sl_pct=0.008,
            breakeven_pct=0.008, time_stop_bars=120,
            trail_after_breakeven=True,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.010),
    ),

    # ── S12: VWAP + vol — MarketEntry (conditions already selective), 30-bar TS
    "S12_VWAPVol": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.015, base_sl_pct=0.006,
            breakeven_pct=0.003, time_stop_bars=30,
            trail_after_breakeven=False,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.007),
    ),

    # ── S11: Basis momentum — slow mean-reversion-leaning, 180-bar TS ─────────
    "S11_Basis": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.015, base_sl_pct=0.006,
            breakeven_pct=0.004, time_stop_bars=180,
            trail_after_breakeven=False,   # mean-reversion: no trail
        ),
        sizing = VolScaledSizer(target_risk_pct=0.007),
    ),

    # ── S13: OB Disagreement — microstructure, 30-bar TS ─────────────────────
    "S13_OBDiv": ExecutionConfig(
        entry  = MarketEntry(),
        exit   = ComboExit(
            base_tp_pct=0.012, base_sl_pct=0.006,
            breakeven_pct=0.003, time_stop_bars=30,
            trail_after_breakeven=False,
        ),
        sizing = VolScaledSizer(target_risk_pct=0.007),
    ),

    # Killed (overlap with S8 / spot_imbalance, worse standalone): S5_OFIScalp, S9_LargeOrd
}
