"""
Trading strategies 1–6.

Each strategy receives a DataFrame with columns:
  price, atr_pred,
  p_up_60, p_dn_60, p_up_100, p_dn_100, vol_pred,
  bb_pct_b, bb_width, macd_hist, rsi_6, rsi_14,
  ofi_perp_10_r15, ofi_perp_10, taker_imb_5, taker_net_15,
  fund_rate, fund_mom_480, fund_z (z-score of fund_rate),
  ret_sma_200, vwap_dev_1440, sma_50, sma_200

Returns three arrays of length n:
  signals  (+1 long, -1 short, 0 flat)
  tp_pct   (take-profit as fraction of entry price, e.g. 0.008 = 0.8%)
  sl_pct   (stop-loss  as fraction of entry price, e.g. 0.004 = 0.4%)
"""

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int = 480) -> pd.Series:
    mu  = series.rolling(window, min_periods=window // 2).mean()
    sig = series.rolling(window, min_periods=window // 2).std()
    return (series - mu) / (sig + 1e-12)


def _rolling_pct(series: pd.Series, window: int = 1440) -> pd.Series:
    """Percentile rank of each value within a rolling window."""
    def pct_rank(x):
        return (x[:-1] < x[-1]).mean() if len(x) > 1 else 0.5
    return series.rolling(window, min_periods=window // 2).apply(pct_rank, raw=True)


# ── Strategy 1: Volatility-Filtered Direction ─────────────────────────────────

def strategy_1(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long when vol model predicts high ATR AND direction model says up.
    Short when vol model predicts high ATR AND direction model says down.
    Filtered by RSI not overextended and BB not at extreme.
    """
    vp  = params.get("vol_thresh", 0.60)
    dp  = params.get("dir_thresh", 0.50)
    tp_mult = params.get("tp_mult", 1.5)
    sl_mult = params.get("sl_mult", 0.8)

    price = df["price"].values
    atr   = df["atr_pred"].values

    long_cond  = (
        (df["vol_pred"]  > vp) &
        (df["p_up_60"]   > dp) &
        (df["bb_pct_b"]  < 0.70) &
        (df["rsi_14"]    < 65)
    ).values

    short_cond = (
        (df["vol_pred"]  > vp) &
        (df["p_dn_60"]   > dp) &
        (df["bb_pct_b"]  > 0.30) &
        (df["rsi_14"]    > 35)
    ).values

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 2: Funding Rate Mean-Reversion ───────────────────────────────────

def strategy_2(df: pd.DataFrame, params: dict) -> tuple:
    """
    Short when funding extreme positive + MACD weakening.
    Long  when funding extreme negative + MACD strengthening.
    """
    fund_z_thresh = params.get("fund_z_thresh", 2.0)
    tp_mult = params.get("tp_mult", 1.5)
    sl_mult = params.get("sl_mult", 1.0)

    price   = df["price"].values
    atr     = df["atr_pred"].values
    fund_z  = _rolling_zscore(df["fund_rate"], window=480).values

    short_cond = (
        (fund_z  >  fund_z_thresh) &
        (df["fund_mom_480"] > 0).values &
        (df["macd_hist"] < 0).values &
        (df["rsi_14"] > 60).values
    )
    long_cond  = (
        (fund_z  < -fund_z_thresh) &
        (df["fund_mom_480"] < 0).values &
        (df["macd_hist"] > 0).values &
        (df["rsi_14"] < 40).values
    )

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 3: Bollinger Band Mean-Reversion ─────────────────────────────────

def strategy_3(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long at lower BB + OFI turning positive + below VWAP (low vol regime).
    Short at upper BB + OFI turning negative + above VWAP (low vol regime).
    """
    vol_ceil    = params.get("vol_ceil", 0.50)
    ofi_thresh  = params.get("ofi_thresh", 0.0)
    tp_mult     = params.get("tp_mult", 1.0)
    sl_mult     = params.get("sl_mult", 0.5)

    price = df["price"].values
    atr   = df["atr_pred"].values

    long_cond  = (
        (df["bb_pct_b"]          < 0.05) &
        (df["ofi_perp_10_r15"]   > ofi_thresh) &
        (df["taker_imb_5"]       > -0.20) &
        (df["vol_pred"]          < vol_ceil) &
        (df["vwap_dev_1440"]     < -0.002)
    ).values
    short_cond = (
        (df["bb_pct_b"]          > 0.95) &
        (df["ofi_perp_10_r15"]   < -ofi_thresh) &
        (df["taker_imb_5"]       < 0.20) &
        (df["vol_pred"]          < vol_ceil) &
        (df["vwap_dev_1440"]     > 0.002)
    ).values

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 4: MACD + SMA Trend Following ────────────────────────────────────

def strategy_4(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long when price > sma_50 > sma_200, MACD expanding up, vol and direction confirm.
    Short when price < sma_50 < sma_200, MACD expanding down.
    """
    vp      = params.get("vol_thresh", 0.60)
    dp      = params.get("dir_thresh", 0.45)
    sma_dev = params.get("sma_dev",    0.002)
    tp_mult = params.get("tp_mult", 2.0)
    sl_mult = params.get("sl_mult", 1.0)

    price = df["price"].values
    atr   = df["atr_pred"].values

    macd_expand_up   = (df["macd_hist"] > 0) & (df["macd_hist"] > df["macd_hist"].shift(1))
    macd_expand_down = (df["macd_hist"] < 0) & (df["macd_hist"] < df["macd_hist"].shift(1))

    long_cond  = (
        macd_expand_up &
        (df["ret_sma_200"]  >  sma_dev) &
        (df["price"]        >  df["sma_50"]) &
        (df["sma_50"]       >  df["sma_200"]) &
        (df["vol_pred"]     >  vp) &
        (df["p_up_60"]      >  dp)
    ).values
    short_cond = (
        macd_expand_down &
        (df["ret_sma_200"]  < -sma_dev) &
        (df["price"]        <  df["sma_50"]) &
        (df["sma_50"]       <  df["sma_200"]) &
        (df["vol_pred"]     >  vp) &
        (df["p_dn_60"]      >  dp)
    ).values

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 5: OFI Momentum Scalp ───────────────────────────────────────────

def strategy_5(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long on abnormal positive OFI spike + taker confirms + RSI not overbought.
    Short on abnormal negative OFI spike.
    Small TP/SL — scalp trade, 15-bar max hold.
    """
    ofi_sigma   = params.get("ofi_sigma", 2.0)
    vol_floor   = params.get("vol_floor", 0.40)
    tp_mult     = params.get("tp_mult", 0.8)
    sl_mult     = params.get("sl_mult", 0.4)

    price = df["price"].values
    atr   = df["atr_pred"].values

    ofi_std   = df["ofi_perp_10_r15"].rolling(60, min_periods=20).std()
    ofi_z     = df["ofi_perp_10_r15"] / (ofi_std + 1e-12)

    long_cond  = (
        (ofi_z              >  ofi_sigma) &
        (df["ofi_perp_10"]  >  0) &
        (df["taker_net_15"] >  0) &
        (df["rsi_6"]        <  70) &
        (df["vol_pred"]     >  vol_floor)
    ).values
    short_cond = (
        (ofi_z              < -ofi_sigma) &
        (df["ofi_perp_10"]  <  0) &
        (df["taker_net_15"] <  0) &
        (df["rsi_6"]        >  30) &
        (df["vol_pred"]     >  vol_floor)
    ).values

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 6: Two-Signal High-Precision ────────────────────────────────────

def strategy_6(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long only when direction, MACD, RSI, OFI, and vol all agree.
    Very low signal rate (~3–5%), highest expected precision.
    """
    vp      = params.get("vol_thresh", 0.55)
    dp_req  = params.get("dir_req",    0.70)    # required direction score
    dp_opp  = params.get("dir_opp",    0.20)    # max allowed opposite score
    tp_mult = params.get("tp_mult", 2.0)
    sl_mult = params.get("sl_mult", 1.0)

    price = df["price"].values
    atr   = df["atr_pred"].values

    long_cond  = (
        (df["p_up_60"]           > dp_req) &
        (df["p_dn_60"]           < dp_opp) &
        (df["macd_hist"]         > 0) &
        (df["rsi_14"]            > 45) & (df["rsi_14"] < 65) &
        (df["ofi_perp_10_r15"]   > 0) &
        (df["vol_pred"]          > vp)
    ).values
    short_cond = (
        (df["p_dn_60"]           > dp_req) &
        (df["p_up_60"]           < dp_opp) &
        (df["macd_hist"]         < 0) &
        (df["rsi_14"]            < 55) & (df["rsi_14"] > 35) &
        (df["ofi_perp_10_r15"]   < 0) &
        (df["vol_pred"]          > vp)
    ).values

    tp_pct  = params.get("tp_pct", 0.010)
    sl_pct  = params.get("sl_pct", 0.004)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 7: OI vs Price Divergence ───────────────────────────────────────

def strategy_7(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long when OI rising fast but price falling (short squeeze setup).
    Short when OI falling fast but price rising (long unwinding).
    Uses pre-computed oi_price_div_15 feature.
    """
    div_sigma   = params.get("div_sigma",  1.5)
    vol_floor   = params.get("vol_floor",  0.45)

    div_std = df["oi_price_div_15"].rolling(120, min_periods=40).std()
    div_z   = df["oi_price_div_15"] / (div_std + 1e-12)

    long_cond  = (
        (div_z              >  div_sigma) &
        (df["taker_net_15"] >  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values
    short_cond = (
        (div_z              < -div_sigma) &
        (df["taker_net_15"] <  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values

    tp_pct  = params.get("tp_pct", 0.020)
    sl_pct  = params.get("sl_pct", 0.007)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 8: Sustained Taker Flow Momentum ────────────────────────────────

def strategy_8(df: pd.DataFrame, params: dict) -> tuple:
    """
    Sustained taker dominance over 60 bars (institutional accumulation).
    Different from S5 which fires on single-bar OFI spikes.
    """
    taker_sigma = params.get("taker_sigma", 1.0)
    vol_floor   = params.get("vol_floor",   0.50)

    taker_std  = df["taker_net_60"].rolling(480, min_periods=120).std()
    taker60_z  = df["taker_net_60"] / (taker_std + 1e-12)
    taker30_z  = df["taker_net_30"] / (taker_std + 1e-12)

    long_cond  = (
        (taker60_z          >  taker_sigma) &
        (taker30_z          >  0.3) &
        (df["ofi_perp_10"]  >  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values
    short_cond = (
        (taker60_z          < -taker_sigma) &
        (taker30_z          < -0.3) &
        (df["ofi_perp_10"]  <  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values

    tp_pct  = params.get("tp_pct", 0.015)
    sl_pct  = params.get("sl_pct", 0.006)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 9: Large Order Imbalance ────────────────────────────────────────

def strategy_9(df: pd.DataFrame, params: dict) -> tuple:
    """
    Net large-order imbalance (bid count - ask count) as institutional proxy.
    Spot leads: when large spot bids dominate, perp should follow.
    """
    imb_sigma = params.get("imb_sigma",  1.5)
    vol_floor = params.get("vol_floor",  0.45)

    spot_net = df["spot_large_bid_count"] - df["spot_large_ask_count"]
    perp_net = df["perp_large_bid_count"] - df["perp_large_ask_count"]

    spot_std = spot_net.rolling(120, min_periods=30).std()
    spot_z   = spot_net / (spot_std + 1e-6)

    long_cond  = (
        (spot_z             >  imb_sigma) &
        (perp_net           >  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values
    short_cond = (
        (spot_z             < -imb_sigma) &
        (perp_net           <  0) &
        (df["vol_pred"]     >  vol_floor)
    ).values

    tp_pct  = params.get("tp_pct", 0.020)
    sl_pct  = params.get("sl_pct", 0.007)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 10: Vol Squeeze Breakout ────────────────────────────────────────

def strategy_10(df: pd.DataFrame, params: dict) -> tuple:
    """
    bb_width compressed below recent average (squeeze) → breakout on MACD cross.
    Captures ranging→trending regime transitions.
    """
    squeeze_ratio = params.get("squeeze_ratio", 0.65)  # bb_width < X * rolling mean

    bbw_mean     = df["bb_width"].rolling(480, min_periods=120).mean()
    in_squeeze   = df["bb_width"] < (bbw_mean * squeeze_ratio)
    recent_squeeze = in_squeeze.rolling(30, min_periods=1).max().astype(bool)

    macd_cross_up   = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    macd_cross_down = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)

    long_cond  = (recent_squeeze & macd_cross_up  & (df["taker_net_15"] > 0)).values
    short_cond = (recent_squeeze & macd_cross_down & (df["taker_net_15"] < 0)).values

    tp_pct  = params.get("tp_pct", 0.030)
    sl_pct  = params.get("sl_pct", 0.008)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 11: Spot-Perp Basis Momentum ────────────────────────────────────

def strategy_11(df: pd.DataFrame, params: dict) -> tuple:
    """
    diff_price z-score (spot ask - perp bid) as basis momentum signal.
    Expanding basis → futures premium growing → follow direction.
    Complement to S2 (which fades extremes; this follows into them).
    """
    basis_sigma = params.get("basis_sigma", 1.5)

    basis_mean = df["diff_price"].rolling(480, min_periods=120).mean()
    basis_std  = df["diff_price"].rolling(480, min_periods=120).std()
    basis_z    = (df["diff_price"] - basis_mean) / (basis_std + 1e-8)

    long_cond  = (
        (basis_z            >  basis_sigma) &
        (df["fund_mom_480"] >  0)
    ).values
    short_cond = (
        (basis_z            < -basis_sigma) &
        (df["fund_mom_480"] <  0)
    ).values

    tp_pct  = params.get("tp_pct", 0.020)
    sl_pct  = params.get("sl_pct", 0.007)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 12: VWAP Deviation + Volume Confirmation ────────────────────────

def strategy_12(df: pd.DataFrame, params: dict) -> tuple:
    """
    Price far from 4h VWAP + high volume + taker turning = overshoot reversion.
    Fixes S3's weakness (BB alone, no volume confirmation).
    """
    vwap_thresh = params.get("vwap_thresh", 0.008)
    vol_sigma   = params.get("vol_sigma",   1.0)
    vol_ceil    = params.get("vol_ceil",    0.60)

    # taker imbalance turning (short-window flipping vs longer window)
    turning_pos = (df["taker_imb_5"] > 0) & (df["taker_imb_30"] < 0)
    turning_neg = (df["taker_imb_5"] < 0) & (df["taker_imb_30"] > 0)

    long_cond  = (
        (df["vwap_dev_240"] < -vwap_thresh) &
        (df["vol_z_spot_60"] >  vol_sigma) &
        turning_pos &
        (df["vol_pred"]     <  vol_ceil)
    ).values
    short_cond = (
        (df["vwap_dev_240"] >  vwap_thresh) &
        (df["vol_z_spot_60"] >  vol_sigma) &
        turning_neg &
        (df["vol_pred"]     <  vol_ceil)
    ).values

    tp_pct  = params.get("tp_pct", 0.015)
    sl_pct  = params.get("sl_pct", 0.006)
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── Strategy 13: Spot/Perp Order Book Divergence ─────────────────────────────

def strategy_13(df: pd.DataFrame, params: dict) -> tuple:
    """
    Spot and perp order book imbalances disagree → spot leads, perp follows.
    When spot bid-heavy but perp ask-heavy: long (perp will catch up).
    """
    imb_thresh = params.get("imb_thresh", 0.10)

    spot_buying_perp_selling = (
        (df["spot_imbalance"] >  imb_thresh) &
        (df["perp_imbalance"] < -imb_thresh) &
        (df["taker_imb_5"]   >  0)
    )
    spot_selling_perp_buying = (
        (df["spot_imbalance"] < -imb_thresh) &
        (df["perp_imbalance"] >  imb_thresh) &
        (df["taker_imb_5"]   <  0)
    )

    tp_pct  = params.get("tp_pct", 0.015)
    sl_pct  = params.get("sl_pct", 0.005)
    signals = np.where(spot_buying_perp_selling.values, 1,
               np.where(spot_selling_perp_buying.values, -1, 0))
    return signals, np.full(len(signals), tp_pct), np.full(len(signals), sl_pct)


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES = {
    "S1_VolDir":    (strategy_1,  "Volatility-Filtered Direction"),
    "S2_Funding":   (strategy_2,  "Funding Rate Mean-Reversion"),
    "S3_BBRevert":  (strategy_3,  "Bollinger Band Mean-Reversion"),
    "S4_MACDTrend": (strategy_4,  "MACD + SMA Trend Following"),
    "S6_TwoSignal": (strategy_6,  "Two-Signal High-Precision"),
    "S7_OIDiverg":  (strategy_7,  "OI vs Price Divergence"),
    "S8_TakerFlow": (strategy_8,  "Sustained Taker Flow Momentum"),
    "S10_Squeeze":  (strategy_10, "Vol Squeeze Breakout"),
    "S12_VWAPVol":  (strategy_12, "VWAP Deviation + Volume"),
    # Resurrected 2026-05-11 (Z3 Step 1 + Step 4): unique signal types not in current 9.
    # Standalone Sharpes (val/test): S11 -5.66/-8.28, S13 -2.81/-2.81 — similar to
    # currently-used S2/S10 which are weak standalone but useful in DQN action space.
    "S11_Basis":    (strategy_11, "Basis Momentum (perp-spot z-score)"),
    "S13_OBDiv":    (strategy_13, "Spot/Perp OB Imbalance Disagreement"),
    # Killed (overlap with S8 / spot_imbalance, worse standalone):
    #   S5_OFIScalp, S9_LargeOrd
}

DEFAULT_PARAMS = {
    # tp_pct / sl_pct are fractions of entry price
    # trail_pct: trailing SL ratchets below running price peak (0 = fixed SL — use for mean-reversion)
    # Momentum strategies (S1, S4, S6): trail_pct = sl_pct (ratchet locks in gains on runners)
    # Mean-reversion strategies (S2, S3): trail_pct = 0 (price overshoots then returns; trail exits early)
    # breakeven precision ≈ sl / (tp + sl - 0.0016 fees)
    "S1_VolDir":    {"vol_thresh": 0.60, "dir_thresh": 0.75,   # dir raised 0.70→0.75
                     "tp_pct": 0.020, "sl_pct": 0.007, "trail_pct": 0.007},
    "S2_Funding":   {"fund_z_thresh": 2.0,                     # mean-reversion: no trail
                     "tp_pct": 0.020, "sl_pct": 0.007, "trail_pct": 0.0},
    "S3_BBRevert":  {"vol_ceil": 0.50, "ofi_thresh": 0.0,      # mean-reversion: no trail
                     "tp_pct": 0.015, "sl_pct": 0.005, "trail_pct": 0.0},
    "S4_MACDTrend": {"vol_thresh": 0.60, "dir_thresh": 0.70, "sma_dev": 0.002,
                     "tp_pct": 0.025, "sl_pct": 0.008, "trail_pct": 0.008},
    "S6_TwoSignal": {"vol_thresh": 0.55, "dir_req": 0.70, "dir_opp": 0.20,
                     "tp_pct": 0.025, "sl_pct": 0.008, "trail_pct": 0.008},
    "S7_OIDiverg":  {"div_sigma": 1.5, "vol_floor": 0.45,
                     "tp_pct": 0.020, "sl_pct": 0.007, "trail_pct": 0.0},
    "S8_TakerFlow": {"taker_sigma": 1.0, "vol_floor": 0.50,
                     "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.005},
    "S10_Squeeze":  {"squeeze_ratio": 0.65,
                     "tp_pct": 0.030, "sl_pct": 0.008, "trail_pct": 0.010},
    "S12_VWAPVol":  {"vwap_thresh": 0.008, "vol_sigma": 1.0, "vol_ceil": 0.60,
                     "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.0},
    "S11_Basis":    {"basis_sigma": 1.5,                       # basis momentum: small TP, mean-reverting tendency
                     "tp_pct": 0.015, "sl_pct": 0.006, "trail_pct": 0.0},
    "S13_OBDiv":    {"imb_thresh": 0.10,                       # OB disagreement: tight bounds, microstructure
                     "tp_pct": 0.012, "sl_pct": 0.006, "trail_pct": 0.0},
}
