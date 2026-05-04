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
  signals (+1 long, -1 short, 0 flat)
  tp_arr  (take-profit price)
  sl_arr  (stop-loss price)
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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


# ── Strategy 6: Two-Signal High-Precision ────────────────────────────────────

def strategy_6(df: pd.DataFrame, params: dict) -> tuple:
    """
    Long only when direction, MACD, RSI, OFI, and vol all agree.
    Very low signal rate (~3–5%), highest expected precision.
    """
    vp      = params.get("vol_thresh", 0.55)
    dp_req  = params.get("dir_req",    0.55)    # required direction score
    dp_opp  = params.get("dir_opp",    0.30)    # max allowed opposite score
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

    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    tp_arr  = np.where(signals ==  1, price + tp_mult * atr,
              np.where(signals == -1, price - tp_mult * atr, price))
    sl_arr  = np.where(signals ==  1, price - sl_mult * atr,
              np.where(signals == -1, price + sl_mult * atr, price))
    return signals, tp_arr, sl_arr


# ── registry ──────────────────────────────────────────────────────────────────

STRATEGIES = {
    "S1_VolDir":    (strategy_1, "Volatility-Filtered Direction"),
    "S2_Funding":   (strategy_2, "Funding Rate Mean-Reversion"),
    "S3_BBRevert":  (strategy_3, "Bollinger Band Mean-Reversion"),
    "S4_MACDTrend": (strategy_4, "MACD + SMA Trend Following"),
    "S5_OFIScalp":  (strategy_5, "OFI Momentum Scalp"),
    "S6_TwoSignal": (strategy_6, "Two-Signal High-Precision"),
}

DEFAULT_PARAMS = {
    "S1_VolDir":    {"vol_thresh": 0.60, "dir_thresh": 0.50, "tp_mult": 1.5, "sl_mult": 0.8},
    "S2_Funding":   {"fund_z_thresh": 2.0, "tp_mult": 1.5, "sl_mult": 1.0},
    "S3_BBRevert":  {"vol_ceil": 0.50, "ofi_thresh": 0.0, "tp_mult": 1.0, "sl_mult": 0.5},
    "S4_MACDTrend": {"vol_thresh": 0.60, "dir_thresh": 0.45, "sma_dev": 0.002,
                     "tp_mult": 2.0, "sl_mult": 1.0},
    "S5_OFIScalp":  {"ofi_sigma": 2.0, "vol_floor": 0.40, "tp_mult": 0.8, "sl_mult": 0.4},
    "S6_TwoSignal": {"vol_thresh": 0.55, "dir_req": 0.55, "dir_opp": 0.30,
                     "tp_mult": 2.0, "sl_mult": 1.0},
}
