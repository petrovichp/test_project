"""
OKX orderbook downloader v3 — enriched data collection.

Improvements over v2:
  ob_depth_span_spot / ob_depth_span_perp
      Dollar range covered by the 200 OB bins.
      v2 computed this (as `spot_span`) but never stored it.
      Enables true price-level features: bin_price = (i/200) * depth_span from mid.

  microprice_spot / microprice_perp
      Weighted mid: (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)
      More accurate short-term fair value than simple (bid+ask)/2.

  Full OHLC from 1m candle
      spot_open, spot_high, spot_low, spot_close
      perp_open, perp_high, perp_low, perp_close
      v2 only stored minute volume.

  Liquidity within price bands (computed from raw OB before binning)
      {spot|perp}_{bid|ask}_liq_{0.1|0.5|1.0}pct
      Dollar liquidity available within ±0.1%, ±0.5%, ±1.0% of mid.
      8 columns total — direct support/resistance measurement.

  Wall detection (largest single order near mid)
      {spot|perp}_{bid|ask}_wall_price  — price level of largest order in top 100 levels
      {spot|perp}_{bid|ask}_wall_size   — BTC size of that order
      8 columns — identifies where large passive orders sit.

  diff_price (perp_ask - spot_ask) — kept from v2 for continuity.

All 800 OB bin columns (spot/perp × bids/asks × 200 bins) are retained.
Schema is backward compatible — v3 adds columns, keeps all v2 columns.

Deploy: same as v2 — Google Cloud Functions, HTTP trigger, 1-min scheduler.
Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, EXC, ORDER_BOOK_TYPE, TOCKENONE, TOCKENTWO
"""

import functions_framework
import os
import time
import pandas as pd
from datetime import datetime
import mysql.connector
import hmac
import hashlib
import requests
import warnings
from typing import Dict, Optional, List, Callable
import json

warnings.filterwarnings('ignore')


# ── DB helpers (unchanged from v2) ───────────────────────────────────────────

def db_check_table_exists(connection, table_name: str) -> bool:
    cursor = connection.cursor()
    cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
    result = cursor.fetchone()
    cursor.close()
    return result is not None


def db_create_table_from_dict(connection, table_name: str, data_dict: dict):
    cursor = connection.cursor()
    columns = ", ".join([f"`{col}` DOUBLE" for col in data_dict if col != "timestamp"])
    query = (f"CREATE TABLE IF NOT EXISTS `{table_name}` "
             f"(`timestamp` BIGINT PRIMARY KEY, {columns});")
    cursor.execute(query)
    connection.commit()
    cursor.close()


def db_insert_data_from_dict(connection, table_name: str, data_dict: dict):
    cursor = connection.cursor()
    cols   = ", ".join([f"`{k}`" for k in data_dict.keys()])
    vals   = ", ".join(["%s"] * len(data_dict))
    cursor.execute(f"INSERT INTO `{table_name}` ({cols}) VALUES ({vals})",
                   list(data_dict.values()))
    connection.commit()
    cursor.close()


# ── OKX API client (unchanged from v2) ───────────────────────────────────────

class OKXMarketData:
    def __init__(self):
        self.base_url = "https://www.okx.com"

    def _get(self, endpoint: str, params: dict = None) -> dict:
        r = requests.get(f"{self.base_url}{endpoint}", params=params,
                         headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_full_orderbook(self, inst_id: str, depth: int = 5000) -> dict:
        return self._get("/api/v5/market/books-full", {"instId": inst_id, "sz": depth})

    def get_candles(self, inst_id: str, bar: str = "1m", limit: int = 2) -> dict:
        return self._get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})

    def get_trades(self, inst_id: str, limit: int = 500) -> dict:
        return self._get("/api/v5/market/trades", {"instId": inst_id, "limit": limit})

    def get_open_interest(self, inst_id: str) -> dict:
        return self._get("/api/v5/public/open-interest", {"instId": inst_id})

    def get_funding_rate(self, inst_id: str) -> dict:
        return self._get("/api/v5/public/funding-rate", {"instId": inst_id})

    def get_taker_volume(self, inst_id: str, period: str = "5m", unit: str = "0") -> dict:
        return self._get("/api/v5/rubik/stat/taker-volume-contract",
                         {"instId": inst_id, "period": period, "unit": unit, "limit": 2})


# ── OB parsing ────────────────────────────────────────────────────────────────

def parse_orderbook(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bids_df, asks_df) with columns [price, size, amount, cum_amount]."""
    bids = pd.DataFrame(raw["data"][0]["bids"], columns=["price", "size", "orderCount"])
    asks = pd.DataFrame(raw["data"][0]["asks"], columns=["price", "size", "orderCount"])
    for df in (bids, asks):
        df["price"]  = df["price"].astype(float)
        df["size"]   = df["size"].astype(float)
        df["amount"] = df["price"] * df["size"]
        df["cum_amount"] = df["amount"].cumsum()
    bids = bids.set_index("price").sort_index(ascending=False)
    asks = asks.set_index("price").sort_index(ascending=True)
    return bids, asks


# ── OB processing — binning into 200 equal-width bins (same as v2) ────────────

def process_order_book(bids: pd.DataFrame, asks: pd.DataFrame,
                       steps: int = 200) -> tuple:
    """
    Returns (binned_bids, binned_asks, span_dollars).
    span_dollars = dollar range covered by the 200 bins — stored in v3, discarded in v2.
    """
    bids_span = bids.index[0] - bids.index[-1]
    asks_span = asks.index[-1] - asks.index[0]
    span      = min(bids_span, asks_span)

    bid_bins = pd.interval_range(start=bids.index[0] - span,
                                  end=bids.index[0], periods=steps)
    ask_bins = pd.interval_range(start=asks.index[0],
                                  end=asks.index[0] + span, periods=steps)

    binned_bids = bids.groupby(pd.cut(bids.index, bins=bid_bins)).agg(
        {"size": "sum", "amount": "sum", "cum_amount": "last"})
    binned_asks = asks.groupby(pd.cut(asks.index, bins=ask_bins)).agg(
        {"size": "sum", "amount": "sum", "cum_amount": "last"})

    binned_bids.index = list(range(steps - 1, -1, -1))
    binned_asks.index = list(range(steps))

    return binned_bids, binned_asks, float(span)


# ── NEW: price-level features from raw OB ─────────────────────────────────────

def liquidity_within_pct(bids: pd.DataFrame, asks: pd.DataFrame,
                          mid: float, pct: float) -> tuple[float, float]:
    """Dollar liquidity (amount) within ±pct of mid on each side."""
    bid_liq = bids[bids.index >= mid * (1 - pct)]["amount"].sum()
    ask_liq = asks[asks.index <= mid * (1 + pct)]["amount"].sum()
    return float(bid_liq), float(ask_liq)


def microprice(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    """
    Weighted mid using best bid/ask quantities.
    microprice = (best_bid * ask_qty + best_ask * bid_qty) / (bid_qty + ask_qty)
    """
    best_bid      = bids.index[0]
    best_ask      = asks.index[0]
    bid_qty       = float(bids["size"].iloc[0])
    ask_qty       = float(asks["size"].iloc[0])
    total_qty     = bid_qty + ask_qty
    if total_qty == 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_qty + best_ask * bid_qty) / total_qty


def wall_detection(side: pd.DataFrame, top_n: int = 100) -> tuple[float, float]:
    """
    Largest single order (by $ amount) within top_n price levels.
    Returns (price_level, btc_size).
    """
    top = side.head(top_n)
    if top.empty:
        return 0.0, 0.0
    idx = top["amount"].idxmax()
    return float(idx), float(top.loc[idx, "size"])


# ── candle parsing ────────────────────────────────────────────────────────────

def parse_candle(raw: dict) -> dict:
    """Parse latest confirmed 1m candle. Returns OHLCV dict."""
    data = pd.DataFrame(raw["data"], columns=[
        "timestamp", "open", "high", "low", "close",
        "volume", "volCcy", "volCcyQuote", "confirm"
    ])
    data["confirm"] = data["confirm"].astype(float)
    confirmed = data[data["confirm"] == 1]
    if confirmed.empty:
        confirmed = data
    row = confirmed.iloc[0]
    return {
        "open":  float(row["open"]),
        "high":  float(row["high"]),
        "low":   float(row["low"]),
        "close": float(row["close"]),
        "vol_ccy_quote": float(row["volCcyQuote"]),
    }


# ── taker flow ────────────────────────────────────────────────────────────────

def recent_taker_flow(okx: OKXMarketData, inst_id: str) -> tuple[float, float]:
    """sell, buy amounts from most recent 5m taker volume."""
    raw = okx.get_taker_volume(inst_id=inst_id)["data"]
    if not raw:
        return 0.0, 0.0
    row = raw[0]
    return float(row[1]), float(row[2])


def recent_deal_imbalance(okx: OKXMarketData, inst_id: str,
                           limit: int = 500) -> tuple[float, float]:
    """sell_amount, buy_amount from last `limit` trades (within ~1 min)."""
    raw   = pd.DataFrame(okx.get_trades(inst_id=inst_id, limit=limit)["data"])
    raw["ts"]   = raw["ts"].astype(int)
    raw["px"]   = raw["px"].astype(float)
    raw["sz"]   = raw["sz"].astype(float)
    cutoff      = raw["ts"].iloc[0] - 60_000
    recent      = raw[raw["ts"] > cutoff].copy()
    recent["amount"] = recent["px"] * recent["sz"]
    sell = recent[recent["side"] == "sell"]["amount"].sum()
    buy  = recent[recent["side"] == "buy"]["amount"].sum()
    return float(sell), float(buy)


# ── main cloud function ───────────────────────────────────────────────────────

@functions_framework.http
def okx_ob_download_v3(request):
    payload    = request.get_json(silent=True) or {}
    TOKEN_ONE  = payload.get("TOCKENONE") or os.getenv("TOCKENONE")
    TOKEN_TWO  = os.getenv("TOCKENTWO")
    EXC        = os.getenv("EXC", "okx")
    OB_TYPE    = os.getenv("ORDER_BOOK_TYPE", "spotpepr")
    DEPTH      = 5000

    SPOT_PAIR  = f"{TOKEN_ONE}-{TOKEN_TWO}"
    PERP_PAIR  = f"{TOKEN_ONE}-{TOKEN_TWO}-SWAP"
    TABLE_ID   = f"{EXC.lower()}_{TOKEN_ONE.lower()}{TOKEN_TWO.lower()}_{OB_TYPE.lower()}_v3"

    db_config  = {
        "host":     os.getenv("DBHOST"),
        "user":     os.getenv("DBUSER"),
        "password": os.getenv("DBPASSW"),
        "database": os.getenv("DBSQL"),
        "port":     25060,
    }

    try:
        okx = OKXMarketData()

        # ── orderbooks ────────────────────────────────────────────────────
        spot_bids, spot_asks = parse_orderbook(okx.get_full_orderbook(SPOT_PAIR, DEPTH))
        perp_bids, perp_asks = parse_orderbook(okx.get_full_orderbook(PERP_PAIR, DEPTH))

        spot_ask_price = float(spot_asks.index[0])
        spot_bid_price = float(spot_bids.index[0])
        perp_ask_price = float(perp_asks.index[0])
        perp_bid_price = float(perp_bids.index[0])
        spot_mid       = (spot_ask_price + spot_bid_price) / 2
        perp_mid       = (perp_ask_price + perp_bid_price) / 2

        # ── bin OB into 200 equal-width bins (same as v2) ────────────────
        spot_bin_bids, spot_bin_asks, spot_depth_span = process_order_book(spot_bids, spot_asks)
        perp_bin_bids, perp_bin_asks, perp_depth_span = process_order_book(perp_bids, perp_asks)

        # ── candles ───────────────────────────────────────────────────────
        spot_candle = parse_candle(okx.get_candles(SPOT_PAIR, bar="1m", limit=2))
        perp_candle = parse_candle(okx.get_candles(PERP_PAIR, bar="1m", limit=2))

        # ── derivatives data ──────────────────────────────────────────────
        oi_usd    = float(okx.get_open_interest(PERP_PAIR)["data"][0]["oiUsd"])
        fund_rate = float(okx.get_funding_rate(PERP_PAIR)["data"][0]["fundingRate"])

        # ── taker flow ────────────────────────────────────────────────────
        taker_sell, taker_buy = recent_taker_flow(okx, PERP_PAIR)
        deal_sell_spot, deal_buy_spot = recent_deal_imbalance(okx, SPOT_PAIR)
        deal_sell_perp, deal_buy_perp = recent_deal_imbalance(okx, PERP_PAIR)

        # ── NEW: microprice ───────────────────────────────────────────────
        mp_spot = microprice(spot_bids, spot_asks)
        mp_perp = microprice(perp_bids, perp_asks)

        # ── NEW: price-level liquidity (±0.1%, ±0.5%, ±1.0% from mid) ────
        s_bid_01, s_ask_01 = liquidity_within_pct(spot_bids, spot_asks, spot_mid, 0.001)
        s_bid_05, s_ask_05 = liquidity_within_pct(spot_bids, spot_asks, spot_mid, 0.005)
        s_bid_10, s_ask_10 = liquidity_within_pct(spot_bids, spot_asks, spot_mid, 0.010)
        p_bid_01, p_ask_01 = liquidity_within_pct(perp_bids, perp_asks, perp_mid, 0.001)
        p_bid_05, p_ask_05 = liquidity_within_pct(perp_bids, perp_asks, perp_mid, 0.005)
        p_bid_10, p_ask_10 = liquidity_within_pct(perp_bids, perp_asks, perp_mid, 0.010)

        # ── NEW: wall detection ───────────────────────────────────────────
        s_wall_bid_px, s_wall_bid_sz = wall_detection(spot_bids)
        s_wall_ask_px, s_wall_ask_sz = wall_detection(spot_asks)
        p_wall_bid_px, p_wall_bid_sz = wall_detection(perp_bids)
        p_wall_ask_px, p_wall_ask_sz = wall_detection(perp_asks)

        # ── OB spread metrics ─────────────────────────────────────────────
        spot_spread_bps = (spot_ask_price - spot_bid_price) / spot_bid_price * 10000
        perp_spread_bps = (perp_ask_price - perp_bid_price) / perp_bid_price * 10000

        spot_total_bid = spot_bids["amount"].sum()
        spot_total_ask = spot_asks["amount"].sum()
        perp_total_bid = perp_bids["amount"].sum()
        perp_total_ask = perp_asks["amount"].sum()

        spot_imbalance = (spot_total_bid - spot_total_ask) / (spot_total_bid + spot_total_ask + 1e-12)
        perp_imbalance = (perp_total_bid - perp_total_ask) / (perp_total_bid + perp_total_ask + 1e-12)
        spot_bid_conc  = spot_bids["amount"].head(10).sum() / (spot_total_bid + 1e-12)
        spot_ask_conc  = spot_asks["amount"].head(10).sum() / (spot_total_ask + 1e-12)
        perp_bid_conc  = perp_bids["amount"].head(10).sum() / (perp_total_bid + 1e-12)
        perp_ask_conc  = perp_asks["amount"].head(10).sum() / (perp_total_ask + 1e-12)

        lrg_thr_spot = (spot_total_bid + spot_total_ask) * 0.01
        lrg_thr_perp = (perp_total_bid + perp_total_ask) * 0.01
        spot_lrg_bid = int((spot_bids["amount"] > lrg_thr_spot).sum())
        spot_lrg_ask = int((spot_asks["amount"] > lrg_thr_spot).sum())
        perp_lrg_bid = int((perp_bids["amount"] > lrg_thr_perp).sum())
        perp_lrg_ask = int((perp_asks["amount"] > lrg_thr_perp).sum())

        # ── assemble record ───────────────────────────────────────────────
        t_s = int(time.time())
        row = {"timestamp": t_s}

        # 800 OB bin columns (amounts normalised by spot_ask_price — same as v2)
        merged = pd.concat([
            spot_bin_bids["amount"].rename(lambda i: f"spot_bids_amount_{i}"),
            spot_bin_asks["amount"].rename(lambda i: f"spot_asks_amount_{i}"),
            perp_bin_bids["amount"].rename(lambda i: f"perp_bids_amount_{i}"),
            perp_bin_asks["amount"].rename(lambda i: f"perp_asks_amount_{i}"),
        ])
        for col, val in (merged / spot_ask_price).items():
            row[col] = float(val)

        # ── metadata (v2 columns, kept identical) ─────────────────────────
        row["oi_usd"]                = oi_usd / perp_ask_price
        row["fund_rate"]             = fund_rate
        row["spot_ask_price"]        = spot_ask_price
        row["spot_bid_price"]        = spot_bid_price
        row["perp_ask_price"]        = perp_ask_price
        row["perp_bid_price"]        = perp_bid_price
        row["span_spot_price"]       = (spot_ask_price - spot_bid_price) / spot_bid_price
        row["span_perp_price"]       = (perp_ask_price - perp_bid_price) / perp_bid_price
        row["spot_minute_volume"]    = spot_candle["vol_ccy_quote"] / spot_ask_price
        row["perp_minute_volume"]    = perp_candle["vol_ccy_quote"] / perp_ask_price
        row["spot_sell_buy_side_deals"] = deal_sell_spot / (deal_buy_spot + 1e-12)
        row["perp_sell_buy_side_deals"] = deal_sell_perp / (deal_buy_perp + 1e-12)
        row["spot_spread_bps"]       = spot_spread_bps
        row["spot_imbalance"]        = spot_imbalance
        row["spot_bid_concentration"]= spot_bid_conc
        row["spot_ask_concentration"]= spot_ask_conc
        row["spot_large_bid_count"]  = spot_lrg_bid
        row["spot_large_ask_count"]  = spot_lrg_ask
        row["perp_spread_bps"]       = perp_spread_bps
        row["perp_imbalance"]        = perp_imbalance
        row["perp_bid_concentration"]= perp_bid_conc
        row["perp_ask_concentration"]= perp_ask_conc
        row["perp_large_bid_count"]  = perp_lrg_bid
        row["perp_large_ask_count"]  = perp_lrg_ask
        row["taker_sell_buy_ratio"]  = taker_sell / (taker_buy + 1e-12)
        row["taker_sell"]            = taker_sell
        row["taker_buy"]             = taker_buy
        row["diff_price"]            = perp_ask_price - spot_ask_price

        # ── NEW v3 columns ────────────────────────────────────────────────

        # OB depth span — CRITICAL: enables true price-level feature engineering
        row["ob_depth_span_spot"]    = spot_depth_span        # $ range of 200 spot bins
        row["ob_depth_span_perp"]    = perp_depth_span        # $ range of 200 perp bins

        # Full OHLC from 1m candle
        row["spot_open"]             = spot_candle["open"]
        row["spot_high"]             = spot_candle["high"]
        row["spot_low"]              = spot_candle["low"]
        row["spot_close"]            = spot_candle["close"]
        row["perp_open"]             = perp_candle["open"]
        row["perp_high"]             = perp_candle["high"]
        row["perp_low"]              = perp_candle["low"]
        row["perp_close"]            = perp_candle["close"]

        # Microprice — weighted mid price
        row["microprice_spot"]       = mp_spot
        row["microprice_perp"]       = mp_perp

        # Price-level liquidity ($ amounts, not normalised — convert in features layer)
        row["spot_bid_liq_01pct"]    = s_bid_01
        row["spot_ask_liq_01pct"]    = s_ask_01
        row["spot_bid_liq_05pct"]    = s_bid_05
        row["spot_ask_liq_05pct"]    = s_ask_05
        row["spot_bid_liq_10pct"]    = s_bid_10
        row["spot_ask_liq_10pct"]    = s_ask_10
        row["perp_bid_liq_01pct"]    = p_bid_01
        row["perp_ask_liq_01pct"]    = p_ask_01
        row["perp_bid_liq_05pct"]    = p_bid_05
        row["perp_ask_liq_05pct"]    = p_ask_05
        row["perp_bid_liq_10pct"]    = p_bid_10
        row["perp_ask_liq_10pct"]    = p_ask_10

        # Wall detection — largest passive order near mid
        row["spot_wall_bid_price"]   = s_wall_bid_px
        row["spot_wall_bid_size"]    = s_wall_bid_sz
        row["spot_wall_ask_price"]   = s_wall_ask_px
        row["spot_wall_ask_size"]    = s_wall_ask_sz
        row["perp_wall_bid_price"]   = p_wall_bid_px
        row["perp_wall_bid_size"]    = p_wall_bid_sz
        row["perp_wall_ask_price"]   = p_wall_ask_px
        row["perp_wall_ask_size"]    = p_wall_ask_sz

        # ── persist ───────────────────────────────────────────────────────
        connection = mysql.connector.connect(**db_config)
        if not db_check_table_exists(connection, TABLE_ID):
            db_create_table_from_dict(connection, TABLE_ID, row)
        db_insert_data_from_dict(connection, TABLE_ID, row)
        connection.close()

    except Exception as e:
        print(f"ERROR: {e}")
        return f"ERROR: {e}", 500

    return f"OK {TABLE_ID} ts={t_s}"
