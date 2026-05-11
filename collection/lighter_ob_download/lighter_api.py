"""
Lighter DEX REST API client — public market-data endpoints only.

Base URL: https://mainnet.zklighter.elliot.ai/api/v1
No auth required for any method in this module.

Market IDs (perpetual futures):
  ETH = 0, BTC = 1, SOL = 2
"""

import time
import requests
import pandas as pd


class Lighter:
    BASE = "https://mainnet.zklighter.elliot.ai/api/v1"

    MARKETS = {"ETH": 0, "BTC": 1, "SOL": 2}

    def get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{self.BASE}{path}", params=params,
                         headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code", 200) != 200:
            raise RuntimeError(f"Lighter API error: {data}")
        return data

    def market_details(self, market_id: int) -> dict:
        """Market metadata, OI, last price, daily stats. Returns the inner dict."""
        raw = self.get("/orderBookDetails", {"market_id": market_id})
        return raw["order_book_details"][0]

    def orderbook(self, market_id: int, limit: int = 250) -> dict:
        """Individual resting orders. Aggregate by price before use."""
        return self.get("/orderBookOrders", {"market_id": market_id, "limit": limit})

    def candles(self, market_id: int, resolution: str = "1m",
                count_back: int = 3) -> dict:
        """
        OHLCV candles. resolution: '1m' | '1h' | '1d'.
        All three timestamp params are required by the API.
        count_back=3 so index [-2] is always a closed bar.
        """
        now_ms = int(time.time() * 1000)
        # window wide enough to always contain count_back closed bars
        spans = {"1m": 10 * 60_000, "1h": 48 * 3600_000, "1d": 10 * 86400_000}
        start_ms = now_ms - spans.get(resolution, 10 * 60_000)
        return self.get("/candles", {
            "market_id":       market_id,
            "resolution":      resolution,
            "start_timestamp": start_ms,
            "end_timestamp":   now_ms,
            "count_back":      count_back,
        })

    def recent_trades(self, market_id: int, limit: int = 100) -> dict:
        """Last N trades. type: 'trade' | 'liquidation' | 'deleverage'."""
        return self.get("/recentTrades", {"market_id": market_id, "limit": limit})

    def funding(self, market_id: int, resolution: str = "1h",
                count_back: int = 3) -> dict:
        """Recent funding payments. resolution: '1h' | '1d'."""
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - 48 * 3600_000
        return self.get("/fundings", {
            "market_id":       market_id,
            "resolution":      resolution,
            "start_timestamp": start_ms,
            "end_timestamp":   now_ms,
            "count_back":      count_back,
        })


# ── OB parsing ────────────────────────────────────────────────────────────────

def parse_ob(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert raw orderBookOrders response to (bids, asks) DataFrames.

    Lighter returns individual resting orders; we aggregate by price level.
    Output columns: size (base units), amount (USD).
    Index: price (float), bids descending, asks ascending.
    """
    def _side(orders: list) -> pd.DataFrame:
        if not orders:
            return pd.DataFrame(columns=["size", "amount"])
        df = pd.DataFrame(orders)[["price", "remaining_base_amount"]]
        df["price"] = df["price"].astype(float)
        df["size"]  = df["remaining_base_amount"].astype(float)
        agg = df.groupby("price")["size"].sum().rename("size").to_frame()
        agg["amount"] = agg.index * agg["size"]
        return agg

    bids = _side(raw.get("bids", [])).sort_index(ascending=False)
    asks = _side(raw.get("asks", [])).sort_index(ascending=True)
    return bids, asks


def parse_candle(raw: dict) -> dict:
    """
    Return OHLCV dict for the last closed bar.
    Uses index [-2]: last entry may be the currently-open bar.
    """
    candles = raw.get("c", [])
    if len(candles) < 2:
        raise ValueError(f"Need ≥2 candles, got {len(candles)}")
    c = candles[-2]
    return {
        "open":       float(c["o"]),
        "high":       float(c["h"]),
        "low":        float(c["l"]),
        "close":      float(c["c"]),
        "volume":     float(c["v"]),      # base volume
        "volume_usd": float(c["V"]),      # quote volume (USD)
    }


def parse_funding(raw: dict) -> float:
    """Return the latest hourly funding rate (float)."""
    fundings = raw.get("fundings", [])
    if not fundings:
        return 0.0
    return float(fundings[-1]["rate"])


def deal_imbalance(raw_trades: dict, lookback_ms: int = 60_000
                   ) -> tuple[float, float]:
    """
    Sell/buy dollar volume in the last `lookback_ms` ms.

    is_maker_ask=True  → ask was maker → buy aggressor → buy trade
    is_maker_ask=False → bid was maker → sell aggressor → sell trade
    """
    trades = raw_trades.get("trades", [])
    if not trades:
        return 0.0, 0.0

    df = pd.DataFrame(trades)
    df["timestamp"]    = df["timestamp"].astype(int)
    df["usd_amount"]   = df["usd_amount"].astype(float)
    df["is_maker_ask"] = df["is_maker_ask"].astype(bool)

    cutoff = df["timestamp"].max() - lookback_ms
    recent = df[df["timestamp"] >= cutoff]

    sell_usd = float(recent[~recent["is_maker_ask"]]["usd_amount"].sum())
    buy_usd  = float(recent[ recent["is_maker_ask"]]["usd_amount"].sum())
    return sell_usd, buy_usd


def liquidation_stats(raw_trades: dict) -> dict:
    """Rolling liquidation USD by side from recentTrades (type='liquidation')."""
    trades = raw_trades.get("trades", [])
    if not trades:
        return {"liq_buy_usd": 0.0, "liq_sell_usd": 0.0}

    df = pd.DataFrame(trades)
    df["usd_amount"] = df["usd_amount"].astype(float)
    liqs = df[df["type"] == "liquidation"]

    # is_maker_ask=True → liquidated a long (sell-side liq), False → short liq
    liq_sell = float(liqs[ liqs["is_maker_ask"]]["usd_amount"].sum())
    liq_buy  = float(liqs[~liqs["is_maker_ask"]]["usd_amount"].sum())
    return {"liq_buy_usd": liq_buy, "liq_sell_usd": liq_sell}
