"""
Binance REST API client — public market-data endpoints only.

Spot base:    https://api.binance.com
Futures base: https://fapi.binance.com
No authentication required for any method in this module.

Symbols: BTC→"BTCUSDT", ETH→"ETHUSDT", SOL→"SOLUSDT"
"""

import requests
import pandas as pd


class Binance:
    SPOT_BASE  = "https://api.binance.com"
    PERP_BASE  = "https://fapi.binance.com"
    SYMBOLS    = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

    def _get(self, base: str, path: str, params: dict = None) -> dict | list:
        r = requests.get(f"{base}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    # ── spot ──────────────────────────────────────────────────────────────────
    def spot_depth(self, symbol: str, limit: int = 500) -> dict:
        return self._get(self.SPOT_BASE, "/api/v3/depth",
                         {"symbol": symbol, "limit": limit})

    def spot_klines(self, symbol: str, interval: str = "1m",
                    limit: int = 3) -> list:
        return self._get(self.SPOT_BASE, "/api/v3/klines",
                         {"symbol": symbol, "interval": interval, "limit": limit})

    def spot_trades(self, symbol: str, limit: int = 500) -> list:
        return self._get(self.SPOT_BASE, "/api/v3/trades",
                         {"symbol": symbol, "limit": limit})

    # ── futures ───────────────────────────────────────────────────────────────
    def perp_depth(self, symbol: str, limit: int = 500) -> dict:
        return self._get(self.PERP_BASE, "/fapi/v1/depth",
                         {"symbol": symbol, "limit": limit})

    def perp_klines(self, symbol: str, interval: str = "1m",
                    limit: int = 3) -> list:
        return self._get(self.PERP_BASE, "/fapi/v1/klines",
                         {"symbol": symbol, "interval": interval, "limit": limit})

    def perp_trades(self, symbol: str, limit: int = 500) -> list:
        return self._get(self.PERP_BASE, "/fapi/v1/trades",
                         {"symbol": symbol, "limit": limit})

    def premium_index(self, symbol: str) -> dict:
        """Current mark price, index price, funding rate, next funding time."""
        return self._get(self.PERP_BASE, "/fapi/v1/premiumIndex",
                         {"symbol": symbol})

    def open_interest(self, symbol: str) -> dict:
        return self._get(self.PERP_BASE, "/fapi/v1/openInterest",
                         {"symbol": symbol})

    def taker_ls_ratio(self, symbol: str, period: str = "5m",
                       limit: int = 1) -> list:
        """Taker buy/sell volume ratio (aggregated over period)."""
        return self._get(self.PERP_BASE, "/futures/data/takerlongshortRatio",
                         {"symbol": symbol, "period": period, "limit": limit})

    def ls_account_ratio(self, symbol: str, period: str = "5m",
                         limit: int = 1) -> list:
        return self._get(self.PERP_BASE,
                         "/futures/data/globalLongShortAccountRatio",
                         {"symbol": symbol, "period": period, "limit": limit})

    def ls_top_account_ratio(self, symbol: str, period: str = "5m",
                              limit: int = 1) -> list:
        return self._get(self.PERP_BASE,
                         "/futures/data/topLongShortAccountRatio",
                         {"symbol": symbol, "period": period, "limit": limit})

    def ls_top_position_ratio(self, symbol: str, period: str = "5m",
                               limit: int = 1) -> list:
        return self._get(self.PERP_BASE,
                         "/futures/data/topLongShortPositionRatio",
                         {"symbol": symbol, "period": period, "limit": limit})


# ── OB parsing ────────────────────────────────────────────────────────────────

def parse_ob(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert depth response to (bids, asks) DataFrames.
    Each level is [price_str, qty_str]. Columns: size, amount. Index: price.
    """
    def _side(levels: list) -> pd.DataFrame:
        if not levels:
            return pd.DataFrame(columns=["size", "amount"])
        df = pd.DataFrame(levels, columns=["price", "size"])
        df["price"]  = df["price"].astype(float)
        df["size"]   = df["size"].astype(float)
        df["amount"] = df["price"] * df["size"]
        return df.set_index("price")[["size", "amount"]]

    bids = _side(raw["bids"]).sort_index(ascending=False)
    asks = _side(raw["asks"]).sort_index(ascending=True)
    return bids, asks


def parse_kline(klines: list) -> dict:
    """
    Last closed bar from klines response (positional array, index -2).
    Includes taker buy volume from kline fields [9]/[10].
    """
    if len(klines) < 2:
        raise ValueError(f"Need ≥2 klines, got {len(klines)}")
    k = klines[-2]
    total_base  = float(k[5])
    total_quote = float(k[7])
    taker_buy_base  = float(k[9])
    taker_buy_quote = float(k[10])
    return {
        "open":             float(k[1]),
        "high":             float(k[2]),
        "low":              float(k[3]),
        "close":            float(k[4]),
        "volume":           total_base,
        "volume_usd":       total_quote,
        "taker_buy_volume": taker_buy_base,
        "taker_buy_usd":    taker_buy_quote,
        "taker_sell_volume": total_base  - taker_buy_base,
        "taker_sell_usd":    total_quote - taker_buy_quote,
        "num_trades":       int(k[8]),
    }


def deal_imbalance(trades: list, lookback_ms: int = 60_000) -> tuple[float, float]:
    """
    Sell/buy USD volume in last lookback_ms ms.
    isBuyerMaker=True → sell aggressor; False → buy aggressor.
    """
    if not trades:
        return 0.0, 0.0
    df = pd.DataFrame(trades)
    df["time"]        = df["time"].astype(int)
    df["quoteQty"]    = df["quoteQty"].astype(float)
    df["isBuyerMaker"] = df["isBuyerMaker"].astype(bool)
    cutoff = df["time"].max() - lookback_ms
    recent = df[df["time"] >= cutoff]
    sell_usd = float(recent[ recent["isBuyerMaker"]]["quoteQty"].sum())
    buy_usd  = float(recent[~recent["isBuyerMaker"]]["quoteQty"].sum())
    return sell_usd, buy_usd
