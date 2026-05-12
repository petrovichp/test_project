"""
Binance REST API client — public market-data endpoints only.

Spot base:    https://api.binance.com
Futures base: https://fapi.binance.com
No authentication required for any method in this module.

Symbols: BTC→"BTCUSDT", ETH→"ETHUSDT", SOL→"SOLUSDT"
"""

import math
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

    def mark_price_klines(self, symbol: str, interval: str = "1m",
                          limit: int = 3) -> list:
        return self._get(self.PERP_BASE, "/fapi/v1/markPriceKlines",
                         {"symbol": symbol, "interval": interval, "limit": limit})

    def index_price_klines(self, symbol: str, interval: str = "1m",
                           limit: int = 3) -> list:
        return self._get(self.PERP_BASE, "/fapi/v1/indexPriceKlines",
                         {"pair": symbol, "interval": interval, "limit": limit})

    def premium_index_klines(self, symbol: str, interval: str = "1m",
                             limit: int = 3) -> list:
        return self._get(self.PERP_BASE, "/fapi/v1/premiumIndexKlines",
                         {"symbol": symbol, "interval": interval, "limit": limit})


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


def parse_price_kline(klines: list) -> dict:
    """Last closed bar for mark/index/premium klines (OHLC only, no volume)."""
    if len(klines) < 2:
        raise ValueError(f"Need ≥2 klines, got {len(klines)}")
    k = klines[-2]
    return {"open": float(k[1]), "high": float(k[2]),
            "low":  float(k[3]), "close": float(k[4])}


def aggregate_buckets_hybrid(side: pd.DataFrame, n_fine: int = 10, n_coarse: int = 10,
                             nsf_fine: int = 4, nsf_coarse: int = 3) -> list[float]:
    """
    Hybrid OB bucketing: n_fine fine buckets near mid + n_coarse coarse buckets further out.
    Fine uses nsf_fine sig-figs, coarse uses nsf_coarse (one magnitude wider).

    BTC ~$80k: fine=$10 buckets (n_fine×$10=$100 range), coarse=$100 (n_coarse×$100=$1000 range)
    ETH ~$2k:  fine=$1,  coarse=$10
    SOL ~$100: fine=$0.1, coarse=$1
    """
    if side.empty:
        return [0.0] * (n_fine + n_coarse)
    best    = float(side.index[0])
    mag     = math.floor(math.log10(abs(best)))
    f_fine  = 10.0 ** (mag - nsf_fine   + 1)
    f_coarse= 10.0 ** (mag - nsf_coarse + 1)
    asc     = side.index[0] < side.index[-1]
    prices  = side.index.to_series().astype(float)
    amounts = side["amount"]

    lev1 = round(best / f_fine) * f_fine
    fine_vals = []
    for i in range(n_fine):
        c    = lev1 + i * f_fine if asc else lev1 - i * f_fine
        mask = (prices / f_fine).round() * f_fine == c
        fine_vals.append(float(amounts[mask].sum()))

    boundary  = lev1 + n_fine * f_fine if asc else lev1 - n_fine * f_fine
    lev1c     = round(boundary / f_coarse) * f_coarse
    cp        = prices[prices >= boundary] if asc else prices[prices < boundary]
    ca        = amounts[cp.index]
    coarse_vals = []
    for i in range(n_coarse):
        c    = lev1c + i * f_coarse if asc else lev1c - i * f_coarse
        mask = (cp / f_coarse).round() * f_coarse == c
        coarse_vals.append(float(ca[mask].sum()))

    return fine_vals + coarse_vals


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
