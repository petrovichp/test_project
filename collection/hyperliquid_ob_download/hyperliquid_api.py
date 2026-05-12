"""
Hyperliquid DEX REST API client — public market-data endpoints only.

All requests are POST to https://api.hyperliquid.xyz/info.
No authentication required for any method in this module.

Coin names (perp): "BTC", "ETH", "SOL"
"""

import time
import requests
import pandas as pd


class Hyperliquid:
    BASE = "https://api.hyperliquid.xyz/info"
    COINS = {"BTC", "ETH", "SOL"}

    def post(self, body: dict) -> dict | list:
        r = requests.post(self.BASE, json=body,
                          headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        return r.json()

    def meta_and_contexts(self) -> list:
        """[meta, asset_contexts] — funding, OI, mark/oracle price, premium for all coins."""
        return self.post({"type": "metaAndAssetCtxs"})

    def orderbook(self, coin: str, n_sig_figs: int = None,
                  mantissa: int = None) -> dict:
        """
        L2 snapshot. levels[0]=bids (desc), levels[1]=asks (asc).

        n_sig_figs: None=full precision, 2–5 aggregates prices to N sig figs.
          BTC ~$82k: nSigFigs=5→$1 buckets, 4→$10, 3→$100
        mantissa: 2 or 5, only valid when n_sig_figs=5.
          Note: mantissa=1 returns HTTP 500 — avoid.
        """
        body = {"type": "l2Book", "coin": coin}
        if n_sig_figs is not None:
            body["nSigFigs"] = n_sig_figs
        if mantissa is not None and n_sig_figs == 5:
            body["mantissa"] = mantissa
        return self.post(body)

    def candles(self, coin: str, interval: str = "1m",
                lookback_ms: int = 600_000) -> list:
        """OHLCV candles. interval: '1m','5m','1h', etc."""
        now_ms = int(time.time() * 1000)
        return self.post({"type": "candleSnapshot", "req": {
            "coin":      coin,
            "interval":  interval,
            "startTime": now_ms - lookback_ms,
            "endTime":   now_ms,
        }})

    def recent_trades(self, coin: str) -> list:
        """Recent trades. side: 'B'=buy-initiated, 'A'=sell-initiated."""
        return self.post({"type": "recentTrades", "coin": coin})

    def predicted_fundings(self) -> list:
        """Predicted next-period funding for all coins + cross-venue rates (Binance, Bybit)."""
        return self.post({"type": "predictedFundings"})

    def perps_at_oi_cap(self) -> list:
        """List of coin names currently at their open-interest cap."""
        return self.post({"type": "perpsAtOpenInterestCap"})


# ── OB parsing ────────────────────────────────────────────────────────────────

def parse_ob(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert l2Book response to (bids, asks) DataFrames.
    Levels are already price-aggregated. Columns: size, amount. Index: price.
    """
    def _side(levels: list) -> pd.DataFrame:
        if not levels:
            return pd.DataFrame(columns=["size", "amount"])
        df = pd.DataFrame(levels)
        df["price"]  = df["px"].astype(float)
        df["size"]   = df["sz"].astype(float)
        df["amount"] = df["price"] * df["size"]
        return df.set_index("price")[["size", "amount"]]

    bids = _side(raw["levels"][0]).sort_index(ascending=False)
    asks = _side(raw["levels"][1]).sort_index(ascending=True)
    return bids, asks


def parse_candle(candles: list) -> dict:
    """Last closed 1m bar — index [-2] avoids the currently-open bar."""
    if len(candles) < 2:
        raise ValueError(f"Need ≥2 candles, got {len(candles)}")
    c = candles[-2]
    return {
        "open":       float(c["o"]),
        "high":       float(c["h"]),
        "low":        float(c["l"]),
        "close":      float(c["c"]),
        "volume":     float(c["v"]),
        "num_trades": int(c["n"]),
    }


def coin_context(meta_ctxs: list, coin: str) -> dict:
    """Extract asset context dict for a named coin from metaAndAssetCtxs response."""
    meta, ctxs = meta_ctxs
    for i, asset in enumerate(meta["universe"]):
        if asset["name"] == coin:
            return ctxs[i]
    raise KeyError(f"Coin {coin} not found in universe")


def parse_predicted_fundings(raw: list, coin: str) -> dict:
    """
    Extract HL predicted + cross-venue rates for a given coin.
    Returns: {hl_predicted, binance_funding, bybit_funding} — hourly rates.
    Response shape: [[coin, [[venue, {fundingRate, ...}], ...]], ...]
    """
    result = {"hl_predicted": 0.0, "binance_funding": 0.0, "bybit_funding": 0.0}
    venue_map = {"HlPerp": "hl_predicted", "BinPerp": "binance_funding",
                 "BybitPerp": "bybit_funding"}
    for entry in raw:
        c      = entry[0] if isinstance(entry, list) else entry.get("coin", "")
        venues = entry[1] if isinstance(entry, list) else entry.get("exchanges", [])
        if c != coin:
            continue
        for ve in venues:
            vname = ve[0] if isinstance(ve, list) else ve.get("exchange", "")
            vdata = ve[1] if isinstance(ve, list) else ve
            key   = venue_map.get(vname)
            if key:
                result[key] = float(vdata.get("fundingRate", 0.0) or 0.0)
        break
    return result


def deal_imbalance(trades: list, lookback_ms: int = 60_000) -> tuple[float, float]:
    """
    Sell/buy USD volume in the last lookback_ms ms.
    side='B' → buy-initiated, side='A' → sell-initiated.
    """
    if not trades:
        return 0.0, 0.0
    df = pd.DataFrame(trades)
    df["time"] = df["time"].astype(int)
    df["px"]   = df["px"].astype(float)
    df["sz"]   = df["sz"].astype(float)
    df["usd"]  = df["px"] * df["sz"]
    cutoff = df["time"].max() - lookback_ms
    recent = df[df["time"] >= cutoff]
    sell_usd = float(recent[recent["side"] == "A"]["usd"].sum())
    buy_usd  = float(recent[recent["side"] == "B"]["usd"].sum())
    return sell_usd, buy_usd
