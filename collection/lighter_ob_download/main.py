"""
Lighter DEX orderbook downloader v1 — lean enriched schema.

Analogous to okx_ob_download_v3 but for Lighter perp-only DEX.

Differences vs OKX v3:
  - Single OB per market (no spot/perp split) → 32 depth columns vs 64
  - Funding rate is hourly (not continuous); stored as latest available
  - OI from market_details (not a separate endpoint)
  - No long/short ratio (not in Lighter public API)
  - No index spread (Lighter IS the index)
  - Liquidations derived from recentTrades type='liquidation'
  - Quote asset is USDC, not USDT

CUMULATIVE DEPTH (16 levels × 2 sides = 32 columns)
  Levels: 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.20, 1.50,
          2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00  (% from mid)
  Column: {bid|ask}_cum_{005|010|...}pct  → cumulative USD amount

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, TOCKENONE
"""

import functions_framework
import os
import time
import pandas as pd
import mysql.connector

from lighter_api import (
    Lighter, parse_ob, parse_candle, parse_funding,
    deal_imbalance, liquidation_stats,
)

DEPTH_LEVELS_PCT = [
    0.05, 0.10, 0.20, 0.30, 0.50, 0.80,
    1.00, 1.20, 1.50, 2.00, 2.50, 3.00,
    4.00, 5.00, 6.00, 7.00,
]


def _tag(pct: float) -> str:
    return f"{pct*100:05.1f}".replace(".", "").lstrip("0").zfill(3)


# ── DB helpers (identical to v3) ──────────────────────────────────────────────

def db_check_table_exists(conn, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"SHOW TABLES LIKE '{table}';")
    result = cur.fetchone()
    cur.close()
    return result is not None


def db_create_table_from_dict(conn, table: str, row: dict):
    cur = conn.cursor()
    cols = ", ".join([f"`{k}` FLOAT" for k in row if k != "timestamp"])
    cur.execute(f"CREATE TABLE IF NOT EXISTS `{table}` "
                f"(`timestamp` BIGINT PRIMARY KEY, {cols});")
    conn.commit()
    cur.close()


def db_insert(conn, table: str, row: dict):
    cur = conn.cursor()
    cols = ", ".join([f"`{k}`" for k in row])
    vals = ", ".join(["%s"] * len(row))
    cur.execute(f"INSERT INTO `{table}` ({cols}) VALUES ({vals})", list(row.values()))
    conn.commit()
    cur.close()


# ── OB derived helpers ────────────────────────────────────────────────────────

def cum_depth_at_pct(side: pd.DataFrame, mid: float, pct: float) -> float:
    if side.index[0] >= mid:   # asks
        mask = side.index <= mid * (1 + pct / 100)
    else:                       # bids
        mask = side.index >= mid * (1 - pct / 100)
    return float(side.loc[mask, "amount"].sum())


def microprice(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    bb, ba = float(bids.index[0]), float(asks.index[0])
    bsz = float(bids["size"].iloc[0])
    asz = float(asks["size"].iloc[0])
    return (bb * asz + ba * bsz) / (bsz + asz) if (bsz + asz) > 0 else (bb + ba) / 2


def wall(side: pd.DataFrame, top_n: int = 100) -> tuple[float, float]:
    top = side.head(top_n)
    if top.empty:
        return 0.0, 0.0
    idx = top["amount"].idxmax()
    return float(idx), float(top.loc[idx, "size"])


def ob_depth_span(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    return float(min(bids.index[0] - bids.index[-1],
                     asks.index[-1] - asks.index[0]))


# ── main ──────────────────────────────────────────────────────────────────────

@functions_framework.http
def lighter_ob_download(request):
    payload = request.get_json(silent=True) or {}
    TOK1    = payload.get("TOCKENONE") or os.getenv("TOCKENONE", "BTC")

    market_id = Lighter.MARKETS.get(TOK1.upper())
    if market_id is None:
        return f"ERROR: unknown token {TOK1}", 400

    TABLE = f"lighter_{TOK1.lower()}usdc_perp_v1"

    db_cfg = {
        "host":     os.getenv("DBHOST"),
        "user":     os.getenv("DBUSER"),
        "password": os.getenv("DBPASSW"),
        "database": os.getenv("DBSQL"),
        "port":     25060,
    }

    try:
        api = Lighter()
        t_s = int(time.time())

        # ── orderbook ─────────────────────────────────────────────────────
        bids, asks = parse_ob(api.orderbook(market_id, limit=250))

        ask_px  = float(asks.index[0])
        bid_px  = float(bids.index[0])
        mid     = (ask_px + bid_px) / 2

        # ── candle ────────────────────────────────────────────────────────
        candle = parse_candle(api.candles(market_id, resolution="1m", count_back=3))

        # ── funding ───────────────────────────────────────────────────────
        fund_rate = parse_funding(api.funding(market_id, resolution="1h", count_back=3))

        # ── market details (OI) ───────────────────────────────────────────
        details = api.market_details(market_id)
        oi_usd = float(details["open_interest"]) * ask_px

        # ── deal imbalance ────────────────────────────────────────────────
        trades_raw = api.recent_trades(market_id, limit=100)
        d_sell, d_buy = deal_imbalance(trades_raw, lookback_ms=60_000)

        # ── liquidations ──────────────────────────────────────────────────
        liq = liquidation_stats(trades_raw)

        # ── OB metrics ────────────────────────────────────────────────────
        def _imb(b, a):
            tb, ta = b["amount"].sum(), a["amount"].sum()
            return (tb - ta) / (tb + ta + 1e-12)

        spread_bps   = (ask_px - bid_px) / bid_px * 10_000
        imbalance    = _imb(bids, asks)
        bid_conc     = bids["amount"].head(10).sum() / (bids["amount"].sum() + 1e-12)
        ask_conc     = asks["amount"].head(10).sum() / (asks["amount"].sum() + 1e-12)
        lrg_thresh   = (bids["amount"].sum() + asks["amount"].sum()) * 0.01
        lrg_bid      = int((bids["amount"] > lrg_thresh).sum())
        lrg_ask      = int((asks["amount"] > lrg_thresh).sum())
        mp           = microprice(bids, asks)
        depth_span   = ob_depth_span(bids, asks)
        wall_bid_px, wall_bid_sz = wall(bids)
        wall_ask_px, wall_ask_sz = wall(asks)

        # ── assemble row ──────────────────────────────────────────────────
        row = {"timestamp": t_s}

        # 16 cumulative depth levels × 2 sides = 32 columns
        for pct in DEPTH_LEVELS_PCT:
            tag = _tag(pct)
            row[f"bid_cum_{tag}pct"] = cum_depth_at_pct(bids, mid, pct)
            row[f"ask_cum_{tag}pct"] = cum_depth_at_pct(asks, mid, pct)

        row.update({
            "ask_price":        ask_px,
            "bid_price":        bid_px,
            "spread_bps":       spread_bps,
            "ob_depth_span":    depth_span,
            "oi_usd":           oi_usd,
            "fund_rate":        fund_rate,
        })

        row.update({
            "open":         candle["open"],
            "high":         candle["high"],
            "low":          candle["low"],
            "close":        candle["close"],
            "minute_volume":  candle["volume"],
            "volume_usd":   candle["volume_usd"],
        })

        row.update({
            "imbalance":          imbalance,
            "bid_concentration":  bid_conc,
            "ask_concentration":  ask_conc,
            "large_bid_count":    lrg_bid,
            "large_ask_count":    lrg_ask,
            "microprice":         mp,
            "wall_bid_price":     wall_bid_px,
            "wall_bid_size":      wall_bid_sz,
            "wall_ask_price":     wall_ask_px,
            "wall_ask_size":      wall_ask_sz,
        })

        row.update({
            "sell_usd":       d_sell,
            "buy_usd":        d_buy,
            "sell_buy_ratio": d_sell / (d_buy + 1e-12),
        })

        row.update(liq)

        # ── persist ───────────────────────────────────────────────────────
        conn = mysql.connector.connect(**db_cfg)
        if not db_check_table_exists(conn, TABLE):
            db_create_table_from_dict(conn, TABLE, row)
        db_insert(conn, TABLE, row)
        conn.close()

    except Exception as e:
        print(f"ERROR: {e}")
        return f"ERROR: {e}", 500

    return f"OK {TABLE} ts={t_s} cols={len(row)}"
