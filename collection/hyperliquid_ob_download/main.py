"""
Hyperliquid DEX orderbook downloader v1 — lean enriched schema.

Analogous to lighter_ob_download but for Hyperliquid perp DEX.

Differences vs Lighter:
  - Single POST endpoint (no GET) — all requests to /info
  - metaAndAssetCtxs gives funding + OI + mark/oracle price in one call
  - OB levels already price-aggregated (no per-order aggregation needed)
  - Candles include trade count (n)
  - Quote asset is USDC (native stablecoin on Hyperliquid)

CUMULATIVE DEPTH (16 levels × 2 sides = 32 columns)
  Column: {bid|ask}_cum_{005|010|...}pct  → cumulative USD amount

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, TOCKENONE
"""

import functions_framework
import os
import time
import pandas as pd
import mysql.connector

from hyperliquid_api import (
    Hyperliquid, parse_ob, parse_candle, coin_context, deal_imbalance,
)

DEPTH_LEVELS_PCT = [
    0.05, 0.10, 0.20, 0.30, 0.50, 0.80,
    1.00, 1.20, 1.50, 2.00, 2.50, 3.00,
    4.00, 5.00, 6.00, 7.00,
]


def _tag(pct: float) -> str:
    return f"{pct*100:05.1f}".replace(".", "").lstrip("0").zfill(3)


# ── DB helpers ────────────────────────────────────────────────────────────────

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
    if side.index[0] >= mid:
        mask = side.index <= mid * (1 + pct / 100)
    else:
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
def hyperliquid_ob_download(request):
    payload = request.get_json(silent=True) or {}
    coin    = (payload.get("TOCKENONE") or os.getenv("TOCKENONE", "BTC")).upper()

    if coin not in Hyperliquid.COINS:
        return f"ERROR: unknown coin {coin}", 400

    TABLE = f"hyperliquid_{coin.lower()}usdc_perp_v1"

    db_cfg = {
        "host":     os.getenv("DBHOST"),
        "user":     os.getenv("DBUSER"),
        "password": os.getenv("DBPASSW"),
        "database": os.getenv("DBSQL"),
        "port":     25060,
    }

    try:
        api = Hyperliquid()
        t_s = int(time.time())

        # ── orderbook ─────────────────────────────────────────────────────
        bids, asks = parse_ob(api.orderbook(coin))
        ask_px = float(asks.index[0])
        bid_px = float(bids.index[0])
        mid    = (ask_px + bid_px) / 2

        # ── meta + asset context (funding, OI, mark, oracle) ──────────────
        ctx = coin_context(api.meta_and_contexts(), coin)
        fund_rate  = float(ctx["funding"])
        mark_px    = float(ctx["markPx"])
        oracle_px  = float(ctx["oraclePx"])
        oi_usd     = float(ctx["openInterest"]) * mark_px
        day_volume = float(ctx["dayNtlVlm"])

        # ── candle ────────────────────────────────────────────────────────
        candle = parse_candle(api.candles(coin, interval="1m", lookback_ms=600_000))

        # ── deal imbalance ────────────────────────────────────────────────
        trades = api.recent_trades(coin)
        d_sell, d_buy = deal_imbalance(trades, lookback_ms=60_000)

        # ── OB metrics ────────────────────────────────────────────────────
        def _imb(b, a):
            tb, ta = b["amount"].sum(), a["amount"].sum()
            return (tb - ta) / (tb + ta + 1e-12)

        spread_bps = (ask_px - bid_px) / bid_px * 10_000
        imbalance  = _imb(bids, asks)
        bid_conc   = bids["amount"].head(10).sum() / (bids["amount"].sum() + 1e-12)
        ask_conc   = asks["amount"].head(10).sum() / (asks["amount"].sum() + 1e-12)
        lrg        = (bids["amount"].sum() + asks["amount"].sum()) * 0.01
        lrg_bid    = int((bids["amount"] > lrg).sum())
        lrg_ask    = int((asks["amount"] > lrg).sum())
        mp         = microprice(bids, asks)
        depth_span = ob_depth_span(bids, asks)
        wbp, wbs   = wall(bids)
        wap, was_  = wall(asks)

        # ── assemble row ──────────────────────────────────────────────────
        row = {"timestamp": t_s}

        for pct in DEPTH_LEVELS_PCT:
            tag = _tag(pct)
            row[f"bid_cum_{tag}pct"] = cum_depth_at_pct(bids, mid, pct)
            row[f"ask_cum_{tag}pct"] = cum_depth_at_pct(asks, mid, pct)

        row.update({
            "ask_price":     ask_px,
            "bid_price":     bid_px,
            "mark_price":    mark_px,
            "oracle_price":  oracle_px,
            "spread_bps":    spread_bps,
            "ob_depth_span": depth_span,
            "oi_usd":        oi_usd,
            "fund_rate":     fund_rate,
            "day_volume_usd": day_volume,
        })

        row.update({
            "open":        candle["open"],
            "high":        candle["high"],
            "low":         candle["low"],
            "close":       candle["close"],
            "volume":      candle["volume"],
            "num_trades":  candle["num_trades"],
        })

        row.update({
            "imbalance":         imbalance,
            "bid_concentration": bid_conc,
            "ask_concentration": ask_conc,
            "large_bid_count":   lrg_bid,
            "large_ask_count":   lrg_ask,
            "microprice":        mp,
            "wall_bid_price":    wbp,
            "wall_bid_size":     wbs,
            "wall_ask_price":    wap,
            "wall_ask_size":     was_,
        })

        row.update({
            "sell_usd":       d_sell,
            "buy_usd":        d_buy,
            "sell_buy_ratio": d_sell / (d_buy + 1e-12),
        })

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
