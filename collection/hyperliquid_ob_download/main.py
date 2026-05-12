"""
Hyperliquid DEX orderbook downloader v1.

OB stored as raw USD levels at 3 nSigFigs resolutions (20 per side each):
  nSigFigs=5 (~tick)  → bid/ask_lev01..20_usd        BTC: $1 buckets
  nSigFigs=4 ($10/$1) → ob4_bid/ask_lev01..20_usd    BTC: $10 buckets
  nSigFigs=3 ($100)   → ob3_bid/ask_lev01..20_usd    BTC: $100 buckets (macro zones)
Level prices omitted — step deterministic from nSigFigs + best bid/ask.

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, TOCKENONE
"""

import functions_framework
import os
import time
import mysql.connector

from hyperliquid_api import (
    Hyperliquid, parse_ob, parse_candle, coin_context,
    parse_predicted_fundings, deal_imbalance,
)

N_LEVELS = 20


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


# ── OB helpers ────────────────────────────────────────────────────────────────

def microprice(bids, asks) -> float:
    bb, ba = float(bids.index[0]), float(asks.index[0])
    bsz = float(bids["size"].iloc[0])
    asz = float(asks["size"].iloc[0])
    return (bb * asz + ba * bsz) / (bsz + asz) if (bsz + asz) > 0 else (bb + ba) / 2


def ob_depth_span(bids, asks) -> float:
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

        # ── orderbooks at 3 nSigFigs resolutions ─────────────────────────
        bids,  asks  = parse_ob(api.orderbook(coin, n_sig_figs=5))
        bids4, asks4 = parse_ob(api.orderbook(coin, n_sig_figs=4))
        bids3, asks3 = parse_ob(api.orderbook(coin, n_sig_figs=3))

        ask_px = float(asks.index[0])
        bid_px = float(bids.index[0])

        # ── meta + asset context ──────────────────────────────────────────
        meta_ctxs  = api.meta_and_contexts()
        ctx        = coin_context(meta_ctxs, coin)
        fund_rate  = float(ctx["funding"])
        mark_px    = float(ctx["markPx"])
        oracle_px  = float(ctx["oraclePx"])
        oi_usd     = float(ctx["openInterest"]) * mark_px
        day_volume = float(ctx["dayNtlVlm"])
        premium    = float(ctx.get("premium") or 0.0)
        impact_bid = float((ctx.get("impactPxs") or [0, 0])[0])
        impact_ask = float((ctx.get("impactPxs") or [0, 0])[1])

        # ── predicted fundings (cross-venue) ──────────────────────────────
        try:
            pred = parse_predicted_fundings(api.predicted_fundings(), coin)
        except Exception:
            pred = {"hl_predicted": 0.0, "binance_funding": 0.0, "bybit_funding": 0.0}

        # ── OI cap flag ───────────────────────────────────────────────────
        try:
            at_oi_cap = int(coin in (api.perps_at_oi_cap() or []))
        except Exception:
            at_oi_cap = 0

        # ── candle ────────────────────────────────────────────────────────
        candle = parse_candle(api.candles(coin, interval="1m", lookback_ms=600_000))

        # ── deal imbalance ────────────────────────────────────────────────
        trades = api.recent_trades(coin)
        d_sell, d_buy = deal_imbalance(trades, lookback_ms=60_000)

        # ── OB scalar metrics (from nSigFigs=5 base) ─────────────────────
        spread_bps = (ask_px - bid_px) / bid_px * 10_000
        mp         = microprice(bids, asks)
        depth_span = ob_depth_span(bids, asks)
        imbalance  = (bids["amount"].sum() - asks["amount"].sum()) / \
                     (bids["amount"].sum() + asks["amount"].sum() + 1e-12)
        bid_conc   = bids["amount"].head(10).sum() / (bids["amount"].sum() + 1e-12)
        ask_conc   = asks["amount"].head(10).sum() / (asks["amount"].sum() + 1e-12)

        # ── assemble row ──────────────────────────────────────────────────
        row = {"timestamp": t_s}

        # ── prices & market ───────────────────────────────────────────────
        row.update({
            "ask_price":     ask_px,
            "bid_price":     bid_px,
            "mark_price":    mark_px,
            "oracle_price":  oracle_px,
            "microprice":    mp,
            "spread_bps":    spread_bps,
            "ob_depth_span": depth_span,
            "oi_usd":        oi_usd,
            "fund_rate":     fund_rate,
            "premium":       premium,
            "impact_bid_px": impact_bid,
            "impact_ask_px": impact_ask,
            "day_volume_usd": day_volume,
            "predicted_funding_hl":      pred["hl_predicted"],
            "predicted_funding_binance": pred["binance_funding"],
            "predicted_funding_bybit":   pred["bybit_funding"],
            "at_oi_cap":     at_oi_cap,
        })

        # ── OHLCV ─────────────────────────────────────────────────────────
        row.update({
            "open":       candle["open"],
            "high":       candle["high"],
            "low":        candle["low"],
            "close":      candle["close"],
            "volume":     candle["volume"],
            "num_trades": candle["num_trades"],
        })

        # ── OB scalar metrics ─────────────────────────────────────────────
        row.update({
            "imbalance":         imbalance,
            "bid_concentration": bid_conc,
            "ask_concentration": ask_conc,
        })

        # ── taker flow ────────────────────────────────────────────────────
        row.update({
            "sell_usd":       d_sell,
            "buy_usd":        d_buy,
            "sell_buy_ratio": (d_sell / d_buy) if d_buy > 0 else 0.0,
        })

        # ── OB levels: 20 per side × 3 resolutions ───────────────────────
        for prefix, b, a in [("",    bids,  asks),
                              ("ob4_", bids4, asks4),
                              ("ob3_", bids3, asks3)]:
            for i in range(1, N_LEVELS + 1):
                tag = f"{i:02d}"
                row[f"{prefix}bid_lev{tag}_usd"] = float(b["amount"].iloc[i-1]) if i <= len(b) else 0.0
                row[f"{prefix}ask_lev{tag}_usd"] = float(a["amount"].iloc[i-1]) if i <= len(a) else 0.0

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
