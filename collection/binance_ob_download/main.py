"""
Binance orderbook downloader v1 — spot + futures enriched schema.

OB stored as nSigFigs=4 price buckets (20 per side per market = 80 cols):
  BTC ~$82k → $10 buckets   ETH ~$2k → $1 buckets   SOL ~$97 → $0.01 buckets
Bucket prices omitted — derivable from best bid/ask + step = 10^(floor(log10(price))-3).

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, TOCKENONE
"""

import functions_framework
import os
import time
import mysql.connector

from binance_api import (
    Binance, parse_ob, parse_kline, parse_price_kline,
    aggregate_buckets_hybrid, deal_imbalance,
)

N_LEVELS = 20  # OB bucket levels per side per market


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


# ── OB helper ─────────────────────────────────────────────────────────────────

def wall(side, top_n: int = 100) -> tuple[float, float]:
    top = side.head(top_n)
    if top.empty:
        return 0.0, 0.0
    idx = top["amount"].idxmax()
    return float(idx), float(top.loc[idx, "size"])


# ── main ──────────────────────────────────────────────────────────────────────

@functions_framework.http
def binance_ob_download(request):
    payload = request.get_json(silent=True) or {}
    tok1    = (payload.get("TOCKENONE") or os.getenv("TOCKENONE", "BTC")).upper()

    symbol = Binance.SYMBOLS.get(tok1)
    if symbol is None:
        return f"ERROR: unknown token {tok1}", 400

    TABLE = f"binance_{tok1.lower()}usdt_spotperp_v1"

    db_cfg = {
        "host":     os.getenv("DBHOST"),
        "user":     os.getenv("DBUSER"),
        "password": os.getenv("DBPASSW"),
        "database": os.getenv("DBSQL"),
        "port":     25060,
    }

    try:
        api = Binance()
        t_s = int(time.time())

        # ── orderbooks ────────────────────────────────────────────────────
        # spot max=5000 (BTC: 0.1%→1% coverage), perp max=1000
        spot_bids, spot_asks = parse_ob(api.spot_depth(symbol, limit=5000))
        perp_bids, perp_asks = parse_ob(api.perp_depth(symbol, limit=1000))

        spot_ask_px = float(spot_asks.index[0])
        spot_bid_px = float(spot_bids.index[0])
        perp_ask_px = float(perp_asks.index[0])
        perp_bid_px = float(perp_bids.index[0])

        # ── candles ───────────────────────────────────────────────────────
        spot_c  = parse_kline(api.spot_klines(symbol, interval="1m", limit=3))
        perp_c  = parse_kline(api.perp_klines(symbol, interval="1m", limit=3))
        mark_c  = parse_price_kline(api.mark_price_klines(symbol, interval="1m", limit=3))
        index_c = parse_price_kline(api.index_price_klines(symbol, interval="1m", limit=3))
        try:
            premium_index = parse_price_kline(
                api.premium_index_klines(symbol, interval="1m", limit=3))["close"]
        except Exception:
            premium_index = 0.0

        # ── derivatives ───────────────────────────────────────────────────
        prem      = api.premium_index(symbol)
        mark_px   = float(prem["markPrice"])
        index_px  = float(prem["indexPrice"])
        fund_rate = float(prem["lastFundingRate"])
        next_fund_ms = float(prem.get("nextFundingTime", 0) or 0)
        time_to_funding_mins = (next_fund_ms / 1000 - t_s) / 60 if next_fund_ms > 0 else 0.0
        oi_usd    = float(api.open_interest(symbol)["openInterest"]) * mark_px

        # ── taker volume (aggregated last 5m) ─────────────────────────────
        try:
            tk = api.taker_ls_ratio(symbol, period="5m", limit=1)[0]
            taker_buy  = float(tk["buyVol"])
            taker_sell = float(tk["sellVol"])
        except Exception:
            taker_buy = taker_sell = 0.0

        # ── long/short ratios ─────────────────────────────────────────────
        try:
            ls_all = float(api.ls_account_ratio(symbol, period="5m", limit=1)[0]["longShortRatio"])
        except Exception:
            ls_all = 1.0
        try:
            ls_top_acc = float(api.ls_top_account_ratio(symbol, period="5m", limit=1)[0]["longShortRatio"])
        except Exception:
            ls_top_acc = 1.0
        try:
            ls_top_pos = float(api.ls_top_position_ratio(symbol, period="5m", limit=1)[0]["longShortRatio"])
        except Exception:
            ls_top_pos = 1.0

        # ── deal imbalance (last 60s) ─────────────────────────────────────
        try:
            d_sell_spot, d_buy_spot = deal_imbalance(api.spot_trades(symbol, limit=1000))
        except Exception:
            d_sell_spot = d_buy_spot = 0.0
        try:
            d_sell_perp, d_buy_perp = deal_imbalance(api.perp_trades(symbol, limit=1000))
        except Exception:
            d_sell_perp = d_buy_perp = 0.0

        # ── OB scalar metrics ─────────────────────────────────────────────
        def _imb(b, a):
            tb, ta = b["amount"].sum(), a["amount"].sum()
            return (tb - ta) / (tb + ta + 1e-12)

        spot_imb = _imb(spot_bids, spot_asks)
        perp_imb = _imb(perp_bids, perp_asks)
        p_wbp, p_wbs = wall(perp_bids)
        p_wap, p_was = wall(perp_asks)

        # ── assemble row ──────────────────────────────────────────────────
        row = {"timestamp": t_s}

        # ── OB buckets: nSigFigs=4, 20 levels × 2 sides × 2 markets ──────
        for prefix, bids, asks in [("spot", spot_bids, spot_asks),
                                    ("perp", perp_bids, perp_asks)]:
            for side_name, buckets in [("bid", aggregate_buckets_hybrid(bids)),
                                        ("ask", aggregate_buckets_hybrid(asks))]:
                for i, usd in enumerate(buckets, 1):
                    row[f"{prefix}_{side_name}_lev{i:02d}_usd"] = usd

        # ── prices & derivatives ──────────────────────────────────────────
        row.update({
            "spot_ask_price":        spot_ask_px,
            "spot_bid_price":        spot_bid_px,
            "perp_ask_price":        perp_ask_px,
            "perp_bid_price":        perp_bid_px,
            "mark_price":            mark_px,
            "index_price":           index_px,
            "oi_usd":                oi_usd,
            "fund_rate":             fund_rate,
            "time_to_funding_mins":  time_to_funding_mins,
        })

        # ── OHLCV ─────────────────────────────────────────────────────────
        row.update({
            "spot_open":              spot_c["open"],
            "spot_high":              spot_c["high"],
            "spot_low":               spot_c["low"],
            "spot_close":             spot_c["close"],
            "spot_minute_volume":     spot_c["volume"],
            "spot_minute_volume_usd": spot_c["volume_usd"],
            "spot_taker_buy_vol":     spot_c["taker_buy_volume"],
            "spot_num_trades":        spot_c["num_trades"],
            "perp_open":              perp_c["open"],
            "perp_high":              perp_c["high"],
            "perp_low":               perp_c["low"],
            "perp_close":             perp_c["close"],
            "perp_minute_volume":     perp_c["volume"],
            "perp_minute_volume_usd": perp_c["volume_usd"],
            "perp_taker_buy_vol":     perp_c["taker_buy_volume"],
            "perp_num_trades":        perp_c["num_trades"],
            "mark_open":              mark_c["open"],
            "mark_high":              mark_c["high"],
            "mark_low":               mark_c["low"],
            "mark_close":             mark_c["close"],
            "index_open":             index_c["open"],
            "index_high":             index_c["high"],
            "index_low":              index_c["low"],
            "index_close":            index_c["close"],
            "premium_index":          premium_index,
        })

        # ── OB metrics ────────────────────────────────────────────────────
        row.update({
            "spot_imbalance":   spot_imb,
            "perp_imbalance":   perp_imb,
            "perp_wall_bid_price": p_wbp,
            "perp_wall_bid_size":  p_wbs,
            "perp_wall_ask_price": p_wap,
            "perp_wall_ask_size":  p_was,
        })

        # ── taker flow & sentiment ────────────────────────────────────────
        row.update({
            "taker_sell":               taker_sell,
            "taker_buy":                taker_buy,
            "spot_sell_buy_side_deals": (d_sell_spot / d_buy_spot) if d_buy_spot > 0 else 0.0,
            "perp_sell_buy_side_deals": (d_sell_perp / d_buy_perp) if d_buy_perp > 0 else 0.0,
            "ls_ratio_all":             ls_all,
            "ls_ratio_top_acc":         ls_top_acc,
            "ls_ratio_top_pos":         ls_top_pos,
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
