"""
Binance orderbook downloader v1 — spot + futures enriched schema.

Analogous to okx_ob_download_v3 but for Binance.
Schema is closely aligned with OKX v3 for cross-exchange feature parity.

Differences vs OKX v3:
  - Kline response includes taker buy/sell volume directly (no separate endpoint)
  - Long/short ratios available (global + top traders) — mirrors OKX
  - Taker volume from /futures/data/takerlongshortRatio (aggregated 5m)
  - Mark price + index price from premiumIndex
  - No liquidation orders endpoint — skipped
  - Quote asset is USDT

CUMULATIVE DEPTH (16 levels × 4 sides = 64 columns)
  Column: {spot|perp}_{bid|ask}_cum_{005|010|...}pct

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, TOCKENONE
"""

import functions_framework
import os
import time
import pandas as pd
import mysql.connector

from binance_api import Binance, parse_ob, parse_kline, deal_imbalance

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
        spot_bids, spot_asks = parse_ob(api.spot_depth(symbol, limit=500))
        perp_bids, perp_asks = parse_ob(api.perp_depth(symbol, limit=500))

        spot_ask_px = float(spot_asks.index[0])
        spot_bid_px = float(spot_bids.index[0])
        perp_ask_px = float(perp_asks.index[0])
        perp_bid_px = float(perp_bids.index[0])
        spot_mid    = (spot_ask_px + spot_bid_px) / 2
        perp_mid    = (perp_ask_px + perp_bid_px) / 2

        # ── candles ───────────────────────────────────────────────────────
        spot_c = parse_kline(api.spot_klines(symbol, interval="1m", limit=3))
        perp_c = parse_kline(api.perp_klines(symbol, interval="1m", limit=3))

        # ── derivatives ───────────────────────────────────────────────────
        prem      = api.premium_index(symbol)
        mark_px   = float(prem["markPrice"])
        index_px  = float(prem["indexPrice"])
        fund_rate = float(prem["lastFundingRate"])

        oi_raw = api.open_interest(symbol)
        oi_usd = float(oi_raw["openInterest"]) * mark_px

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

        # ── deal imbalance (last 60s from recent trades) ──────────────────
        try:
            d_sell_spot, d_buy_spot = deal_imbalance(api.spot_trades(symbol, limit=500))
        except Exception:
            d_sell_spot = d_buy_spot = 0.0
        try:
            d_sell_perp, d_buy_perp = deal_imbalance(api.perp_trades(symbol, limit=500))
        except Exception:
            d_sell_perp = d_buy_perp = 0.0

        # ── OB metrics ────────────────────────────────────────────────────
        def _imb(b, a):
            tb, ta = b["amount"].sum(), a["amount"].sum()
            return (tb - ta) / (tb + ta + 1e-12)

        spot_spread_bps = (spot_ask_px - spot_bid_px) / spot_bid_px * 10_000
        perp_spread_bps = (perp_ask_px - perp_bid_px) / perp_bid_px * 10_000
        spot_imb  = _imb(spot_bids, spot_asks)
        perp_imb  = _imb(perp_bids, perp_asks)
        spot_bid_conc = spot_bids["amount"].head(10).sum() / (spot_bids["amount"].sum() + 1e-12)
        spot_ask_conc = spot_asks["amount"].head(10).sum() / (spot_asks["amount"].sum() + 1e-12)
        perp_bid_conc = perp_bids["amount"].head(10).sum() / (perp_bids["amount"].sum() + 1e-12)
        perp_ask_conc = perp_asks["amount"].head(10).sum() / (perp_asks["amount"].sum() + 1e-12)
        lrg_s = (spot_bids["amount"].sum() + spot_asks["amount"].sum()) * 0.01
        lrg_p = (perp_bids["amount"].sum() + perp_asks["amount"].sum()) * 0.01
        spot_lrg_bid = int((spot_bids["amount"] > lrg_s).sum())
        spot_lrg_ask = int((spot_asks["amount"] > lrg_s).sum())
        perp_lrg_bid = int((perp_bids["amount"] > lrg_p).sum())
        perp_lrg_ask = int((perp_asks["amount"] > lrg_p).sum())

        # ── assemble row ──────────────────────────────────────────────────
        row = {"timestamp": t_s}

        # 16 levels × 4 sides = 64 columns
        for pct in DEPTH_LEVELS_PCT:
            tag = _tag(pct)
            row[f"spot_bid_cum_{tag}pct"] = cum_depth_at_pct(spot_bids, spot_mid, pct)
            row[f"spot_ask_cum_{tag}pct"] = cum_depth_at_pct(spot_asks, spot_mid, pct)
            row[f"perp_bid_cum_{tag}pct"] = cum_depth_at_pct(perp_bids, perp_mid, pct)
            row[f"perp_ask_cum_{tag}pct"] = cum_depth_at_pct(perp_asks, perp_mid, pct)

        row.update({
            "spot_ask_price":    spot_ask_px,
            "spot_bid_price":    spot_bid_px,
            "perp_ask_price":    perp_ask_px,
            "perp_bid_price":    perp_bid_px,
            "mark_price":        mark_px,
            "index_price":       index_px,
            "diff_price":        perp_ask_px - spot_ask_px,
            "span_spot_price":   (spot_ask_px - spot_bid_px) / spot_bid_px,
            "span_perp_price":   (perp_ask_px - perp_bid_px) / perp_bid_px,
            "ob_depth_span_spot": ob_depth_span(spot_bids, spot_asks),
            "ob_depth_span_perp": ob_depth_span(perp_bids, perp_asks),
            "oi_usd":            oi_usd,
            "fund_rate":         fund_rate,
        })

        row.update({
            "spot_open":   spot_c["open"],   "spot_high":  spot_c["high"],
            "spot_low":    spot_c["low"],    "spot_close": spot_c["close"],
            "spot_minute_volume": spot_c["volume"],
            "perp_open":   perp_c["open"],   "perp_high":  perp_c["high"],
            "perp_low":    perp_c["low"],    "perp_close": perp_c["close"],
            "perp_minute_volume": perp_c["volume"],
        })

        row.update({
            "spot_spread_bps":        spot_spread_bps,
            "spot_imbalance":         spot_imb,
            "spot_bid_concentration": spot_bid_conc,
            "spot_ask_concentration": spot_ask_conc,
            "spot_large_bid_count":   spot_lrg_bid,
            "spot_large_ask_count":   spot_lrg_ask,
            "microprice_spot":        microprice(spot_bids, spot_asks),
            "spot_wall_bid_price":    wall(spot_bids)[0],
            "spot_wall_bid_size":     wall(spot_bids)[1],
            "spot_wall_ask_price":    wall(spot_asks)[0],
            "spot_wall_ask_size":     wall(spot_asks)[1],
            "perp_spread_bps":        perp_spread_bps,
            "perp_imbalance":         perp_imb,
            "perp_bid_concentration": perp_bid_conc,
            "perp_ask_concentration": perp_ask_conc,
            "perp_large_bid_count":   perp_lrg_bid,
            "perp_large_ask_count":   perp_lrg_ask,
            "microprice_perp":        microprice(perp_bids, perp_asks),
            "perp_wall_bid_price":    wall(perp_bids)[0],
            "perp_wall_bid_size":     wall(perp_bids)[1],
            "perp_wall_ask_price":    wall(perp_asks)[0],
            "perp_wall_ask_size":     wall(perp_asks)[1],
        })

        row.update({
            "taker_sell":              taker_sell,
            "taker_buy":               taker_buy,
            "taker_sell_buy_ratio":    taker_sell / (taker_buy + 1e-12),
            "spot_sell_buy_side_deals": d_sell_spot / (d_buy_spot + 1e-12),
            "perp_sell_buy_side_deals": d_sell_perp / (d_buy_perp + 1e-12),
        })

        row.update({
            "ls_ratio_all":     ls_all,
            "ls_ratio_top_acc": ls_top_acc,
            "ls_ratio_top_pos": ls_top_pos,
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
