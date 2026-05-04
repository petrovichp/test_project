"""
OKX orderbook downloader v3 — lean enriched schema.

BREAKING CHANGE vs v2:
  800 raw OB bins REMOVED — replaced with 16 cumulative depth levels.
  Rationale: bins were aggregated into buckets anyway; DeepLOB failed;
  cumulative depth is directly tradeable information at 14× smaller size.

CUMULATIVE DEPTH (16 levels × 4 sides = 64 columns)
  Levels: 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00, 1.20, 1.50,
          2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00  (% from mid)
  Key levels: 0.80% = direction label threshold, 1.0-3.0% = TP zone
  Column: {spot|perp}_{bid|ask}_cum_{005|010|...}pct  → cumulative $ amount
  Requires ob_depth_span to map bins to price — stored as separate column.

NEW SIGNALS (vs v2)
  ob_depth_span_spot/perp     — dollar range of OB depth (was computed but discarded in v2)
  microprice_spot/perp        — weighted mid price
  OHLC spot+perp              — open/high/low/close from 1m candle (v2 had volume only)
  wall_bid/ask price+size     — largest passive order in top 100 levels × spot+perp
  liquidation rolling         — long/short liq $ in 5m and 15m windows
  long_short_ratio            — all accounts + top trader accounts + top trader positions
  mark_price OHLC             — perp mark price candle
  index_spread_bps            — OKX spot vs multi-exchange index

STORAGE
  v2: 861 columns × 8 bytes = ~6.9 KB/row = ~9.9 MB/day
  v3: ~133 columns × 4 bytes = ~0.5 KB/row = ~0.7 MB/day  (14× smaller)

Env vars: DBHOST, DBUSER, DBPASSW, DBSQL, EXC, ORDER_BOOK_TYPE, TOCKENONE, TOCKENTWO
"""

import functions_framework
import os
import time
import pandas as pd
from datetime import datetime
import mysql.connector
import requests
import warnings
import json

warnings.filterwarnings("ignore")

# ── depth levels (% from mid) ─────────────────────────────────────────────────
DEPTH_LEVELS_PCT = [
    0.05,   # microstructure / spread zone
    0.10,   # tight scalp SL
    0.20,   # short scalp SL
    0.30,   # typical SL level
    0.50,   # halfway to label threshold
    0.80,   # ◄ direction label threshold (>0.8% = signal)
    1.00,   # ◄ typical TP
    1.20,   # mid TP
    1.50,   # ◄ extended TP
    2.00,   # strong move territory
    2.50,   # significant resistance / support
    3.00,   # strong trend signal
    4.00,   # extreme move
    5.00,   # outer bound — original range
    6.00,   # extended range
    7.00,   # ◄ max level (large swing / squeeze territory)
]

# column tag for each level  e.g. 0.05 → "005", 1.20 → "120", 7.00 → "700"
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


# ── OKX API client ────────────────────────────────────────────────────────────

class OKX:
    BASE = "https://www.okx.com"

    def get(self, path: str, params: dict = None) -> dict:
        r = requests.get(f"{self.BASE}{path}", params=params,
                         headers={"Content-Type": "application/json"}, timeout=10)
        r.raise_for_status()
        return r.json()

    def orderbook(self, inst_id: str, depth: int = 5000) -> dict:
        return self.get("/api/v5/market/books-full", {"instId": inst_id, "sz": depth})

    def candles(self, inst_id: str, bar: str = "1m", limit: int = 2) -> dict:
        return self.get("/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": limit})

    def mark_candles(self, inst_id: str, bar: str = "1m", limit: int = 2) -> dict:
        return self.get("/api/v5/market/mark-price-candles",
                        {"instId": inst_id, "bar": bar, "limit": limit})

    def index_candles(self, inst_id: str, bar: str = "1m", limit: int = 2) -> dict:
        return self.get("/api/v5/market/index-candles",
                        {"instId": inst_id, "bar": bar, "limit": limit})

    def trades(self, inst_id: str, limit: int = 500) -> dict:
        return self.get("/api/v5/market/trades", {"instId": inst_id, "limit": limit})

    def open_interest(self, inst_id: str) -> dict:
        return self.get("/api/v5/public/open-interest", {"instId": inst_id})

    def funding_rate(self, inst_id: str) -> dict:
        return self.get("/api/v5/public/funding-rate", {"instId": inst_id})

    def taker_volume(self, inst_id: str, period: str = "5m") -> dict:
        return self.get("/api/v5/rubik/stat/taker-volume-contract",
                        {"instId": inst_id, "period": period, "unit": "0", "limit": 2})

    def liquidations(self, inst_id: str, limit: int = 100) -> dict:
        return self.get("/api/v5/public/liquidation-orders",
                        {"instType": "SWAP", "instId": inst_id,
                         "state": "filled", "limit": limit})

    def ls_ratio_all(self, inst_id: str, period: str = "5m") -> dict:
        return self.get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
                        {"instId": inst_id, "period": period, "limit": 2})

    def ls_ratio_top_account(self, inst_id: str, period: str = "5m") -> dict:
        return self.get("/api/v5/rubik/stat/contracts/long-short-account-ratio-contract-top-trader",
                        {"instId": inst_id, "period": period, "limit": 2})

    def ls_ratio_top_position(self, inst_id: str, period: str = "5m") -> dict:
        return self.get("/api/v5/rubik/stat/contracts/long-short-position-ratio-contract-top-trader",
                        {"instId": inst_id, "period": period, "limit": 2})

    def index_ticker(self, inst_id: str) -> dict:
        return self.get("/api/v5/market/index-tickers", {"instId": inst_id})


# ── OB parsing ────────────────────────────────────────────────────────────────

def parse_ob(raw: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    bids = pd.DataFrame(raw["data"][0]["bids"], columns=["price", "size", "n"])
    asks = pd.DataFrame(raw["data"][0]["asks"], columns=["price", "size", "n"])
    for df in (bids, asks):
        df["price"]  = df["price"].astype(float)
        df["size"]   = df["size"].astype(float)
        df["amount"] = df["price"] * df["size"]
    bids = bids.set_index("price").sort_index(ascending=False)
    asks = asks.set_index("price").sort_index(ascending=True)
    return bids, asks


def ob_depth_span(bids: pd.DataFrame, asks: pd.DataFrame) -> float:
    """Dollar range that the OB covers on each side (minimum of bid/ask span)."""
    return float(min(bids.index[0] - bids.index[-1],
                     asks.index[-1] - asks.index[0]))


def cum_depth_at_pct(side: pd.DataFrame, mid: float, pct: float) -> float:
    """Cumulative $ amount within pct% of mid on one side."""
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
    """Price and BTC size of the largest single order in top N levels."""
    top = side.head(top_n)
    if top.empty:
        return 0.0, 0.0
    idx = top["amount"].idxmax()
    return float(idx), float(top.loc[idx, "size"])


# ── candle parsing ────────────────────────────────────────────────────────────

def parse_candle(raw: dict) -> dict:
    cols = ["timestamp", "open", "high", "low", "close",
            "volume", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(raw["data"], columns=cols)
    df["confirm"] = df["confirm"].astype(float)
    row = df[df["confirm"] == 1].iloc[0] if (df["confirm"] == 1).any() else df.iloc[0]
    return {k: float(row[k]) for k in ["open", "high", "low", "close", "volCcyQuote"]}


def parse_mark_candle(raw: dict) -> dict:
    cols = ["timestamp", "open", "high", "low", "close", "confirm"]
    df = pd.DataFrame(raw["data"], columns=cols)
    df["confirm"] = df["confirm"].astype(float)
    row = df[df["confirm"] == 1].iloc[0] if (df["confirm"] == 1).any() else df.iloc[0]
    return {k: float(row[k]) for k in ["open", "high", "low", "close"]}


# ── liquidation rolling aggregation ──────────────────────────────────────────

def liquidation_stats(raw: dict, windows_ms: list[int]) -> dict:
    """Rolling liquidation $ by side over multiple time windows."""
    rows = raw.get("data", [])
    if not rows:
        return {f"liq_long_{w//60000}m": 0.0 for w in windows_ms} | \
               {f"liq_short_{w//60000}m": 0.0 for w in windows_ms}

    df = pd.DataFrame(rows)
    df["ts"]    = df["ts"].astype(int)
    df["sz"]    = df["sz"].astype(float)
    df["bkLoss"] = df["bkLoss"].astype(float)    # $ value of liquidation
    now_ms = int(time.time() * 1000)

    result = {}
    for w in windows_ms:
        cut = now_ms - w
        recent = df[df["ts"] >= cut]
        result[f"liq_long_{w//60000}m"]  = float(recent[recent["side"] == "sell"]["bkLoss"].sum())
        result[f"liq_short_{w//60000}m"] = float(recent[recent["side"] == "buy"]["bkLoss"].sum())
    return result


# ── long/short ratio ──────────────────────────────────────────────────────────

def latest_ls_ratio(raw: dict) -> float:
    rows = raw.get("data", [])
    return float(rows[0][1]) if rows else 1.0


# ── deal imbalance from recent trades ────────────────────────────────────────

def deal_imbalance(okx: OKX, inst_id: str) -> tuple[float, float]:
    df = pd.DataFrame(okx.trades(inst_id, limit=500)["data"])
    df["ts"] = df["ts"].astype(int)
    df["px"] = df["px"].astype(float)
    df["sz"] = df["sz"].astype(float)
    df["amount"] = df["px"] * df["sz"]
    cutoff = df["ts"].iloc[0] - 60_000
    recent = df[df["ts"] > cutoff]
    return (float(recent[recent["side"] == "sell"]["amount"].sum()),
            float(recent[recent["side"] == "buy"]["amount"].sum()))


# ── main ──────────────────────────────────────────────────────────────────────

@functions_framework.http
def okx_ob_download_v3(request):
    payload   = request.get_json(silent=True) or {}
    TOK1      = payload.get("TOCKENONE") or os.getenv("TOCKENONE")
    TOK2      = os.getenv("TOCKENTWO")
    EXC       = os.getenv("EXC", "okx")
    OB_TYPE   = os.getenv("ORDER_BOOK_TYPE", "spotpepr")

    SPOT = f"{TOK1}-{TOK2}"
    PERP = f"{TOK1}-{TOK2}-SWAP"
    IDX  = f"{TOK1}-{TOK2}"          # index ticker instId
    TABLE = f"{EXC.lower()}_{TOK1.lower()}{TOK2.lower()}_{OB_TYPE.lower()}_v3"

    db_cfg = {
        "host": os.getenv("DBHOST"), "user": os.getenv("DBUSER"),
        "password": os.getenv("DBPASSW"), "database": os.getenv("DBSQL"),
        "port": 25060,
    }

    try:
        okx = OKX()
        t_s = int(time.time())

        # ── orderbooks ────────────────────────────────────────────────────
        spot_bids, spot_asks = parse_ob(okx.orderbook(SPOT))
        perp_bids, perp_asks = parse_ob(okx.orderbook(PERP))

        spot_ask_px = float(spot_asks.index[0])
        spot_bid_px = float(spot_bids.index[0])
        perp_ask_px = float(perp_asks.index[0])
        perp_bid_px = float(perp_bids.index[0])
        spot_mid    = (spot_ask_px + spot_bid_px) / 2
        perp_mid    = (perp_ask_px + perp_bid_px) / 2

        # ── candles ───────────────────────────────────────────────────────
        spot_c = parse_candle(okx.candles(SPOT))
        perp_c = parse_candle(okx.candles(PERP))
        mark_c = parse_mark_candle(okx.mark_candles(PERP))

        # ── derivatives ───────────────────────────────────────────────────
        oi_usd    = float(okx.open_interest(PERP)["data"][0]["oiUsd"])
        fund_rate = float(okx.funding_rate(PERP)["data"][0]["fundingRate"])

        # ── taker volume ──────────────────────────────────────────────────
        tv        = okx.taker_volume(PERP)["data"]
        taker_sell = float(tv[0][1]) if tv else 0.0
        taker_buy  = float(tv[0][2]) if tv else 0.0

        # ── deal imbalance from trades ────────────────────────────────────
        d_sell_spot, d_buy_spot = deal_imbalance(okx, SPOT)
        d_sell_perp, d_buy_perp = deal_imbalance(okx, PERP)

        # ── liquidations ──────────────────────────────────────────────────
        liq_data = liquidation_stats(okx.liquidations(PERP, limit=100),
                                     windows_ms=[5*60_000, 15*60_000])

        # ── long/short ratios ─────────────────────────────────────────────
        ls_all      = latest_ls_ratio(okx.ls_ratio_all(PERP))
        ls_top_acc  = latest_ls_ratio(okx.ls_ratio_top_account(PERP))
        ls_top_pos  = latest_ls_ratio(okx.ls_ratio_top_position(PERP))

        # ── index spread ──────────────────────────────────────────────────
        idx_px = float(okx.index_ticker(IDX)["data"][0]["idxPx"])
        index_spread_bps = (spot_mid - idx_px) / idx_px * 10_000

        # ── OB derived features ───────────────────────────────────────────
        s_depth_span = ob_depth_span(spot_bids, spot_asks)
        p_depth_span = ob_depth_span(perp_bids, perp_asks)
        mp_spot = microprice(spot_bids, spot_asks)
        mp_perp = microprice(perp_bids, perp_asks)
        s_wall_bid_px, s_wall_bid_sz = wall(spot_bids)
        s_wall_ask_px, s_wall_ask_sz = wall(spot_asks)
        p_wall_bid_px, p_wall_bid_sz = wall(perp_bids)
        p_wall_ask_px, p_wall_ask_sz = wall(perp_asks)

        # ── OB metrics ────────────────────────────────────────────────────
        def _imb(b, a):
            tb, ta = b["amount"].sum(), a["amount"].sum()
            return (tb - ta) / (tb + ta + 1e-12)

        spot_spread_bps = (spot_ask_px - spot_bid_px) / spot_bid_px * 10_000
        perp_spread_bps = (perp_ask_px - perp_bid_px) / perp_bid_px * 10_000
        spot_imb = _imb(spot_bids, spot_asks)
        perp_imb = _imb(perp_bids, perp_asks)
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

        # 16 cumulative depth levels × 4 sides = 64 columns
        for pct in DEPTH_LEVELS_PCT:
            tag = _tag(pct)
            row[f"spot_bid_cum_{tag}pct"] = cum_depth_at_pct(spot_bids, spot_mid, pct)
            row[f"spot_ask_cum_{tag}pct"] = cum_depth_at_pct(spot_asks, spot_mid, pct)
            row[f"perp_bid_cum_{tag}pct"] = cum_depth_at_pct(perp_bids, perp_mid, pct)
            row[f"perp_ask_cum_{tag}pct"] = cum_depth_at_pct(perp_asks, perp_mid, pct)

        # metadata — prices, OI, funding
        row.update({
            "spot_ask_price":           spot_ask_px,
            "spot_bid_price":           spot_bid_px,
            "perp_ask_price":           perp_ask_px,
            "perp_bid_price":           perp_bid_px,
            "diff_price":               perp_ask_px - spot_ask_px,
            "span_spot_price":          (spot_ask_px - spot_bid_px) / spot_bid_px,
            "span_perp_price":          (perp_ask_px - perp_bid_px) / perp_bid_px,
            "ob_depth_span_spot":       s_depth_span,
            "ob_depth_span_perp":       p_depth_span,
            "oi_usd":                   oi_usd / perp_ask_px,
            "fund_rate":                fund_rate,
        })

        # OHLCV
        row.update({
            "spot_open":    spot_c["open"],    "spot_high":  spot_c["high"],
            "spot_low":     spot_c["low"],     "spot_close": spot_c["close"],
            "spot_minute_volume": spot_c["volCcyQuote"] / spot_ask_px,
            "perp_open":    perp_c["open"],    "perp_high":  perp_c["high"],
            "perp_low":     perp_c["low"],     "perp_close": perp_c["close"],
            "perp_minute_volume": perp_c["volCcyQuote"] / perp_ask_px,
            "mark_open":    mark_c["open"],    "mark_high":  mark_c["high"],
            "mark_low":     mark_c["low"],     "mark_close": mark_c["close"],
        })

        # OB metrics
        row.update({
            "spot_spread_bps":       spot_spread_bps,
            "spot_imbalance":        spot_imb,
            "spot_bid_concentration":spot_bid_conc,
            "spot_ask_concentration":spot_ask_conc,
            "spot_large_bid_count":  spot_lrg_bid,
            "spot_large_ask_count":  spot_lrg_ask,
            "perp_spread_bps":       perp_spread_bps,
            "perp_imbalance":        perp_imb,
            "perp_bid_concentration":perp_bid_conc,
            "perp_ask_concentration":perp_ask_conc,
            "perp_large_bid_count":  perp_lrg_bid,
            "perp_large_ask_count":  perp_lrg_ask,
            "microprice_spot":       mp_spot,
            "microprice_perp":       mp_perp,
        })

        # wall detection
        row.update({
            "spot_wall_bid_price": s_wall_bid_px, "spot_wall_bid_size": s_wall_bid_sz,
            "spot_wall_ask_price": s_wall_ask_px, "spot_wall_ask_size": s_wall_ask_sz,
            "perp_wall_bid_price": p_wall_bid_px, "perp_wall_bid_size": p_wall_bid_sz,
            "perp_wall_ask_price": p_wall_ask_px, "perp_wall_ask_size": p_wall_ask_sz,
        })

        # taker flow
        row.update({
            "taker_sell": taker_sell, "taker_buy": taker_buy,
            "taker_sell_buy_ratio": taker_sell / (taker_buy + 1e-12),
            "spot_sell_buy_side_deals": d_sell_spot / (d_buy_spot + 1e-12),
            "perp_sell_buy_side_deals": d_sell_perp / (d_buy_perp + 1e-12),
        })

        # NEW: liquidations
        row.update(liq_data)

        # NEW: long/short ratios
        row.update({
            "ls_ratio_all":       ls_all,
            "ls_ratio_top_acc":   ls_top_acc,
            "ls_ratio_top_pos":   ls_top_pos,
        })

        # NEW: index spread
        row["index_spread_bps"] = index_spread_bps

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
