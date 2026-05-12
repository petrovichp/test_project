"""Full row snapshot test — prints every column and value. Run: python3 test_full_row.py [BTC|ETH|SOL]"""

import os
import sys
import time
import math

from binance_api import (
    Binance, parse_ob, parse_kline, parse_price_kline,
    aggregate_buckets_hybrid, deal_imbalance,
)

N_LEVELS = 20
tok1 = (sys.argv[1] if len(sys.argv) > 1 else "BTC").upper()
symbol = Binance.SYMBOLS.get(tok1)
if symbol is None:
    print(f"ERROR: unknown token {tok1}"); sys.exit(1)

t0 = time.time()
api = Binance()
t_s = int(time.time())

spot_bids, spot_asks = parse_ob(api.spot_depth(symbol, limit=5000))
perp_bids, perp_asks = parse_ob(api.perp_depth(symbol, limit=1000))

spot_ask_px = float(spot_asks.index[0])
spot_bid_px = float(spot_bids.index[0])
perp_ask_px = float(perp_asks.index[0])
perp_bid_px = float(perp_bids.index[0])

spot_c  = parse_kline(api.spot_klines(symbol, interval="1m", limit=3))
perp_c  = parse_kline(api.perp_klines(symbol, interval="1m", limit=3))
mark_c  = parse_price_kline(api.mark_price_klines(symbol, interval="1m", limit=3))
index_c = parse_price_kline(api.index_price_klines(symbol, interval="1m", limit=3))
try:
    premium_index = parse_price_kline(
        api.premium_index_klines(symbol, interval="1m", limit=3))["close"]
except Exception:
    premium_index = 0.0

prem         = api.premium_index(symbol)
mark_px      = float(prem["markPrice"])
index_px     = float(prem["indexPrice"])
fund_rate    = float(prem["lastFundingRate"])
next_fund_ms = float(prem.get("nextFundingTime", 0) or 0)
time_to_funding_mins = (next_fund_ms / 1000 - t_s) / 60 if next_fund_ms > 0 else 0.0
oi_usd       = float(api.open_interest(symbol)["openInterest"]) * mark_px

try:
    tk = api.taker_ls_ratio(symbol, period="5m", limit=1)[0]
    taker_buy  = float(tk["buyVol"])
    taker_sell = float(tk["sellVol"])
except Exception:
    taker_buy = taker_sell = 0.0

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

try:
    d_sell_spot, d_buy_spot = deal_imbalance(api.spot_trades(symbol, limit=1000))
except Exception:
    d_sell_spot = d_buy_spot = 0.0
try:
    d_sell_perp, d_buy_perp = deal_imbalance(api.perp_trades(symbol, limit=1000))
except Exception:
    d_sell_perp = d_buy_perp = 0.0

def _imb(b, a):
    tb, ta = b["amount"].sum(), a["amount"].sum()
    return (tb - ta) / (tb + ta + 1e-12)

def wall(side, top_n=100):
    top = side.head(top_n)
    if top.empty: return 0.0, 0.0
    idx = top["amount"].idxmax()
    return float(idx), float(top.loc[idx, "size"])

spot_imb = _imb(spot_bids, spot_asks)
perp_imb = _imb(perp_bids, perp_asks)
p_wbp, p_wbs = wall(perp_bids)
p_wap, p_was = wall(perp_asks)

row = {"timestamp": t_s}

for prefix, bids, asks in [("spot", spot_bids, spot_asks),
                            ("perp", perp_bids, perp_asks)]:
    for side_name, buckets in [("bid", aggregate_buckets_hybrid(bids)),
                                ("ask", aggregate_buckets_hybrid(asks))]:
        for i, usd in enumerate(buckets, 1):
            row[f"{prefix}_{side_name}_lev{i:02d}_usd"] = usd

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
row.update({
    "spot_imbalance":      spot_imb,
    "perp_imbalance":      perp_imb,
    "perp_wall_bid_price": p_wbp,
    "perp_wall_bid_size":  p_wbs,
    "perp_wall_ask_price": p_wap,
    "perp_wall_ask_size":  p_was,
})
row.update({
    "taker_sell":               taker_sell,
    "taker_buy":                taker_buy,
    "spot_sell_buy_side_deals": (d_sell_spot / d_buy_spot) if d_buy_spot > 0 else 0.0,
    "perp_sell_buy_side_deals": (d_sell_perp / d_buy_perp) if d_buy_perp > 0 else 0.0,
    "ls_ratio_all":             ls_all,
    "ls_ratio_top_acc":         ls_top_acc,
    "ls_ratio_top_pos":         ls_top_pos,
})

elapsed = time.time() - t0

# ── OB depth info ─────────────────────────────────────────────────────────────
spot_depth_pct = (spot_bids.index[0] - spot_bids.index[-1]) / spot_bids.index[0] * 100
perp_depth_pct = (perp_bids.index[0] - perp_bids.index[-1]) / perp_bids.index[0] * 100

SEP = "=" * 80
print(SEP)
print(f"  binance_{tok1.lower()}usdt   cols={len(row)}   ts={t_s}   [{elapsed:.1f}s]")
print(f"  spot OB: {len(spot_bids)} bid + {len(spot_asks)} ask levels   depth={spot_depth_pct:.3f}% from mid")
print(f"  perp OB: {len(perp_bids)} bid + {len(perp_asks)} ask levels   depth={perp_depth_pct:.3f}% from mid")
print(SEP)

sections = [
    ("OB BUCKETS — SPOT BID", [k for k in row if k.startswith("spot_bid_")]),
    ("OB BUCKETS — SPOT ASK", [k for k in row if k.startswith("spot_ask_lev")]),
    ("OB BUCKETS — PERP BID", [k for k in row if k.startswith("perp_bid_")]),
    ("OB BUCKETS — PERP ASK", [k for k in row if k.startswith("perp_ask_lev")]),
    ("PRICES & DERIVATIVES",  ["spot_ask_price","spot_bid_price","perp_ask_price","perp_bid_price",
                                "mark_price","index_price","diff_price","oi_usd",
                                "fund_rate","time_to_funding_mins"]),
    ("SPOT OHLCV",            ["spot_open","spot_high","spot_low","spot_close",
                                "spot_minute_volume","spot_minute_volume_usd",
                                "spot_taker_buy_vol","spot_taker_sell_vol","spot_num_trades"]),
    ("PERP OHLCV",            ["perp_open","perp_high","perp_low","perp_close",
                                "perp_minute_volume","perp_minute_volume_usd",
                                "perp_taker_buy_vol","perp_taker_sell_vol","perp_num_trades"]),
    ("MARK / INDEX OHLC",     ["mark_open","mark_high","mark_low","mark_close",
                                "index_open","index_high","index_low","index_close","premium_index"]),
    ("OB METRICS",            ["spot_imbalance","perp_imbalance",
                                "perp_wall_bid_price","perp_wall_bid_size",
                                "perp_wall_ask_price","perp_wall_ask_size"]),
    ("TAKER FLOW & SENTIMENT",["taker_sell","taker_buy",
                                "spot_sell_buy_side_deals","perp_sell_buy_side_deals",
                                "ls_ratio_all","ls_ratio_top_acc","ls_ratio_top_pos"]),
]

for title, keys in sections:
    print(f"\n  {title}")
    print("  " + "-" * 60)
    for k in keys:
        v = row.get(k, "MISSING")
        if isinstance(v, float):
            print(f"  {k:<40s}  {v:>15.4f}")
        else:
            print(f"  {k:<40s}  {v!r:>15}")
