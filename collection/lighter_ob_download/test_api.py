"""
Smoke test for the Lighter API class.
Run: python test_api.py
"""

import sys
import time

from lighter_api import (
    Lighter, parse_ob, parse_candle, parse_funding,
    deal_imbalance, liquidation_stats,
)

SEP  = "=" * 70
SEP2 = "-" * 70

PASS = "[OK]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = ""):
    mark = PASS if condition else FAIL
    print(f"  {mark}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def test_market_details(api: Lighter) -> bool:
    print(SEP)
    print("market_details — BTC / ETH / SOL")
    print(SEP2)
    ok = True
    for name, mid in Lighter.MARKETS.items():
        raw = api.market_details(mid)
        price = float(raw.get("last_trade_price", 0))
        oi    = float(raw.get("open_interest", 0))
        sym   = raw.get("symbol", "?")
        ok &= check(
            f"{name} (market_id={mid}) symbol={sym}",
            price > 0 and oi >= 0,
            f"last_price={price:,.2f}  OI={oi:,.4f}",
        )
    return ok


def test_orderbook(api: Lighter) -> bool:
    print(SEP)
    print("orderbook + parse_ob — BTC")
    print(SEP2)
    raw = api.orderbook(market_id=1, limit=200)
    bids, asks = parse_ob(raw)

    ok = True
    ok &= check("bids non-empty", len(bids) > 0, f"{len(bids)} price levels")
    ok &= check("asks non-empty", len(asks) > 0, f"{len(asks)} price levels")
    ok &= check("bids descending", bids.index[0] > bids.index[-1])
    ok &= check("asks ascending",  asks.index[0] < asks.index[-1])
    ok &= check("bid < ask (no cross)",
                bids.index[0] < asks.index[0],
                f"best_bid={bids.index[0]:,.2f}  best_ask={asks.index[0]:,.2f}")

    mid = (bids.index[0] + asks.index[0]) / 2
    spread_bps = (asks.index[0] - bids.index[0]) / mid * 10_000
    print(f"       mid={mid:,.2f}  spread={spread_bps:.2f} bps")
    print(f"       top-3 bids: {list(bids.index[:3])}")
    print(f"       top-3 asks: {list(asks.index[:3])}")
    return ok


def test_candles(api: Lighter) -> bool:
    print(SEP)
    print("candles + parse_candle — BTC 1m")
    print(SEP2)
    raw = api.candles(market_id=1, resolution="1m", count_back=3)
    c = parse_candle(raw)

    ok = True
    ok &= check("OHLCV keys present",
                all(k in c for k in ["open", "high", "low", "close", "volume"]))
    ok &= check("high >= low", c["high"] >= c["low"],
                f"O={c['open']} H={c['high']} L={c['low']} C={c['close']}")
    ok &= check("volume > 0", c["volume"] > 0,
                f"vol={c['volume']:.4f} BTC  vol_usd=${c['volume_usd']:,.0f}")
    return ok


def test_recent_trades(api: Lighter) -> bool:
    print(SEP)
    print("recent_trades + deal_imbalance + liquidation_stats — ETH")
    print(SEP2)
    raw = api.recent_trades(market_id=0, limit=100)
    trades = raw.get("trades", [])

    ok = True
    ok &= check("trades non-empty", len(trades) > 0, f"{len(trades)} trades")

    sell_usd, buy_usd = deal_imbalance(raw, lookback_ms=60_000)
    ok &= check("deal imbalance computable", (sell_usd + buy_usd) >= 0,
                f"sell=${sell_usd:,.0f}  buy=${buy_usd:,.0f}")

    liq = liquidation_stats(raw)
    ok &= check("liquidation stats present",
                "liq_buy_usd" in liq and "liq_sell_usd" in liq,
                f"liq_buy=${liq['liq_buy_usd']:,.0f}  liq_sell=${liq['liq_sell_usd']:,.0f}")

    types = set(t["type"] for t in trades)
    print(f"       trade types seen: {types}")
    return ok


def test_funding(api: Lighter) -> bool:
    print(SEP)
    print("funding + parse_funding — SOL 1h")
    print(SEP2)
    raw = api.funding(market_id=2, resolution="1h", count_back=3)
    rate = parse_funding(raw)

    ok = True
    ok &= check("funding rate parsed", isinstance(rate, float),
                f"rate={rate:.6f}")
    fundings = raw.get("fundings", [])
    ok &= check("entries returned", len(fundings) > 0, f"{len(fundings)} entries")
    if fundings:
        latest = fundings[-1]
        print(f"       latest: rate={latest['rate']}  "
              f"direction={latest.get('direction', '?')}  "
              f"ts={latest['timestamp']}")
    return ok


def main():
    t0 = time.time()
    api = Lighter()

    results = [
        test_market_details(api),
        test_orderbook(api),
        test_candles(api),
        test_recent_trades(api),
        test_funding(api),
    ]

    print(SEP)
    passed = sum(results)
    total  = len(results)
    status = "ALL PASS" if passed == total else f"{passed}/{total} PASSED"
    print(f"Result: {status}    [{time.time() - t0:.1f}s]")
    print(SEP)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
