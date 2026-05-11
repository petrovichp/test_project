"""Smoke test for the Hyperliquid API class. Run: python3 test_api.py"""

import sys
import time
from hyperliquid_api import Hyperliquid, parse_ob, parse_candle, coin_context, deal_imbalance

SEP  = "=" * 70
SEP2 = "-" * 70

def check(label, condition, detail=""):
    mark = "[OK]" if condition else "[FAIL]"
    print(f"  {mark}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def test_meta_and_contexts(api):
    print(SEP); print("metaAndAssetCtxs — BTC / ETH / SOL"); print(SEP2)
    raw = api.meta_and_contexts()
    ok = True
    ok &= check("response is 2-element list", isinstance(raw, list) and len(raw) == 2)
    for coin in ["BTC", "ETH", "SOL"]:
        ctx = coin_context(raw, coin)
        mark = float(ctx["markPx"])
        oi   = float(ctx["openInterest"])
        fund = float(ctx["funding"])
        ok &= check(f"{coin} mark_price={mark:,.2f}  oi={oi:,.2f}  fund={fund:.6f}",
                    mark > 0 and oi > 0)
    return ok


def test_orderbook(api):
    print(SEP); print("orderbook + parse_ob — BTC"); print(SEP2)
    raw = api.orderbook("BTC")
    bids, asks = parse_ob(raw)
    ok = True
    ok &= check("bids non-empty", len(bids) > 0, f"{len(bids)} levels")
    ok &= check("asks non-empty", len(asks) > 0, f"{len(asks)} levels")
    ok &= check("bids descending", bids.index[0] > bids.index[-1])
    ok &= check("asks ascending",  asks.index[0] < asks.index[-1])
    ok &= check("no cross", bids.index[0] < asks.index[0],
                f"best_bid={bids.index[0]:,.2f}  best_ask={asks.index[0]:,.2f}")
    mid = (bids.index[0] + asks.index[0]) / 2
    span_pct = min(bids.index[0]-bids.index[-1], asks.index[-1]-asks.index[0]) / mid * 100
    print(f"       spread={(asks.index[0]-bids.index[0])/bids.index[0]*10_000:.4f} bps  span={span_pct:.3f}%")
    return ok


def test_candles(api):
    print(SEP); print("candles + parse_candle — ETH 1m"); print(SEP2)
    raw = api.candles("ETH", interval="1m", lookback_ms=600_000)
    ok = True
    ok &= check("candles returned", len(raw) >= 2, f"{len(raw)} candles")
    c = parse_candle(raw)
    ok &= check("OHLCV keys present", all(k in c for k in ["open","high","low","close","volume"]))
    ok &= check("high >= low", c["high"] >= c["low"],
                f"O={c['open']} H={c['high']} L={c['low']} C={c['close']}")
    ok &= check("volume > 0", c["volume"] > 0, f"vol={c['volume']:.4f}")
    return ok


def test_recent_trades(api):
    print(SEP); print("recent_trades + deal_imbalance — SOL"); print(SEP2)
    trades = api.recent_trades("SOL")
    ok = True
    ok &= check("trades non-empty", len(trades) > 0, f"{len(trades)} trades")
    sides = set(t["side"] for t in trades)
    ok &= check("sides are B/A", sides.issubset({"B", "A"}), str(sides))
    sell_usd, buy_usd = deal_imbalance(trades, 60_000)
    ok &= check("deal imbalance computable", sell_usd + buy_usd >= 0,
                f"sell=${sell_usd:,.0f}  buy=${buy_usd:,.0f}")
    return ok


def main():
    t0 = time.time()
    api = Hyperliquid()
    results = [
        test_meta_and_contexts(api),
        test_orderbook(api),
        test_candles(api),
        test_recent_trades(api),
    ]
    print(SEP)
    passed = sum(results)
    print(f"Result: {'ALL PASS' if passed==len(results) else f'{passed}/{len(results)} PASSED'}    [{time.time()-t0:.1f}s]")
    print(SEP)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
