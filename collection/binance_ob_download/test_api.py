"""Smoke test for the Binance API class. Run: python3 test_api.py"""

import sys
import time
from binance_api import Binance, parse_ob, parse_kline, deal_imbalance

SEP  = "=" * 70
SEP2 = "-" * 70

def check(label, condition, detail=""):
    mark = "[OK]" if condition else "[FAIL]"
    print(f"  {mark}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def test_orderbooks(api):
    print(SEP); print("spot + perp orderbooks — BTCUSDT"); print(SEP2)
    spot_bids, spot_asks = parse_ob(api.spot_depth("BTCUSDT", limit=500))
    perp_bids, perp_asks = parse_ob(api.perp_depth("BTCUSDT", limit=500))
    ok = True
    for name, bids, asks in [("spot", spot_bids, spot_asks), ("perp", perp_bids, perp_asks)]:
        mid = (bids.index[0] + asks.index[0]) / 2
        span = min(bids.index[0]-bids.index[-1], asks.index[-1]-asks.index[0]) / mid * 100
        ok &= check(f"{name} no cross  best_bid={bids.index[0]:,.2f}  best_ask={asks.index[0]:,.2f}",
                    bids.index[0] < asks.index[0],
                    f"levels={len(bids)}+{len(asks)}  span={span:.3f}%")
    return ok


def test_klines(api):
    print(SEP); print("spot + perp klines — ETHUSDT 1m"); print(SEP2)
    ok = True
    for name, raw in [("spot", api.spot_klines("ETHUSDT","1m",3)),
                      ("perp", api.perp_klines("ETHUSDT","1m",3))]:
        c = parse_kline(raw)
        ok &= check(f"{name}  O={c['open']} H={c['high']} L={c['low']} C={c['close']}",
                    c["high"] >= c["low"] and c["volume"] > 0,
                    f"vol={c['volume']:.3f}  taker_buy={c['taker_buy_volume']:.3f}")
    return ok


def test_derivatives(api):
    print(SEP); print("premium_index + open_interest — SOLUSDT"); print(SEP2)
    prem = api.premium_index("SOLUSDT")
    oi   = api.open_interest("SOLUSDT")
    ok = True
    ok &= check("mark_price",   float(prem["markPrice"]) > 0,    f"mark={float(prem['markPrice']):.4f}")
    ok &= check("index_price",  float(prem["indexPrice"]) > 0,   f"index={float(prem['indexPrice']):.4f}")
    ok &= check("fund_rate",    "lastFundingRate" in prem,        f"rate={prem['lastFundingRate']}")
    ok &= check("open_interest", float(oi["openInterest"]) > 0,  f"oi={float(oi['openInterest']):,.2f}")
    return ok


def test_sentiment(api):
    print(SEP); print("taker ratio + long/short ratios — BTCUSDT"); print(SEP2)
    ok = True
    try:
        tk = api.taker_ls_ratio("BTCUSDT", period="5m", limit=1)[0]
        ok &= check("taker ratio", float(tk["buySellRatio"]) > 0,
                    f"buy={tk['buyVol']}  sell={tk['sellVol']}  ratio={tk['buySellRatio']}")
    except Exception as e:
        ok &= check("taker ratio", False, str(e))
    try:
        ls = api.ls_account_ratio("BTCUSDT", period="5m", limit=1)[0]
        ok &= check("ls_account_ratio", float(ls["longShortRatio"]) > 0,
                    f"long={ls['longAccount']}  short={ls['shortAccount']}")
    except Exception as e:
        ok &= check("ls_account_ratio", False, str(e))
    return ok


def test_deal_imbalance(api):
    print(SEP); print("deal_imbalance from trades — BTCUSDT spot + perp"); print(SEP2)
    ok = True
    for name, trades in [("spot", api.spot_trades("BTCUSDT", limit=500)),
                          ("perp", api.perp_trades("BTCUSDT", limit=500))]:
        sell, buy = deal_imbalance(trades, 60_000)
        ok &= check(f"{name}  sell=${sell:,.0f}  buy=${buy:,.0f}", sell + buy >= 0)
    return ok


def main():
    t0 = time.time()
    api = Binance()
    results = [
        test_orderbooks(api),
        test_klines(api),
        test_derivatives(api),
        test_sentiment(api),
        test_deal_imbalance(api),
    ]
    print(SEP)
    passed = sum(results)
    print(f"Result: {'ALL PASS' if passed==len(results) else f'{passed}/{len(results)} PASSED'}    [{time.time()-t0:.1f}s]")
    print(SEP)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
