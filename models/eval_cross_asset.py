"""
Z2.1 cross-asset evaluation — VOTE5_v8_H256_DD trained on ETH and SOL.

Compares per-ticker WF / val / test against the BTC baseline. Reports
per-fold breakdown to show whether the BTC strategy stack composes
on different liquidity / volatility regimes.

Run: python3 -m models.eval_cross_asset
"""
import json, pathlib, statistics, sys, time
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.audit_vote5_dd import run_fold, run_walkforward

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]
N_FOLDS = 6


def load_net(ticker: str, seed: int) -> DuelingDQN:
    net = DuelingDQN(52, 12, 256)
    tag = f"VOTE5_v8_H256_DD" if ticker == "btc" else f"VOTE5_v8_H256_DD_{ticker}"
    net.load_state_dict(torch.load(
        CACHE / "policies" / f"{ticker}_dqn_policy_{tag}_seed{seed}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full(ticker: str):
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"{ticker}_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts", "regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_ticker(ticker: str):
    vol = np.load(CACHE / "preds" / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    nets = [load_net(ticker, s) for s in SEEDS]
    full = load_full(ticker)
    n_full = len(full["price"])

    # Per-ticker fold boundaries derived from the RL period (train+val+test)
    rl_start = 0
    rl_end = n_full
    fold_size = (rl_end - rl_start) // N_FOLDS

    rows = []
    all_trades = []
    for i in range(N_FOLDS):
        a = rl_start + i * fold_size
        b = rl_start + (i + 1) * fold_size if i < N_FOLDS - 1 else rl_end
        sub = {k: full[k][a:b] for k in full}
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        eq, sh, eq_f, trades = run_fold(
            sub["state"], sub["valid_actions"], sub["signals"], sub["price"],
            sub["atr"], sub["regime_id"], sub["ts"],
            tp, sl, tr, tab, be, ts, nets, fee=0.0, fold_id=i+1)
        btc_ret = (sub["price"][-1] / sub["price"][0] - 1) * 100
        rows.append(dict(fold=i+1, sharpe=sh, equity=eq_f, trades=len(trades),
                         btc_ret=btc_ret))
        all_trades.extend(trades)

    wf = statistics.mean(r["sharpe"] for r in rows)
    pos = sum(1 for r in rows if r["sharpe"] > 0)

    # val + test single-shot
    sp_v = np.load(CACHE / "state" / f"{ticker}_dqn_state_val_v8_s11s13.npz")
    sp_t = np.load(CACHE / "state" / f"{ticker}_dqn_state_test_v8_s11s13.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)
    _, vsh, veq, vtr = run_fold(sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
                                  sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
                                  tp_v, sl_v, tr_v, tab_v, be_v, ts_v, nets, fee=0.0, fold_id=0)
    _, tsh, teq, ttr = run_fold(sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
                                  sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
                                  tp_t, sl_t, tr_t, tab_t, be_t, ts_t, nets, fee=0.0, fold_id=0)

    print(f"\n  {ticker.upper()} VOTE5_v8_H256_DD")
    print(f"  {'fold':<5} {'BTC ret':>9} {'Sharpe':>9} {'trades':>7} {'equity':>8}")
    for r in rows:
        print(f"  {r['fold']:<5} {r['btc_ret']:>+8.2f}% {r['sharpe']:>+9.3f} {r['trades']:>7} {r['equity']:>8.3f}")
    print(f"  WF mean Sharpe: {wf:>+8.3f}  ({pos}/6 folds positive)")
    print(f"  val: Sharpe={vsh:>+7.3f}  eq={veq:>5.3f}  trades={len(vtr)}")
    print(f"  test: Sharpe={tsh:>+7.3f}  eq={teq:>5.3f}  trades={len(ttr)}")

    return dict(ticker=ticker, wf=wf, wf_pos=pos, val=vsh, test=tsh,
                val_eq=veq, test_eq=teq, val_trades=len(vtr), test_trades=len(ttr),
                per_fold=rows)


def main():
    t0 = time.perf_counter()
    print(f"\n{'='*120}\n  Z2.1 — VOTE5_v8_H256_DD cross-asset evaluation\n{'='*120}")
    out_rows = []
    for ticker in ("btc", "eth", "sol"):
        try:
            r = eval_ticker(ticker)
            out_rows.append(r)
        except FileNotFoundError as e:
            print(f"\n  {ticker.upper()}: not trained yet ({e})")

    print(f"\n{'='*120}\n  SUMMARY")
    print(f"  {'ticker':<6} {'WF':>9} {'val':>9} {'test':>9} {'folds+':>7} {'val tr':>7} {'test tr':>8}")
    for r in out_rows:
        print(f"  {r['ticker']:<6} {r['wf']:>+9.3f} {r['val']:>+9.3f} {r['test']:>+9.3f} "
              f"{r['wf_pos']:>4}/6  {r['val_trades']:>7} {r['test_trades']:>8}")

    out = CACHE / "results" / "cross_asset_eval.json"
    out.write_text(json.dumps(out_rows, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
