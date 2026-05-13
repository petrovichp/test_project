"""
Step 3 — Maker-on-entry feasibility for S1_VolDir (and per-strategy
breakdown). Tests whether passive limit orders at entry can save the 2.5bp
maker/taker differential.

Method (snapshot-based, like P1 revised):
  1. For each trade, simulate "would my limit at decision-bar price have
     filled within N bars before entry?"
     - For long entry at decision_bar t: limit at prices[t]; filled if any
       future bar k in [1, N] has prices[t+k] <= prices[t]
     - For short entry at t: filled if any prices[t+k] >= prices[t]
  2. If filled within N bars: entry_fee = 2bp (maker)
  3. If not filled: entry_fee = 4.5bp (taker, current behavior)
  4. Exit fee stays per P1 revised rule (TP=2bp, else=4.5bp)

Caveats:
  - Assumes filled price = decision-bar snapshot price (no slippage modeled)
  - Adverse selection: maker-filled trades may have worse PnL because price
    came back to our limit (= momentum reversed against our signal)
  - The current simulator uses prices[entry_bar] = prices[t+1]; we keep
    that entry price and just adjust the fee
  - Patience N tested: {1, 3, 5, 10} bars

Output: per-strategy maker fill rate + per-strategy WF Sharpe under
the maker-on-entry model.

Run: python3 -m models.maker_on_entry_eval
"""
import json, math, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.audit_vote5_dd import run_walkforward

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
STRAT_KEYS = ["S1_VolDir","S2_Funding","S3_BBExt","S4_MACD","S6_TwoSignal",
              "S7_OIDiverg","S8_TakerSus","S10_Squeeze","S12_VWAPVol",
              "S11_Basis","S13_OBDiv"]
TAKER_FEE = 0.00045
MAKER_FEE = 0.00020
TAKER_BPS = 4.5
MAKER_BPS = 2.0
RL_START = 100_000
RL_END   = 383_174
N_FOLDS  = 6
FOLD_SIZE = (RL_END - RL_START) // N_FOLDS
N_BARS_PER_YEAR = 525_960


def load_v8_nets():
    out = []
    for s in SEEDS:
        n = DuelingDQN(52, 12, 256)
        n.load_state_dict(torch.load(
            CACHE / "policies" / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt", map_location="cpu"))
        n.eval(); out.append(n)
    return out


def load_full():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for key in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def reconstruct_sharpe(trades, pnl_field):
    by_fold = defaultdict(list)
    for tr in trades:
        by_fold[tr["fold"]].append(tr)
    per_fold = []
    for fid in range(1, N_FOLDS + 1):
        tl = sorted(by_fold.get(fid, []), key=lambda t: t["t_close"])
        n_bars = FOLD_SIZE if fid < N_FOLDS else (RL_END - RL_START) - (N_FOLDS - 1) * FOLD_SIZE
        eq = np.full(n_bars, 1.0); cur = 1.0
        for tr in tl:
            tc = int(tr["t_close"])
            if 0 <= tc < n_bars:
                cur *= 1.0 + tr[pnl_field]
                eq[tc:] = cur
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sh = float(rets.mean() / rets.std() * math.sqrt(N_BARS_PER_YEAR)) if rets.std() > 1e-12 else 0.0
        per_fold.append(dict(fold=fid, sharpe=sh, n_trades=len(tl)))
    wf = statistics.mean(r["sharpe"] for r in per_fold)
    return wf, per_fold


def estimate_maker_fill(prices, decision_bar_global, direction, patience_bars):
    """Return True if a passive limit at prices[decision_bar_global] would
    fill within `patience_bars` bars under the snapshot-based proxy.

    Long (direction=+1): bid at prices[t]; filled if any prices[t+k] <= prices[t]
    Short (direction=-1): ask at prices[t]; filled if any prices[t+k] >= prices[t]
    """
    limit = prices[decision_bar_global]
    end = min(len(prices), decision_bar_global + 1 + patience_bars)
    for k in range(decision_bar_global + 1, end):
        if direction == 1 and prices[k] <= limit:
            return True
        if direction == -1 and prices[k] >= limit:
            return True
    return False


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full()
    full_prices = full["price"]
    n_total = len(full_prices)
    nets = load_v8_nets()

    print(f"\n{'='*120}\n  Step 3 — Maker-on-entry feasibility")
    print(f"  taker {TAKER_BPS}bp, maker {MAKER_BPS}bp; entry maker fill estimated from snapshot price proxy")
    print(f"{'='*120}\n")

    print(f"  Collecting trades at fee={TAKER_BPS}bp uniform ...")
    rows, trades = run_walkforward(nets, full, atr_median, fee=TAKER_FEE, with_reason=True)
    print(f"    {len(trades)} trades collected\n")

    # ── Per-strategy maker fill rate at different patience N ──
    print(f"{'='*120}\n  Per-strategy maker fill rate (snapshot-based proxy)\n{'='*120}\n")
    patiences = [1, 3, 5, 10]
    print(f"  {'strategy':<14} {'n_trades':>9} " + " ".join(f"{'N=':<3}{n:<3}" for n in patiences))
    fills_by_strat_N = defaultdict(dict)
    for k_idx in range(11):
        sub = [tr for tr in trades if tr["strat_idx"] == k_idx]
        if not sub: continue
        row = [STRAT_KEYS[k_idx], len(sub)]
        for N in patiences:
            fills = 0
            for tr in sub:
                decision_bar = (tr["fold"] - 1) * FOLD_SIZE + tr["t_open"] - 1
                if not (0 <= decision_bar < n_total): continue
                if estimate_maker_fill(full_prices, decision_bar, tr["direction"], N):
                    fills += 1
            rate = fills / len(sub) * 100
            fills_by_strat_N[k_idx][N] = rate
            row.append(f"{rate:>5.1f}%")
        print(f"  {row[0]:<14} {row[1]:>9} " + " ".join(f"{v}" for v in row[2:]))

    # ── Per-strategy adverse selection check: filled vs unfilled PnL ──
    print(f"\n{'='*120}\n  Adverse selection check (N=5): filled-subset vs unfilled-subset mean PnL%\n{'='*120}\n")
    N_default = 5
    print(f"  {'strategy':<14} {'n':>5} {'fill%':>7} {'mean_pnl_all%':>14} {'mean_pnl_filled%':>17} {'mean_pnl_miss%':>15} {'adverse?':>9}")
    for k_idx in range(11):
        sub = [tr for tr in trades if tr["strat_idx"] == k_idx]
        if not sub: continue
        filled_pnls, miss_pnls = [], []
        for tr in sub:
            decision_bar = (tr["fold"] - 1) * FOLD_SIZE + tr["t_open"] - 1
            if not (0 <= decision_bar < n_total): continue
            if estimate_maker_fill(full_prices, decision_bar, tr["direction"], N_default):
                filled_pnls.append(tr["pnl"])
            else:
                miss_pnls.append(tr["pnl"])
        all_pnls = filled_pnls + miss_pnls
        fill_pct = len(filled_pnls) / max(1, len(all_pnls)) * 100
        ma = np.mean(all_pnls)*100 if all_pnls else 0
        mf = np.mean(filled_pnls)*100 if filled_pnls else 0
        mm = np.mean(miss_pnls)*100 if miss_pnls else 0
        adv = "YES" if (filled_pnls and miss_pnls and mf < mm - 0.05) else ""
        print(f"  {STRAT_KEYS[k_idx]:<14} {len(sub):>5} {fill_pct:>6.1f}% "
              f"{ma:>+14.3f} {mf:>+17.3f} {mm:>+15.3f} {adv:>9}")

    # ── Apply fee savings & compute new Sharpe at each patience ──
    print(f"\n{'='*120}\n  WF Sharpe under maker-on-entry (per patience N)\n{'='*120}\n")

    # Build per-trade decision_bar and filled flag for each N
    for tr in trades:
        tr["decision_bar"] = (tr["fold"] - 1) * FOLD_SIZE + tr["t_open"] - 1

    print(f"  {'scheme':<60} {'WF':>8} {'pos':>5}")

    # Baseline: uniform 9bp (the same we ran at fee=4.5bp)
    for tr in trades:
        tr["pnl_baseline"] = tr["pnl"]   # already at 9bp
    wf_b, pf_b = reconstruct_sharpe(trades, "pnl_baseline")
    print(f"  {'BASELINE uniform 4.5bp/4.5bp':<60} {wf_b:>+8.3f}   {sum(1 for r in pf_b if r['sharpe']>0)}/6")

    # P1 revised: TP exit maker
    maker_savings_exit = TAKER_FEE - MAKER_FEE
    for tr in trades:
        tr["pnl_tp_maker"] = tr["pnl"] + (maker_savings_exit if tr["exit_reason"] == 0 else 0.0)
    wf, pf = reconstruct_sharpe(trades, "pnl_tp_maker")
    print(f"  {'+ TP-exit maker (P1 revised baseline)':<60} {wf:>+8.3f}   {sum(1 for r in pf if r['sharpe']>0)}/6")

    # Maker-on-entry per patience (apply to ALL strategies)
    for N in patiences:
        for tr in trades:
            d = tr["decision_bar"]
            if not (0 <= d < n_total):
                tr[f"pnl_maker_N{N}"] = tr["pnl_tp_maker"]; continue
            filled = estimate_maker_fill(full_prices, d, tr["direction"], N)
            entry_savings = maker_savings_exit if filled else 0.0
            tr[f"pnl_maker_N{N}"] = tr["pnl_tp_maker"] + entry_savings
        wf, pf = reconstruct_sharpe(trades, f"pnl_maker_N{N}")
        print(f"  {f'+ maker-on-entry all strategies (N={N})':<60} {wf:>+8.3f}   {sum(1 for r in pf if r['sharpe']>0)}/6")

    # Maker-on-entry restricted to S1 only (the proposal scope)
    for N in patiences:
        for tr in trades:
            d = tr["decision_bar"]
            if not (0 <= d < n_total) or tr["strat_idx"] != 0:  # S1 only
                tr[f"pnl_maker_S1_N{N}"] = tr["pnl_tp_maker"]; continue
            filled = estimate_maker_fill(full_prices, d, tr["direction"], N)
            entry_savings = maker_savings_exit if filled else 0.0
            tr[f"pnl_maker_S1_N{N}"] = tr["pnl_tp_maker"] + entry_savings
        wf, pf = reconstruct_sharpe(trades, f"pnl_maker_S1_N{N}")
        print(f"  {f'+ maker-on-entry S1_VolDir only (N={N})':<60} {wf:>+8.3f}   {sum(1 for r in pf if r['sharpe']>0)}/6")

    # Combined: maker-on-entry + AGGRESSIVE sizing (R1) — full stack
    print(f"\n{'='*120}\n  Full stack: maker-on-entry + R1 AGGRESSIVE sizing\n{'='*120}\n")
    print(f"  {'scheme':<60} {'WF':>8} {'pos':>5}")
    for N in patiences:
        for tr in trades:
            d = tr["decision_bar"]
            if 0 <= d < n_total:
                filled = estimate_maker_fill(full_prices, d, tr["direction"], N)
                entry_savings = maker_savings_exit if filled else 0.0
            else:
                entry_savings = 0.0
            base_pnl = tr["pnl"] + entry_savings + (maker_savings_exit if tr["exit_reason"] == 0 else 0.0)
            sz = ((tr["votes_count"] - 2) / 3) ** 2
            tr[f"pnl_stack_N{N}"] = base_pnl * sz
        wf, pf = reconstruct_sharpe(trades, f"pnl_stack_N{N}")
        print(f"  {f'all-strategies maker-on-entry (N={N}) + sizing':<60} {wf:>+8.3f}   {sum(1 for r in pf if r['sharpe']>0)}/6")

    out = CACHE / "results" / "maker_on_entry_eval.json"
    out.write_text(json.dumps(dict(elapsed=time.perf_counter()-t0), indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
