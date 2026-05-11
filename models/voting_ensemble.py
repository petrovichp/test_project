"""
Vote-based ensemble for BASELINE_FULL.

Instead of averaging Q-values across K nets (which produces qualitatively new
policies — see docs/ensemble_baseline.md), each net argmaxes independently
and votes are aggregated.

Variants tested:
  - q_avg:        Q-value average then argmax (re-tested for K=10 reference)
  - plurality:    most-common vote wins; tie → NO_TRADE
  - majority_t:   action wins if it has ≥ t/K votes; else NO_TRADE
  - unanimous:    action wins only if ALL K nets voted for it; else NO_TRADE

Reports:
  - val + test single-shot Sharpes
  - WF 6-fold per-fold Sharpes + aggregate
  - Per-bar vote-agreement histogram (how often N nets agreed on the most-voted action)
"""
import json, pathlib, statistics, time
from collections import Counter
from typing import Callable
import numpy as np
import torch

from models.dqn_network          import DQN
from models.dqn_rollout          import _build_exit_arrays
from models.group_c2_walkforward import RL_START_REL, RL_END_REL

CACHE = pathlib.Path("cache")
N_FOLDS = 6


# ── voting policies ──────────────────────────────────────────────────────────

class _VotePolicy:
    """Vote-based ensemble policy. mode controls aggregation. Records vote
    histogram across calls if `track_agreement=True`."""

    def __init__(self, nets, mode: str = "plurality", threshold: int = None,
                  track_agreement: bool = False):
        self.nets = nets
        self.K    = len(nets)
        self.mode = mode
        self.threshold = threshold
        self.track = track_agreement
        if track_agreement:
            # agreement_hist[k] = number of bars where the most-voted action got k votes
            self.agreement_hist = np.zeros(self.K + 1, dtype=np.int64)
            # n_trade_by_agreement[k] = number of trades fired when top action got k votes
            self.n_trade_by_agreement = np.zeros(self.K + 1, dtype=np.int64)

    def __call__(self, s: np.ndarray, v: np.ndarray) -> int:
        """state s (50,) and valid mask v (10,) → chosen action int."""
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool()                 # (10,)

            if self.mode == "q_avg":
                Qs = torch.stack([net(sb) for net in self.nets], dim=0)  # (K,1,A)
                q  = Qs.mean(dim=0).squeeze(0)                            # (A,)
                q  = q.masked_fill(~vb, -1e9)
                action = int(q.argmax().item())
                if self.track:
                    self.agreement_hist[1] += 1   # treat Q-avg as "1-vote" winner
                return action

            # vote-based (plurality / majority_t / unanimous)
            votes = []
            for net in self.nets:
                q = net(sb).squeeze(0)
                q = q.masked_fill(~vb, -1e9)
                votes.append(int(q.argmax().item()))

            counts = Counter(votes)
            top_action, top_count = counts.most_common(1)[0]

            if self.track:
                self.agreement_hist[top_count] += 1

            if self.mode == "plurality":
                # tie → NO_TRADE
                top_two = counts.most_common(2)
                if len(top_two) >= 2 and top_two[0][1] == top_two[1][1]:
                    action = 0
                else:
                    action = top_action
            elif self.mode == "majority_t":
                action = top_action if top_count >= self.threshold else 0
            elif self.mode == "unanimous":
                action = top_action if top_count == self.K else 0
            else:
                raise ValueError(f"unknown mode {self.mode}")

            if self.track and action != 0:
                self.n_trade_by_agreement[top_count] += 1
            return action


# ── eval loop (mirrors models/dqn_selector.evaluate_policy but takes policy callable) ──

def evaluate_with_policy(policy: Callable, state, valid, signals, prices,
                          tp, sl, trail, tab, be, ts_bars, fee: float = 0.0):
    from models.diagnostics_ab import _simulate_one_trade_fee
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []
    n_trades = 0
    n_actions = np.zeros(valid.shape[1], dtype=np.int64)
    n_steps = 0

    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False
        action = policy(s_t, valid_t)
        n_actions[action] += 1
        n_steps += 1

        if action == 0:
            t_next = t + 1
        else:
            k = action - 1
            direction = int(signals[t, k])
            if direction == 0:
                t_next = t + 1
            else:
                pnl, n_held = _simulate_one_trade_fee(
                    prices, t + 1, direction,
                    float(tp[t, k]), float(sl[t, k]),
                    float(trail[t, k]), float(tab[t, k]),
                    float(be[t, k]),   int(ts_bars[t, k]),
                    0, fee,
                )
                t_close = t + 1 + n_held
                if t_close >= n_bars: t_close = n_bars - 1
                eq_arr[t:t_close + 1] = equity
                equity *= (1.0 + float(pnl))
                eq_arr[t_close + 1:] = equity
                if t_close == n_bars - 1: eq_arr[-1] = equity
                peak = max(peak, equity)
                last_pnl = float(pnl)
                trade_pnls.append(float(pnl))
                n_trades += 1
                t_next = t_close + 1
        t = t_next

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    win_rate = (np.array(trade_pnls) > 0).mean() if trade_pnls else 0.0
    eq_dd = eq_arr / np.maximum.accumulate(eq_arr)
    max_dd = float(eq_dd.min() - 1.0)
    return dict(sharpe=float(sharpe), equity_final=float(equity),
                 n_trades=int(n_trades), win_rate=float(win_rate),
                 max_dd=max_dd)


# ── data utilities ───────────────────────────────────────────────────────────

def load_net(tag: str) -> DQN:
    net = DQN(50, 10, 64)
    net.load_state_dict(torch.load(CACHE / "policies" / f"btc_dqn_policy_{tag}.pt", map_location="cpu"))
    net.eval()
    return net


def load_full_rl_period(ticker: str = "btc"):
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"{ticker}_dqn_state_{split}.npz")
        for key in ("state", "valid_actions", "signals", "price", "atr", "ts"):
            arrs.setdefault(key, []).append(sp[key])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_split(policy: Callable, split: str, atr_median: float):
    sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}.npz")
    tp, sl, tr, tab, be, ts = _build_exit_arrays(sp["price"], sp["atr"], atr_median)
    out = evaluate_with_policy(policy, sp["state"], sp["valid_actions"],
                                 sp["signals"], sp["price"], tp, sl, tr, tab, be, ts)
    return dict(sharpe=out["sharpe"], equity=out["equity_final"],
                trades=out["n_trades"], win_rate=out["win_rate"], max_dd=out["max_dd"])


def eval_walkforward(policy_factory: Callable, atr_median: float, full):
    """policy_factory: () -> _VotePolicy (fresh per fold so trackers reset)."""
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS
    rows = []
    for i in range(N_FOLDS):
        a_pq = RL_START_REL + i * fold_size
        b_pq = RL_START_REL + (i + 1) * fold_size if i < N_FOLDS - 1 else RL_END_REL
        a = a_pq - RL_START_REL; b = b_pq - RL_START_REL
        sub_state   = full["state"][a:b]
        sub_valid   = full["valid_actions"][a:b]
        sub_signals = full["signals"][a:b]
        sub_prices  = full["price"][a:b]
        sub_atr     = full["atr"][a:b]
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub_prices, sub_atr, atr_median)
        pol = policy_factory()
        out = evaluate_with_policy(pol, sub_state, sub_valid, sub_signals, sub_prices,
                                     tp, sl, tr, tab, be, ts)
        rows.append(dict(fold=i+1, sharpe=out["sharpe"],
                          equity=out["equity_final"], trades=out["n_trades"]))
    return rows


def run_variant(name: str, mode: str, threshold: int, K_seeds: list,
                 atr_median: float, full, track_agreement: bool = False):
    nets = [load_net(_TAG_FOR[s]) for s in K_seeds]

    def make_policy():
        return _VotePolicy(nets, mode=mode, threshold=threshold,
                            track_agreement=track_agreement)

    pol_v = make_policy(); v = eval_split(pol_v, "val",  atr_median)
    pol_t = make_policy(); t = eval_split(pol_t, "test", atr_median)
    wf    = eval_walkforward(make_policy, atr_median, full)

    res = dict(
        name=name, mode=mode, threshold=threshold, K=len(K_seeds), seeds=K_seeds,
        val_sharpe=v["sharpe"], val_eq=v["equity"], val_trades=v["trades"],
        test_sharpe=t["sharpe"], test_eq=t["equity"], test_trades=t["trades"],
        wf_per_fold=[r["sharpe"] for r in wf],
        wf_eq_per_fold=[r["equity"] for r in wf],
        wf_trades_per_fold=[r["trades"] for r in wf],
        wf_mean=statistics.mean([r["sharpe"] for r in wf]),
        wf_pos=sum(1 for r in wf if r["sharpe"] > 0),
    )
    if track_agreement:
        # combine the val tracker only (single trajectory diagnostic)
        res["val_agreement_hist"]    = pol_v.agreement_hist.tolist()
        res["val_n_trade_by_agree"]  = pol_v.n_trade_by_agreement.tolist()
    return res


# ── seed registry ────────────────────────────────────────────────────────────

# 10 seeds: original 5 (42,7,123,0,99) + 5 new (1, 13, 25, 50, 77).
SEEDS_K10 = [42, 7, 123, 0, 99, 1, 13, 25, 50, 77]
SEEDS_K5  = [42, 7, 123, 0, 99]
_TAG_FOR  = {s: ("BASELINE_FULL" if s == 42 else f"BASELINE_FULL_seed{s}")
              for s in SEEDS_K10}


def main():
    t0 = time.perf_counter()
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_rl_period("btc")

    print(f"\n{'='*120}\n  VOTING ENSEMBLE — K=5 + K=10 across multiple aggregation modes\n{'='*120}")

    # check policies exist
    missing = [s for s in SEEDS_K10 if not (CACHE / "policies" / f"btc_dqn_policy_{_TAG_FOR[s]}.pt").exists()]
    if missing:
        print(f"  ! missing policy files for seeds {missing} — skipping K=10 variants")
        K10 = []
    else:
        K10 = SEEDS_K10

    variants = []
    for K_label, seeds in [("K5", SEEDS_K5)] + ([("K10", K10)] if K10 else []):
        if not seeds:
            continue
        K = len(seeds)
        variants.append(("q_avg",     "q_avg",     None, seeds, K_label))
        variants.append(("plurality", "plurality", None, seeds, K_label))
        # majority threshold: 60% of K (rounded up)
        variants.append((f"maj{int(0.6*K + 0.5)}",  "majority_t", int(0.6*K + 0.5), seeds, K_label))
        # strong consensus: 80% of K
        variants.append((f"strong{int(0.8*K)}",     "majority_t", int(0.8*K), seeds, K_label))
        variants.append(("unanimous", "unanimous", None, seeds, K_label))

    results = []
    for short_name, mode, thr, seeds, K_label in variants:
        name = f"{K_label}_{short_name}"
        track = (K_label == "K10" and mode in ("plurality", "majority_t"))
        print(f"\n  Running {name} (K={len(seeds)}, mode={mode}, threshold={thr}) ...")
        res = run_variant(name, mode, thr, seeds, atr_median, full,
                          track_agreement=track)
        results.append(res)

    # print summary
    print(f"\n\n{'name':<22} {'val':>8} {'test':>8} {'val eq':>8} {'test eq':>8} "
          f"{'WF mean':>10} {'WF pos':>7}  per-fold WF + (trades total)")
    print("-"*150)
    for r in results:
        wf_str = [round(s, 2) for s in r["wf_per_fold"]]
        n_trades_total = sum(r["wf_trades_per_fold"])
        print(f"{r['name']:<22} {r['val_sharpe']:>+8.2f} {r['test_sharpe']:>+8.2f} "
              f"{r['val_eq']:>8.3f} {r['test_eq']:>8.3f} "
              f"{r['wf_mean']:>+10.3f} {r['wf_pos']:>3}/6  {wf_str}  ({n_trades_total} trades)")

    # agreement histograms (K=10 only)
    print(f"\n  ── Agreement histograms (K=10) ──  bars where top-action got N votes (from val):")
    for r in results:
        if r["name"].startswith("K10_") and "val_agreement_hist" in r:
            hist = r["val_agreement_hist"]
            n_trades = r["val_n_trade_by_agree"]
            print(f"\n  {r['name']}:")
            print(f"    N votes : " + " ".join(f"{i:>5}" for i in range(11)))
            print(f"    bars    : " + " ".join(f"{c:>5}" for c in hist))
            print(f"    trades  : " + " ".join(f"{c:>5}" for c in n_trades))
            # show as %
            total = sum(hist)
            if total:
                pct = [c/total*100 for c in hist]
                print(f"    % bars  : " + " ".join(f"{p:>4.1f}%" for p in pct))

    # baselines: include best single seed (42) & K=5 q_avg from prior run for context
    print(f"\n  Reference single-seed (BASELINE_FULL = seed=42):")
    print(f"    val=+7.295  test=+3.666  WF=+9.034 (6/6)  per-fold [13.03, 14.82, 6.29, 9.56, 8.17, 2.33]")

    out = CACHE / "results" / "voting_ensemble_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out.name}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
