"""
Z4 — walk-forward eval of QR-DQN (CVaR action selection) and Transformer
ensembles vs `VOTE5_v8_H256_DD` baseline.

For QR-DQN, evaluation uses CVaR-α action selection (alpha=0.3 default,
the same as training). The eval is also reported at alpha=1.0 (mean-Q,
standard greedy) for diagnostic comparison.
"""
import json, statistics, time
from pathlib import Path
import numpy as np
import torch

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.audit_vote5_dd import run_fold, run_walkforward, vote_action
from models.qr_network import QRDuelingDQN, cvar_action, mean_q_action
from models.transformer_network import TransformerDQN
from config.cache_paths import POLICIES, STATE, PREDS, RESULTS

CACHE = Path("cache")
SEEDS = [42, 7, 123, 0, 99]
N_QUANTILES = 32


# ── policy wrappers that present a uniform `policy(s, v) -> int` API ─────────

class _QRVote:
    """K-net plurality vote using CVaR-α action selection."""
    def __init__(self, nets, alpha):
        self.nets = nets; self.alpha = alpha
    def __call__(self, s, v):
        sb = torch.from_numpy(s).float().unsqueeze(0)
        vb = torch.from_numpy(v).bool().unsqueeze(0)
        votes = []
        for net in self.nets:
            if self.alpha >= 1.0:
                votes.append(int(mean_q_action(net, sb, vb).item()))
            else:
                votes.append(int(cvar_action(net, sb, vb, self.alpha).item()))
        # plurality, tie → 0
        from collections import Counter
        c = Counter(votes).most_common(2)
        if len(c) >= 2 and c[0][1] == c[1][1]:
            return 0
        return c[0][0]


# ── eval pack (reuses audit_vote5_dd.run_fold which expects nets list +
#    Q-style argmax; we instead need our custom policy_fn). So implement a
#    parallel run_fold here. Keep aligned to existing primitives.

from models.diagnostics_ab import _simulate_one_trade_fee


def _run_fold_custom(policy_fn, state, valid, signals, prices, atr,
                       regime_id, ts, tp, sl, trail, tab, be, ts_bars,
                       fee=0.0, fold_id=0):
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trades = []
    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False
        action = policy_fn(s_t, valid_t)
        if action == 0:
            t += 1; continue
        k = action - 1
        direction = int(signals[t, k])
        if direction == 0:
            t += 1; continue
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
        peak = max(peak, equity); last_pnl = float(pnl)
        trades.append(dict(fold=fold_id, t_open=int(t+1), t_close=int(t_close),
                          bars_held=int(n_held), strat_idx=int(k),
                          direction=int(direction), pnl=float(pnl)))
        t = t_close + 1
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    return eq_arr, float(sharpe), float(equity), trades


def load_full_v8():
    arrs = {}
    for split in ("train", "val", "test"):
        sp = np.load(STATE / f"btc_dqn_state_{split}_v8_s11s13.npz")
        for k in ("state","valid_actions","signals","price","atr","ts","regime_id"):
            arrs.setdefault(k, []).append(sp[k])
    return {k: np.concatenate(arrs[k], axis=0) for k in arrs}


def eval_pack(make_policy, label, atr_median, full):
    from models.audit_vote5_dd import RL_START_REL, RL_END_REL, N_FOLDS
    fold_size = (RL_END_REL - RL_START_REL) // N_FOLDS

    sp_v = np.load(STATE / "btc_dqn_state_val_v8_s11s13.npz")
    sp_t = np.load(STATE / "btc_dqn_state_test_v8_s11s13.npz")
    tp_v, sl_v, tr_v, tab_v, be_v, ts_v = _build_exit_arrays(sp_v["price"], sp_v["atr"], atr_median)
    tp_t, sl_t, tr_t, tab_t, be_t, ts_t = _build_exit_arrays(sp_t["price"], sp_t["atr"], atr_median)

    rows = []
    n_trades_total = 0
    for i in range(N_FOLDS):
        a = i * fold_size
        b = (i + 1) * fold_size if i < N_FOLDS - 1 else (RL_END_REL - RL_START_REL)
        sub = {kk: full[kk][a:b] for kk in full}
        tp, sl, tr, tab, be, ts = _build_exit_arrays(sub["price"], sub["atr"], atr_median)
        eq, sh, _, trades = _run_fold_custom(
            make_policy(), sub["state"], sub["valid_actions"], sub["signals"],
            sub["price"], sub["atr"], sub["regime_id"], sub["ts"],
            tp, sl, tr, tab, be, ts, fee=0.0, fold_id=i+1)
        rows.append(sh); n_trades_total += len(trades)
    wf = statistics.mean(rows); pos = sum(1 for r in rows if r > 0)

    _, vsh, _, vtr = _run_fold_custom(
        make_policy(), sp_v["state"], sp_v["valid_actions"], sp_v["signals"],
        sp_v["price"], sp_v["atr"], sp_v["regime_id"], sp_v["ts"],
        tp_v, sl_v, tr_v, tab_v, be_v, ts_v, fee=0.0)
    _, tsh, _, ttr = _run_fold_custom(
        make_policy(), sp_t["state"], sp_t["valid_actions"], sp_t["signals"],
        sp_t["price"], sp_t["atr"], sp_t["regime_id"], sp_t["ts"],
        tp_t, sl_t, tr_t, tab_t, be_t, ts_t, fee=0.0)

    print(f"  {label:<40}  WF {wf:>+7.3f}  val {vsh:>+7.3f}  test {tsh:>+7.3f}  "
          f"folds+ {pos}/6  trades(WF/val/test) {n_trades_total:>4}/{len(vtr):>4}/{len(ttr):>4}")
    return dict(label=label, wf=wf, val=vsh, test=tsh, folds_pos=pos,
                trades_wf=n_trades_total, trades_val=len(vtr), trades_test=len(ttr),
                per_fold=rows)


def load_qr_net(seed):
    n = QRDuelingDQN(52, 12, 256, N_QUANTILES)
    n.load_state_dict(torch.load(POLICIES / f"btc_dqn_policy_QRDQN_v8_seed{seed}.pt",
                                   map_location="cpu"))
    n.eval(); return n


def load_xfmr_net(seed):
    n = TransformerDQN(52, 12, d_model=8, n_heads=2, n_layers=1, hidden=128)
    n.load_state_dict(torch.load(POLICIES / f"btc_dqn_policy_XFMR_v8_seed{seed}.pt",
                                   map_location="cpu"))
    n.eval(); return n


def load_dueling_v8(seed):
    n = DuelingDQN(52, 12, 256)
    n.load_state_dict(torch.load(POLICIES / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{seed}.pt",
                                   map_location="cpu"))
    n.eval(); return n


def main():
    t0 = time.perf_counter()
    vol = np.load(PREDS / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    full = load_full_v8()

    print(f"\n{'='*120}\n  Z4 — QR-DQN + Transformer eval vs VOTE5_v8_H256_DD\n{'='*120}\n")
    results = []

    # ── QR-DQN ───────────────────────────────────────────────────────
    try:
        qr_nets = [load_qr_net(s) for s in SEEDS]
        # CVaR-0.3 vote
        results.append(eval_pack(
            lambda: _QRVote(qr_nets, alpha=0.3),
            "QRDQN_v8 VOTE5 CVaR=0.3", atr_median, full))
        # CVaR-1.0 (mean-Q) vote — diagnostic
        results.append(eval_pack(
            lambda: _QRVote(qr_nets, alpha=1.0),
            "QRDQN_v8 VOTE5 mean-Q",   atr_median, full))
        # single best seed at CVaR-0.3
        for s in SEEDS:
            results.append(eval_pack(
                lambda s=s: _QRVote([load_qr_net(s)], alpha=0.3),
                f"QRDQN_v8 single s={s} CVaR=0.3", atr_median, full))
    except FileNotFoundError as e:
        print(f"  QR-DQN policies not all trained yet: {e}")

    # ── Transformer ──────────────────────────────────────────────────
    try:
        xfmr_nets = [load_xfmr_net(s) for s in SEEDS]
        # Standard vote (mean-Q argmax with plurality)
        from models.voting_ensemble import _VotePolicy
        results.append(eval_pack(
            lambda: _VotePolicy(xfmr_nets, mode="plurality"),
            "XFMR_v8 VOTE5 plurality", atr_median, full))
        for s in SEEDS:
            results.append(eval_pack(
                lambda s=s: _VotePolicy([load_xfmr_net(s)], mode="plurality"),
                f"XFMR_v8 single s={s}", atr_median, full))
    except FileNotFoundError as e:
        print(f"  Transformer policies not all trained yet: {e}")

    # ── Baseline parity ──────────────────────────────────────────────
    from models.voting_ensemble import _VotePolicy
    dd_nets = [load_dueling_v8(s) for s in SEEDS]
    results.append(eval_pack(
        lambda: _VotePolicy(dd_nets, mode="plurality"),
        "BASELINE VOTE5_v8_H256_DD", atr_median, full))

    out = RESULTS / "z4_qrdqn_xfmr_eval.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  → {out}    [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
