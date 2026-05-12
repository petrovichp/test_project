"""
Z4.4 — QR-DQN training loop (CVaR-aware).

Mirrors models/dqn_selector.py training pattern but with quantile-regression
loss and optional CVaR action selection at inference.

Best checkpoint by CVaR-α val Sharpe (or mean-Q if alpha=1.0).

Run:
  python3 -m models.qr_dqn_train btc --tag QRDQN_v8 --seed 42 --hidden 256 \
        --state-version v8_s11s13 --alpha 0.3 --trade-penalty 0.001
"""
import argparse, sys, time, json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.qr_network import QRDuelingDQN, quantile_huber_loss, cvar_action, mean_q_action
from models.dqn_replay  import ReplayBuffer
from models.dqn_rollout import rollout_chunk, _build_exit_arrays
from config.cache_paths import POLICIES, STATE, PREDS

CACHE = ROOT / "cache"

# Hyperparameters (mirror dqn_selector)
GAMMA              = 0.99
LR                 = 1e-3
BATCH_SIZE         = 128
BUFFER_SIZE        = 80_000
WARMUP_STEPS       = 5_000
TOTAL_GRAD_STEPS   = 200_000
REFRESH_EVERY      = 500
REFRESH_M          = 2_000
TARGET_SYNC_EVERY  = 1_000
VAL_EVERY          = 5_000
EARLY_STOP_PATIENCE= 25_000
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS    = 80_000
GRAD_CLIP          = 10.0
REWARD_SCALE       = 100.0
USE_STRATIFIED_PER = True
N_QUANTILES        = 32


def epsilon(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac


class _UniformRandom:
    def __init__(self, rng): self.rng = rng
    def __call__(self, s, v): return int(self.rng.choice(np.where(v)[0]))


class _EpsilonGreedy:
    """ε-greedy over mean-Q (training-time exploration; CVaR only used at eval)."""
    def __init__(self, net, rng, step_ref):
        self.net = net; self.rng = rng; self.step_ref = step_ref
    def __call__(self, s, v):
        eps = epsilon(self.step_ref["step"])
        if self.rng.random() < eps:
            return int(self.rng.choice(np.where(v)[0]))
        sb = torch.from_numpy(s).float().unsqueeze(0)
        vb = torch.from_numpy(v).bool().unsqueeze(0)
        return int(mean_q_action(self.net, sb, vb).item())


class _GreedyCVaR:
    """Eval policy: CVaR-α greedy."""
    def __init__(self, net, alpha):
        self.net = net; self.alpha = alpha
    def __call__(self, s, v):
        sb = torch.from_numpy(s).float().unsqueeze(0)
        vb = torch.from_numpy(v).bool().unsqueeze(0)
        if self.alpha >= 1.0:
            return int(mean_q_action(self.net, sb, vb).item())
        return int(cvar_action(self.net, sb, vb, self.alpha).item())


def qr_bellman_loss(online, target, batch, tau, gamma, double=True):
    s         = torch.from_numpy(batch["state"]).float()
    a         = torch.from_numpy(batch["action"].astype(np.int64))
    r         = torch.from_numpy(batch["reward"]).float()
    dur       = torch.from_numpy(batch["duration"].astype(np.int64))
    s_next    = torch.from_numpy(batch["state_next"]).float()
    v_next    = torch.from_numpy(batch["valid_next"]).bool()
    done      = torch.from_numpy(batch["done"]).bool()

    # Online quantiles for taken action a
    q_online = online(s)                                          # (B, A, Q)
    theta    = q_online.gather(
        1, a.view(-1, 1, 1).expand(-1, 1, online.n_quantiles)
    ).squeeze(1)                                                   # (B, Q)

    with torch.no_grad():
        # Double-DQN style: online picks action by mean-Q on s_next; target evaluates that action's quantile dist
        q_online_next = online(s_next).mean(dim=-1)               # (B, A)
        q_online_next = q_online_next.masked_fill(~v_next, -1e9)
        a_next = q_online_next.argmax(dim=1, keepdim=True)        # (B, 1)
        q_target_next = target(s_next)                            # (B, A, Q)
        theta_next = q_target_next.gather(
            1, a_next.unsqueeze(2).expand(-1, 1, online.n_quantiles)
        ).squeeze(1)                                               # (B, Q)
        gamma_dur = (gamma ** dur.float()).unsqueeze(1)           # (B, 1)
        target_theta = r.unsqueeze(1) + gamma_dur * theta_next
        # zero-out next-state contribution where done
        target_theta = torch.where(done.unsqueeze(1),
                                    r.unsqueeze(1).expand(-1, online.n_quantiles),
                                    target_theta)

    return quantile_huber_loss(theta, target_theta, tau)


def evaluate_qr_policy(net, alpha, state, valid, signals, prices,
                        tp, sl, trail, tab, be, ts_bars, fee=0.0):
    from models.diagnostics_ab import _simulate_one_trade_fee
    n_bars = len(state)
    equity = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    n_trades = 0
    policy = _GreedyCVaR(net, alpha)
    t = 0
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t].copy()
        if not valid_t.any():
            valid_t[0] = True; valid_t[1:] = False
        action = policy(s_t, valid_t)
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
        n_trades += 1
        t = t_close + 1
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
               if rets.std() > 1e-12 else 0.0)
    return dict(sharpe=float(sharpe), equity_final=float(equity), n_trades=int(n_trades))


def train(ticker="btc", seed=42, tag="QRDQN_v8", hidden=256, alpha=0.3,
           trade_penalty=0.001, state_version="v8_s11s13"):
    torch.manual_seed(seed); np.random.seed(seed)
    rng_warm = np.random.default_rng(seed)
    rng_eps  = np.random.default_rng(seed + 1)
    t_start = time.perf_counter()

    print(f"\n{'='*70}\n  QR-DQN — {tag}  seed={seed}  alpha(CVaR)={alpha}\n"
          f"  hidden={hidden}  n_quantiles={N_QUANTILES}  state={state_version}\n{'='*70}")

    suffix = "" if state_version == "v5" else f"_{state_version}"
    sp_tr = np.load(STATE / f"{ticker}_dqn_state_train{suffix}.npz")
    sp_v  = np.load(STATE / f"{ticker}_dqn_state_val{suffix}.npz")
    state_dim = int(sp_tr["state"].shape[1])
    n_actions = int(sp_tr["valid_actions"].shape[1])

    vol = np.load(PREDS / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts = _build_exit_arrays(
        sp_tr["price"], sp_tr["atr"], atr_median)
    v_tp, v_sl, v_tr_, v_tab, v_be, v_ts = _build_exit_arrays(
        sp_v["price"], sp_v["atr"], atr_median)

    online = QRDuelingDQN(state_dim, n_actions, hidden, N_QUANTILES)
    target = QRDuelingDQN(state_dim, n_actions, hidden, N_QUANTILES)
    target.load_state_dict(online.state_dict()); target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    buf = ReplayBuffer(capacity=BUFFER_SIZE, state_dim=state_dim, n_actions=n_actions)
    tau = (torch.arange(N_QUANTILES, dtype=torch.float32) + 0.5) / N_QUANTILES

    print(f"  state_dim={state_dim}  n_actions={n_actions}  params={online.n_params():,}")

    # Warmup
    print(f"  warmup: {WARMUP_STEPS} random transitions ...")
    cursor_tr = dict(t=0, equity=1.0, peak=1.0, last_pnl=0.0)
    rand_pol = _UniformRandom(rng_warm)
    while len(buf) < WARMUP_STEPS:
        cursor_tr = rollout_chunk(
            sp_tr["state"], sp_tr["valid_actions"], sp_tr["signals"], sp_tr["price"],
            tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
            policy_fn=rand_pol, buffer=buf, cursor=cursor_tr,
            max_transitions=min(2000, WARMUP_STEPS - len(buf)),
            reward_scale=REWARD_SCALE, valid_mask_override=None,
            fee=0.0, trade_penalty=trade_penalty,
        )

    step_ref = {"step": 0}
    eps_pol = _EpsilonGreedy(online, rng_eps, step_ref)
    best_val_sharpe = -np.inf; best_step = 0
    history = []
    losses = []

    print(f"  training: max {TOTAL_GRAD_STEPS:,} grad steps, val every {VAL_EVERY}")
    t_loop = time.perf_counter()
    for step in range(1, TOTAL_GRAD_STEPS + 1):
        step_ref["step"] = step

        if step % REFRESH_EVERY == 0:
            cursor_tr = rollout_chunk(
                sp_tr["state"], sp_tr["valid_actions"], sp_tr["signals"], sp_tr["price"],
                tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
                policy_fn=eps_pol, buffer=buf, cursor=cursor_tr,
                max_transitions=REFRESH_M,
                reward_scale=REWARD_SCALE, valid_mask_override=None,
                fee=0.0, trade_penalty=trade_penalty,
            )

        sampler = buf.sample_stratified_prioritized if USE_STRATIFIED_PER else buf.sample_prioritized
        batch, idx, is_w_np = sampler(BATCH_SIZE, alpha=0.6, beta=0.6)
        loss = qr_bellman_loss(online, target, batch, tau, GAMMA)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
        optimizer.step()
        # PER priorities: use mean-Q TD-error magnitude as proxy
        with torch.no_grad():
            td_proxy = torch.full((BATCH_SIZE,), float(loss.item()), dtype=torch.float32).numpy()
        buf.update_priorities(idx, td_proxy)
        losses.append(float(loss.item()))

        if step % TARGET_SYNC_EVERY == 0:
            target.load_state_dict(online.state_dict())

        if step % VAL_EVERY == 0:
            val = evaluate_qr_policy(
                online, alpha,
                sp_v["state"], sp_v["valid_actions"], sp_v["signals"], sp_v["price"],
                v_tp, v_sl, v_tr_, v_tab, v_be, v_ts, fee=0.0,
            )
            improved = val["sharpe"] > best_val_sharpe
            marker = "★" if improved else " "
            elapsed = time.perf_counter() - t_loop
            print(f"  [step {step:>6,}] {marker} ε={epsilon(step):.3f}  "
                  f"loss(avg{VAL_EVERY})={np.mean(losses[-VAL_EVERY:]):.4f}  "
                  f"val Sharpe={val['sharpe']:+6.3f}  trades={val['n_trades']:>4}  "
                  f"eq={val['equity_final']:.3f}  [{elapsed:>4.0f}s]")
            history.append(dict(step=step, val_sharpe=val["sharpe"],
                                  val_trades=val["n_trades"], val_equity=val["equity_final"]))
            if improved:
                best_val_sharpe = val["sharpe"]; best_step = step
                torch.save(online.state_dict(),
                            POLICIES / f"{ticker}_dqn_policy_{tag}_seed{seed}.pt")
            if step - best_step > EARLY_STOP_PATIENCE:
                print(f"\n  Early stop at step {step:,}  best={best_val_sharpe:+.3f} at {best_step:,}")
                break

    elapsed = time.perf_counter() - t_start
    print(f"\n  TRAINING SUMMARY  wall {elapsed:.0f}s  best val Sharpe={best_val_sharpe:+.3f}")
    print(f"  policy → cache/policies/{ticker}_dqn_policy_{tag}_seed{seed}.pt")

    hist_path = POLICIES / f"{ticker}_dqn_train_history_{tag}_seed{seed}.json"
    hist_path.write_text(json.dumps(dict(
        tag=tag, seed=seed, alpha=alpha, n_quantiles=N_QUANTILES,
        run_at=datetime.utcnow().isoformat(),
        best_val_sharpe=best_val_sharpe, best_step=best_step,
        elapsed_seconds=elapsed, history=history,
    ), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--tag", default="QRDQN_v8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.3,
                     help="CVaR alpha for action selection (1.0 = mean-Q greedy)")
    ap.add_argument("--trade-penalty", type=float, default=0.001, dest="trade_penalty")
    ap.add_argument("--state-version", default="v8_s11s13", dest="state_version")
    args = ap.parse_args()
    train(args.ticker, args.seed, args.tag, args.hidden, args.alpha,
           args.trade_penalty, args.state_version)
