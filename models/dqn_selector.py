"""
DQN strategy selector — Phase 3.3 training loop.

Continuous on-policy refresh pattern:
  warmup    : 5,000 random transitions to seed the buffer
  main loop : every REFRESH_EVERY grad steps, push REFRESH_M new transitions
              from the current ε-greedy policy
  validate  : every VAL_EVERY grad steps, run greedy policy on DQN-val,
              compute Sharpe from the resulting trade equity curve
  save best : whenever val Sharpe improves, save online_net to disk
  early stop: terminate after EARLY_STOP_PATIENCE steps with no improvement

Loss: n-step Huber on Bellman target with PER importance-sampling weights.

Run: python3 -m models.dqn_selector [ticker]
"""

import sys, time, json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.dqn_network    import DQN, masked_argmax, masked_max
from models.dqn_replay     import ReplayBuffer
from models.dqn_rollout    import rollout_chunk, _build_exit_arrays, STRAT_KEYS
from backtest.single_trade import simulate_one_trade

CACHE = ROOT / "cache"

# ── hyperparameters ──────────────────────────────────────────────────────────
GAMMA              = 0.99
LR                 = 1e-3
BATCH_SIZE         = 128
BUFFER_SIZE        = 80_000
WARMUP_STEPS       = 5_000          # random-policy warmup
TOTAL_GRAD_STEPS   = 200_000
REFRESH_EVERY      = 500            # grad steps between rollout refreshes
REFRESH_M          = 2_000          # new transitions per refresh
TARGET_SYNC_EVERY  = 1_000          # hard copy online → target
VAL_EVERY          = 5_000          # eval on DQN-val
EARLY_STOP_PATIENCE= 25_000         # grad steps with no val improvement
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS    = 80_000
PER_ALPHA          = 0.6
PER_BETA_START, PER_BETA_END = 0.4, 1.0
HUBER_DELTA        = 1.0
GRAD_CLIP          = 10.0

# ── Path A: amplification + stratified PER ──────────────────────────────────
REWARD_SCALE       = 100.0          # multiplier for buffer reward (raw used for eval)
USE_STRATIFIED_PER = True           # 50/50 NO_TRADE vs trade per batch

# ── Path C: action-space restriction (binary diagnostic) ─────────────────────
# Set to "binary" for {NO_TRADE, S1_VolDir} only; "all" for full 9-strategy.
ACTION_MODE        = "all"          # overridden by CLI


# ── ε schedule + β schedule ──────────────────────────────────────────────────

def epsilon(step: int) -> float:
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac


def per_beta(step: int) -> float:
    frac = min(1.0, step / TOTAL_GRAD_STEPS)
    return PER_BETA_START + (PER_BETA_END - PER_BETA_START) * frac


# ── policy helpers ───────────────────────────────────────────────────────────

class _UniformRandom:
    def __init__(self, rng): self.rng = rng
    def __call__(self, s, v):
        return int(self.rng.choice(np.where(v)[0]))


class _EpsilonGreedy:
    def __init__(self, net: DQN, rng, epsilon_fn, step_ref: dict):
        self.net = net; self.rng = rng
        self.epsilon_fn = epsilon_fn
        self.step_ref = step_ref          # mutable {"step": int}
    def __call__(self, s, v):
        eps = self.epsilon_fn(self.step_ref["step"])
        if self.rng.random() < eps:
            return int(self.rng.choice(np.where(v)[0]))
        # greedy via masked argmax
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool().unsqueeze(0)
            return int(masked_argmax(self.net, sb, vb).item())


class _Greedy:
    """Pure deterministic greedy over masked Q. Used for validation."""
    def __init__(self, net: DQN): self.net = net
    def __call__(self, s, v):
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            vb = torch.from_numpy(v).bool().unsqueeze(0)
            return int(masked_argmax(self.net, sb, vb).item())


# ── Bellman target + Huber loss ──────────────────────────────────────────────

def bellman_loss(online, target, batch: dict, gamma: float,
                  is_w: torch.Tensor, double: bool = False):
    """target = r + γ^duration · max_a' Q_target(s', a') · (1 - done)
    Loss   = Huber(Q_online(s, a), target) weighted by IS weights.

    If double=True (Double DQN): online net picks argmax action, target net
    evaluates that action's Q-value. Reduces overestimation bias.

    Returns (loss, td_errors_detached_numpy)."""
    s         = torch.from_numpy(batch["state"]).float()
    a         = torch.from_numpy(batch["action"].astype(np.int64))
    r         = torch.from_numpy(batch["reward"]).float()
    dur       = torch.from_numpy(batch["duration"].astype(np.int64))
    s_next    = torch.from_numpy(batch["state_next"]).float()
    v_next    = torch.from_numpy(batch["valid_next"]).bool()
    done      = torch.from_numpy(batch["done"]).bool()

    q_sa = online(s).gather(1, a.unsqueeze(1)).squeeze(1)         # (B,)

    with torch.no_grad():
        if double:
            # Double DQN: online net picks argmax, target net evaluates
            q_online_next = online(s_next).masked_fill(~v_next, -1e9)
            a_next        = q_online_next.argmax(dim=1, keepdim=True)
            q_next_max    = target(s_next).gather(1, a_next).squeeze(1)
        else:
            q_next_max = masked_max(target, s_next, v_next)        # (B,)
        bootstrap  = (gamma ** dur.float()) * q_next_max
        bootstrap[done] = 0.0
        tgt        = r + bootstrap

    td_err = q_sa - tgt
    loss = (is_w * F.huber_loss(q_sa, tgt, reduction="none", delta=HUBER_DELTA)).mean()
    return loss, td_err.detach().cpu().numpy()


# ── validation evaluator ─────────────────────────────────────────────────────

def evaluate_policy(net: DQN, state, valid, signals, prices,
                     tp, sl, trail, tab, be, ts_bars,
                     valid_mask_override: np.ndarray = None,
                     fee: float = None):
    """Run greedy policy through `state` length, return per-trade PnL list,
    final equity, Sharpe (annualized for 1-min bars).

    `fee`: per-side trading fee. If None, uses backtest.costs.TAKER_FEE."""
    from backtest.costs import TAKER_FEE
    from models.diagnostics_ab import _simulate_one_trade_fee
    if fee is None:
        fee = TAKER_FEE
    n_bars   = len(state)
    equity   = 1.0; peak = 1.0; last_pnl = 0.0
    eq_arr   = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []; trade_dirs = []; trade_strats = []
    n_actions = np.zeros(valid.shape[1], dtype=np.int64)
    n_trades = 0
    n_steps  = 0

    t = 0
    policy = _Greedy(net)
    while t < n_bars - 2:
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t] if valid_mask_override is None else (valid[t] & valid_mask_override)
        if not valid_t.any():
            valid_t = valid[t].copy()
            valid_t[1:] = False
            valid_t[0]  = True
        action  = policy(s_t, valid_t)
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
                # apply equity at exit bar
                t_close = t + 1 + n_held
                if t_close >= n_bars: t_close = n_bars - 1
                eq_arr[t:t_close + 1] = equity
                equity   *= (1.0 + float(pnl))
                eq_arr[t_close + 1:] = equity                       # extend tail
                if t_close == n_bars - 1:
                    eq_arr[-1] = equity                             # ensure last bar reflects post-exit
                peak      = max(peak, equity)
                last_pnl  = float(pnl)
                trade_pnls.append(float(pnl))
                trade_dirs.append(int(direction))
                trade_strats.append(int(k))
                n_trades += 1
                t_next = t_close + 1
        t = t_next

    # Sharpe from per-bar equity returns (annualized for 1-min bars)
    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    win_rate = (np.array(trade_pnls) > 0).mean() if trade_pnls else 0.0
    total_ret = equity - 1.0

    return dict(
        sharpe       = float(sharpe),
        equity_final = float(equity),
        equity_peak  = float(peak),
        max_dd       = float((np.minimum.accumulate(eq_arr / np.maximum.accumulate(eq_arr)) - 1).min()),
        n_trades     = int(n_trades),
        n_steps      = int(n_steps),
        win_rate     = float(win_rate),
        total_return = float(total_ret),
        trade_pnls   = trade_pnls,
        trade_dirs   = trade_dirs,
        trade_strats = trade_strats,
        action_counts= n_actions,
        eq_curve     = eq_arr,
    )


# ── main training ────────────────────────────────────────────────────────────

def train(ticker: str = "btc", seed: int = 42, action_mode: str = "all",
           tag: str = "v1", fee: float = None, trade_penalty: float = 0.0,
           ablate_actions: list = None, state_version: str = "v5",
           hidden: int = 64, algo: str = "dqn"):
    torch.manual_seed(seed); np.random.seed(seed)
    rng_warm = np.random.default_rng(seed)
    rng_eps  = np.random.default_rng(seed + 1)

    from backtest.costs import TAKER_FEE
    if fee is None:
        fee = TAKER_FEE

    t_start = time.perf_counter()
    print(f"\n{'='*70}\n  PHASE 3.3 — DQN TRAINING  ({ticker.upper()})  "
          f"action_mode={action_mode}  tag={tag}\n"
          f"  reward_scale={REWARD_SCALE}  stratified_PER={USE_STRATIFIED_PER}  "
          f"fee={fee:.4f}  trade_penalty={trade_penalty:.4f}\n"
          f"{'='*70}")

    # ── action-mask override for Path C binary diagnostic ────────────────────
    valid_override = None
    if action_mode == "binary":
        # allow NO_TRADE (idx 0) + S1_VolDir (idx 1) only
        valid_override = np.zeros(10, dtype=np.bool_)
        valid_override[0] = True
        valid_override[1] = True
        print(f"  Path C: action mask restricted to {{NO_TRADE, S1_VolDir}}")
    elif ablate_actions:
        # ablate selected strategy actions; NO_TRADE always preserved
        valid_override = np.ones(10, dtype=np.bool_)
        for idx in ablate_actions:
            if 1 <= idx <= 9:
                valid_override[idx] = False
        valid_override[0] = True
        from models.dqn_rollout import STRAT_KEYS as _SK
        names = [_SK[i-1] for i in ablate_actions if 1 <= i <= 9]
        print(f"  ABLATION: actions {ablate_actions} → strategies {names} masked during training+val")

    # ── load DQN-train and DQN-val arrays ────────────────────────────────────
    suffix = "" if state_version == "v5" else f"_{state_version}"
    sp_tr = np.load(CACHE / f"{ticker}_dqn_state_train{suffix}.npz")
    sp_v  = np.load(CACHE / f"{ticker}_dqn_state_val{suffix}.npz")
    state_dim = int(sp_tr["state"].shape[1])
    print(f"  state_version={state_version}  state_dim={state_dim}")
    print(f"  DQN-train: state {sp_tr['state'].shape}")
    print(f"  DQN-val  : state {sp_v['state'].shape}")

    # vol median for ATR-scaled exits
    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    print(f"  atr_train_median = {atr_median:.4f}")

    # exit arrays (computed once per split)
    print("  building exit arrays for train + val ...")
    tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts = _build_exit_arrays(
        sp_tr["price"], sp_tr["atr"], atr_median)
    v_tp, v_sl, v_tr, v_tab, v_be, v_ts = _build_exit_arrays(
        sp_v["price"], sp_v["atr"], atr_median)

    # ── networks + optimizer + buffer ────────────────────────────────────────
    use_dueling = algo in ("dueling", "double_dueling")
    use_double  = algo in ("double",  "double_dueling")
    n_actions   = int(sp_tr["valid_actions"].shape[1])    # dynamic — supports v8 (12 actions)
    print(f"  hidden_size={hidden}  algo={algo}  n_actions={n_actions}  "
          f"(dueling={use_dueling}, double={use_double})")
    if use_dueling:
        from models.dqn_network import DuelingDQN
        online = DuelingDQN(state_dim=state_dim, n_actions=n_actions, hidden=hidden)
        target = DuelingDQN(state_dim=state_dim, n_actions=n_actions, hidden=hidden)
    else:
        online = DQN(state_dim=state_dim, n_actions=n_actions, hidden=hidden)
        target = DQN(state_dim=state_dim, n_actions=n_actions, hidden=hidden)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    buf = ReplayBuffer(capacity=BUFFER_SIZE, state_dim=state_dim, n_actions=n_actions)

    print(f"  online net params: {online.n_params():,}  optimizer Adam lr={LR}")

    # ── warmup with random policy ────────────────────────────────────────────
    print(f"\n  Warmup: {WARMUP_STEPS:,} random transitions ...")
    cursor_tr = dict(t=0, equity=1.0, peak=1.0, last_pnl=0.0)
    rand_pol = _UniformRandom(rng_warm)
    t_warm = time.perf_counter()
    while len(buf) < WARMUP_STEPS:
        cursor_tr = rollout_chunk(
            sp_tr["state"], sp_tr["valid_actions"], sp_tr["signals"], sp_tr["price"],
            tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
            policy_fn=rand_pol, buffer=buf, cursor=cursor_tr,
            max_transitions=min(2000, WARMUP_STEPS - len(buf)),
            reward_scale=REWARD_SCALE, valid_mask_override=valid_override,
            fee=fee, trade_penalty=trade_penalty,
        )
    n_trade_in_buf = int((buf.action[:len(buf)] != 0).sum())
    print(f"    buffer={len(buf):,} after warmup  ({n_trade_in_buf:,} trades, "
          f"{n_trade_in_buf/len(buf)*100:.1f}%)  [{time.perf_counter()-t_warm:.1f}s]")

    # ── main training loop ───────────────────────────────────────────────────
    step_ref = {"step": 0}
    eps_pol  = _EpsilonGreedy(online, rng_eps, epsilon, step_ref)

    history = []     # list of dicts per validation event
    best_val_sharpe = -np.inf
    best_step       = 0

    print(f"\n  Training: max {TOTAL_GRAD_STEPS:,} grad steps "
          f"(refresh M={REFRESH_M} every {REFRESH_EVERY}, "
          f"target sync every {TARGET_SYNC_EVERY}, val every {VAL_EVERY})")
    print(f"  ε: {EPS_START} → {EPS_END} over {EPS_DECAY_STEPS:,} steps  "
          f"PER β: {PER_BETA_START} → {PER_BETA_END}")
    print()
    t_loop = time.perf_counter()
    losses = []

    for step in range(1, TOTAL_GRAD_STEPS + 1):
        step_ref["step"] = step

        # rollout refresh (skip step 0)
        if step % REFRESH_EVERY == 0:
            cursor_tr = rollout_chunk(
                sp_tr["state"], sp_tr["valid_actions"], sp_tr["signals"], sp_tr["price"],
                tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
                policy_fn=eps_pol, buffer=buf, cursor=cursor_tr,
                max_transitions=REFRESH_M,
                reward_scale=REWARD_SCALE, valid_mask_override=valid_override,
                fee=fee, trade_penalty=trade_penalty,
            )

        # gradient step
        sampler = (buf.sample_stratified_prioritized if USE_STRATIFIED_PER
                   else buf.sample_prioritized)
        batch, idx, is_w_np = sampler(
            BATCH_SIZE, alpha=PER_ALPHA, beta=per_beta(step))
        is_w = torch.from_numpy(is_w_np)
        loss, td_errs = bellman_loss(online, target, batch, GAMMA, is_w, double=use_double)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
        optimizer.step()
        buf.update_priorities(idx, td_errs)
        losses.append(float(loss.item()))

        # target sync
        if step % TARGET_SYNC_EVERY == 0:
            target.load_state_dict(online.state_dict())

        # validation
        if step % VAL_EVERY == 0:
            t_eval = time.perf_counter()
            val = evaluate_policy(
                online, sp_v["state"], sp_v["valid_actions"],
                sp_v["signals"], sp_v["price"],
                v_tp, v_sl, v_tr, v_tab, v_be, v_ts,
                valid_mask_override=valid_override,
                fee=fee,
            )
            improved = val["sharpe"] > best_val_sharpe
            marker = "★" if improved else " "
            mean_loss = float(np.mean(losses[-VAL_EVERY:]))
            elapsed = time.perf_counter() - t_loop
            print(f"  [step {step:>6,}] {marker} ε={epsilon(step):.3f}  "
                  f"loss(avg{VAL_EVERY})={mean_loss:.4f}  "
                  f"val Sharpe={val['sharpe']:>+6.3f}  "
                  f"trades={val['n_trades']:>4}  win%={val['win_rate']*100:>4.1f}  "
                  f"eq={val['equity_final']:.3f}  dd={val['max_dd']*100:>+5.1f}%  "
                  f"[loop {elapsed:>5.0f}s + val {time.perf_counter()-t_eval:.1f}s]")

            history.append(dict(
                step=step, eps=epsilon(step),
                mean_loss=mean_loss,
                val_sharpe=val["sharpe"], val_trades=val["n_trades"],
                val_winrate=val["win_rate"], val_equity=val["equity_final"],
                val_max_dd=val["max_dd"],
                actions=val["action_counts"].tolist(),
            ))

            if improved:
                best_val_sharpe = val["sharpe"]
                best_step       = step
                torch.save(online.state_dict(),
                            CACHE / f"{ticker}_dqn_policy_{tag}.pt")

            # early stopping
            if step - best_step > EARLY_STOP_PATIENCE:
                print(f"\n  Early stop: no val improvement for {EARLY_STOP_PATIENCE:,} steps "
                      f"(best Sharpe={best_val_sharpe:+.3f} at step {best_step:,})")
                break

    elapsed_total = time.perf_counter() - t_start

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  TRAINING SUMMARY\n{'='*70}")
    print(f"  total wall time      : {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  grad steps completed : {step:,}")
    print(f"  best val Sharpe      : {best_val_sharpe:+.3f}  (step {best_step:,})")
    print(f"  policy saved to      : cache/{ticker}_dqn_policy_{tag}.pt")

    # save training history
    hist_path = CACHE / f"{ticker}_dqn_train_history_{tag}.json"
    hist_path.write_text(json.dumps(dict(
        ticker=ticker, run_at=datetime.utcnow().isoformat(),
        config=dict(
            gamma=GAMMA, lr=LR, batch=BATCH_SIZE, buffer=BUFFER_SIZE,
            warmup=WARMUP_STEPS, total_steps=TOTAL_GRAD_STEPS,
            refresh_every=REFRESH_EVERY, refresh_m=REFRESH_M,
            target_sync_every=TARGET_SYNC_EVERY, val_every=VAL_EVERY,
            patience=EARLY_STOP_PATIENCE,
            eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_STEPS,
            per_alpha=PER_ALPHA, per_beta_start=PER_BETA_START, per_beta_end=PER_BETA_END,
            huber_delta=HUBER_DELTA, grad_clip=GRAD_CLIP,
        ),
        best_val_sharpe=best_val_sharpe, best_step=best_step,
        elapsed_seconds=elapsed_total,
        history=history,
    ), indent=2))
    print(f"  history saved to     : {hist_path.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--mode", choices=["all", "binary"], default="all",
                     help="all=9 strategies; binary=Path C diagnostic ({NO_TRADE,S1})")
    ap.add_argument("--tag",  default="A",
                     help="output filename suffix (e.g. policy_A.pt)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee",  type=float, default=None,
                     help="per-side fee (default = TAKER_FEE = 0.0008)")
    ap.add_argument("--trade-penalty", type=float, default=0.0, dest="trade_penalty",
                     help="fixed penalty per trade entry in buffer reward (default 0)")
    ap.add_argument("--ablate-actions", default="", dest="ablate_actions",
                     help="comma-separated action indices [1..9] to mask during training "
                          "(e.g. '5' masks S6_TwoSignal, '5,8' masks S6+S10)")
    ap.add_argument("--state-version", default="v5", dest="state_version",
                     choices=["v5", "v6", "v7_pa", "v7_basis", "v8_s11s13", "v9_basis_s11s13"],
                     help="state-array version: v5=50-dim (default), v6=54-dim with direction probs, "
                          "v7_pa=54-dim with price-action context, v7_basis=55-dim with basis+funding, "
                          "v8_s11s13=52-dim with S11+S13 strategy flags + 12-action space, "
                          "v9_basis_s11s13=57-dim combining v7_basis + v8 (Z2/Z3 Step 5)")
    ap.add_argument("--hidden", type=int, default=64,
                     help="DQN hidden layer size (default 64; larger for capacity test)")
    ap.add_argument("--algo", default="dqn",
                     choices=["dqn", "double", "dueling", "double_dueling"],
                     help="RL algorithm: dqn (default), double, dueling, or both")
    args = ap.parse_args()
    ablate = [int(x) for x in args.ablate_actions.split(",") if x.strip()]
    train(args.ticker, seed=args.seed, action_mode=args.mode, tag=args.tag,
           fee=args.fee, trade_penalty=args.trade_penalty, ablate_actions=ablate,
           state_version=args.state_version, hidden=args.hidden, algo=args.algo)
