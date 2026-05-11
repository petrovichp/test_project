"""
Exit-timing DQN (Group B).

RL-augmented exit: each trade still has its strategy's rule-based exits
(TP / SL / BE / trail / time-stop) active; the DQN can additionally
trigger EXIT_NOW at any in-trade bar. Whichever fires first ends the trade.

State (28 dims, in-trade only):
   0  unrealized_pnl_pct           clip ±10
   1  n_bars_in_trade / 240        normalized
   2  sl_distance_pct              clip [0, 5]   (pct points from price to current SL)
   3  entry_direction              ∈ {-1, +1}
   4-9   log_return       × 6 lags  [60,30,15,5,1,0]   (sliced from base_state[20:26])
  10-15  taker_net_60_z   × 6 lags                      (sliced from base_state[26:32])
  16-21  ofi_perp_10      × 6 lags                      (sliced from base_state[32:38])
  22-27  vwap_dev_240     × 6 lags                      (sliced from base_state[38:44])

Actions: 0 = HOLD, 1 = EXIT_NOW
Reward : 0 on HOLD, realized trade PnL at terminal bar (rule-fired or RL-fired).
γ      : 0.99 within trade.

Entries (B1-B3): sequential, first-firing strategy at each bar. ~30k entries
across DQN-train, ~30 bars per trade.

Run:
  python3 -m models.exit_dqn [ticker] --tag B1 --fee 0.0008
  python3 -m models.exit_dqn [ticker] --tag B4_S1 --fee 0.0004 --strat-filter 0
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

from models.dqn_replay  import ReplayBuffer
from models.dqn_rollout import _build_exit_arrays, STRAT_KEYS

CACHE = ROOT / "cache"

# ── hyperparameters ──────────────────────────────────────────────────────────
EXIT_STATE_DIM     = 28
N_ACTIONS          = 2                   # HOLD, EXIT_NOW
MAX_TRADE_BARS     = 240                 # hard cap on in-trade horizon

GAMMA              = 1.0          # within trade: undiscounted (sparse terminal reward, bounded ≤240 bars)
LR                 = 1e-3
BATCH_SIZE         = 128
BUFFER_SIZE        = 80_000
WARMUP_STEPS       = 5_000
TOTAL_GRAD_STEPS   = 80_000              # smaller than entry: simpler problem
REFRESH_EVERY      = 500
REFRESH_M          = 2_000
TARGET_SYNC_EVERY  = 1_000
VAL_EVERY          = 4_000
EARLY_STOP_PATIENCE= 16_000
EPS_START, EPS_END = 1.0, 0.05
EPS_DECAY_STEPS    = 40_000
PER_ALPHA          = 0.6
PER_BETA_START, PER_BETA_END = 0.4, 1.0
HUBER_DELTA        = 1.0
GRAD_CLIP          = 10.0
REWARD_SCALE       = 100.0


# ── network ──────────────────────────────────────────────────────────────────

class ExitDQN(nn.Module):
    def __init__(self, state_dim: int = EXIT_STATE_DIM, n_actions: int = N_ACTIONS,
                  hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_actions)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── state builder ────────────────────────────────────────────────────────────

def build_exit_state(base_state_t: np.ndarray, unreal_pnl: float,
                       n_bars_in: int, sl_dist: float, direction: int,
                       out: np.ndarray = None) -> np.ndarray:
    """28-dim exit state. base_state_t = (50,) cached DQN state at current bar."""
    if out is None:
        out = np.zeros(EXIT_STATE_DIM, dtype=np.float32)
    out[0] = max(-10.0, min(10.0, unreal_pnl * 100.0))
    out[1] = n_bars_in / MAX_TRADE_BARS
    out[2] = max(0.0, min(5.0, sl_dist * 100.0))
    out[3] = float(direction)
    # windowed feature lags from cached base state (already standardized)
    out[4:10]  = base_state_t[20:26]   # log_return × 6
    out[10:16] = base_state_t[26:32]   # taker_net_60_z × 6
    out[16:22] = base_state_t[32:38]   # ofi_perp_10 × 6
    out[22:28] = base_state_t[38:44]   # vwap_dev_240 × 6
    return out


# ── per-trade simulator with policy queries ──────────────────────────────────

def simulate_trade_episode(
    state, prices, entry_bar: int, direction: int,
    tp_pct: float, sl_pct: float, trail_pct: float, tab_pct: float,
    breakeven_pct: float, time_stop_bars: int,
    fee: float, policy_fn,                           # callable(s_28,) -> {0,1}
    transitions_out: list = None,
):
    """Walk one trade bar-by-bar, querying policy_fn at each in-trade bar.

    Rule-based exits (TP/SL/BE/trail/time) and RL EXIT_NOW are both active;
    whichever fires first ends the trade. Returns (pnl, n_held, exit_reason).

    `transitions_out` (if given) is appended with dicts:
       {state, action, reward, duration=1, state_next, done}
    The transitions are post-processed by the caller to push into the buffer.
    """
    n_bars = len(prices)
    entry  = prices[entry_bar] * (1.0 + direction * fee)

    # initial SL / trail state
    cur_sl    = entry * (1.0 - direction * sl_pct)
    cur_trail = trail_pct
    be_done   = False

    # validity check (matches single_trade.py guard)
    tp_price0 = entry * (1.0 + direction * tp_pct)
    if direction == 1:
        if not (tp_price0 > entry and cur_sl < entry):
            return 0.0, 0, "INVALID"
    else:
        if not (tp_price0 < entry and cur_sl > entry):
            return 0.0, 0, "INVALID"

    end = min(n_bars, entry_bar + 1 + min(MAX_TRADE_BARS,
              time_stop_bars if time_stop_bars > 0 else MAX_TRADE_BARS))

    prev_s = None; prev_a = None; prev_r = None  # delayed push

    for i in range(entry_bar + 1, end):
        price = prices[i]
        n_in  = i - entry_bar

        # update breakeven, then trail, BEFORE checking exits (matches simulate_one_trade)
        if breakeven_pct > 0.0 and not be_done:
            if direction * (price / entry - 1.0) >= breakeven_pct:
                cur_sl = entry
                be_done = True
                if tab_pct > 0.0:
                    cur_trail = tab_pct
        if cur_trail > 0.0:
            if direction == 1:
                cand = price * (1.0 - cur_trail)
                if cand > cur_sl: cur_sl = cand
            else:
                cand = price * (1.0 + cur_trail)
                if cand < cur_sl: cur_sl = cand

        unreal  = direction * (price / entry - 1.0) - 2.0 * fee
        sl_dist = max(0.0, direction * (price - cur_sl) / max(price, 1e-12))

        s_t = build_exit_state(state[i], unreal, n_in, sl_dist, direction)

        # check forced rule exits FIRST
        tp_price = entry * (1.0 + direction * tp_pct)
        hit_tp = (direction == 1 and price >= tp_price) or (direction == -1 and price <= tp_price)
        hit_sl = (direction == 1 and price <= cur_sl)   or (direction == -1 and price >= cur_sl)
        time_stop_hit = (time_stop_bars > 0 and n_in >= time_stop_bars)

        # query policy for this bar's intended action (HOLD/EXIT_NOW)
        action_intended = int(policy_fn(s_t))

        # determine actual outcome
        if hit_tp:
            ep = tp_price
            pnl = direction * (ep / entry - 1.0) - 2.0 * fee
            done = True; reason = "TP"; effective_action = 0
        elif hit_sl:
            ep = cur_sl
            pnl = direction * (ep / entry - 1.0) - 2.0 * fee
            done = True; reason = "SL"; effective_action = 0
        elif time_stop_hit:
            pnl = unreal
            done = True; reason = "TIME"; effective_action = 0
        elif action_intended == 1:                          # RL EXIT_NOW
            # exit at NEXT bar (1-bar exec lag)
            j = i + 1 if i + 1 < n_bars else i
            pnl = direction * (prices[j] / entry - 1.0) - 2.0 * fee
            done = True; reason = "RL_EXIT"; effective_action = 1
            n_in = j - entry_bar
        else:
            pnl = 0.0
            done = False; reason = ""; effective_action = 0

        # delayed transition push: prev (s_{i-1}, a_{i-1}, r_{i-1}) gets s_t as s_next
        if transitions_out is not None and prev_s is not None:
            transitions_out.append(dict(
                state=prev_s, action=prev_a, reward=prev_r,
                state_next=s_t, done=False,
            ))

        if done:
            # final transition: terminal reward = realized PnL (raw, fees included)
            if transitions_out is not None:
                # terminal s_next can be s_t (placeholder; done=True so no bootstrap)
                transitions_out.append(dict(
                    state=s_t, action=effective_action, reward=pnl,
                    state_next=s_t, done=True,
                ))
            return pnl, n_in, reason

        prev_s = s_t; prev_a = effective_action; prev_r = 0.0

    # ran past loop without exit (end of price array): close at last bar
    last = prices[end - 1]
    pnl = direction * (last / entry - 1.0) - 2.0 * fee
    n_in = end - 1 - entry_bar
    if transitions_out is not None:
        s_t = build_exit_state(state[end - 1],
                                pnl, n_in, 0.0, direction)
        transitions_out.append(dict(
            state=s_t, action=0, reward=pnl,
            state_next=s_t, done=True,
        ))
    return pnl, n_in, "EOD"


# ── chunked rollout driver ───────────────────────────────────────────────────

def rollout_chunk_exit(
    state, signals_strat, prices,
    tp, sl, trail, tab, be, ts_bars,
    policy_fn,                                 # callable(s_28,) -> {0,1}
    buffer: ReplayBuffer, cursor: dict,
    max_transitions: int,
    fee: float, reward_scale: float = REWARD_SCALE,
    strat_filter: int = -1,                    # -1 = any, else only strat k
):
    """Walks bars sequentially, picks first firing strategy (filtered to `strat_filter`
    if set), simulates one trade with rule+policy exits, pushes transitions, advances."""
    n_bars   = len(prices)
    K        = signals_strat.shape[1]
    valid_next = np.array([True, True], dtype=np.bool_)

    t        = cursor["t"]
    n_pushed = 0
    n_trades = 0
    rl_exits = 0; tp_exits = 0; sl_exits = 0; time_exits = 0
    pnls_chunk = []

    while n_pushed < max_transitions:
        if t >= n_bars - 2:
            t = 0  # wrap

        # find next entry event
        k_use = -1
        while t < n_bars - 2:
            if strat_filter >= 0:
                if signals_strat[t, strat_filter] != 0:
                    k_use = strat_filter; break
            else:
                for k in range(K):
                    if signals_strat[t, k] != 0:
                        k_use = k; break
                if k_use >= 0:
                    break
            t += 1
        if k_use < 0:
            break

        direction = int(signals_strat[t, k_use])
        # collect transitions for this trade
        transitions = []
        pnl, n_held, reason = simulate_trade_episode(
            state, prices, entry_bar=t + 1, direction=direction,
            tp_pct=float(tp[t, k_use]), sl_pct=float(sl[t, k_use]),
            trail_pct=float(trail[t, k_use]), tab_pct=float(tab[t, k_use]),
            breakeven_pct=float(be[t, k_use]),
            time_stop_bars=int(ts_bars[t, k_use]),
            fee=fee, policy_fn=policy_fn,
            transitions_out=transitions,
        )

        if reason == "INVALID" or len(transitions) == 0:
            t += 1
            continue

        # push transitions to buffer with reward scaling
        for tr_i, tr in enumerate(transitions):
            r_buf = tr["reward"] * reward_scale
            buffer.push(
                tr["state"], tr["action"], r_buf, 1,
                tr["state_next"], valid_next, tr["done"],
            )
            n_pushed += 1
            if n_pushed >= max_transitions:
                break

        n_trades += 1
        pnls_chunk.append(pnl)
        if reason == "RL_EXIT":   rl_exits  += 1
        elif reason == "TP":      tp_exits  += 1
        elif reason == "SL":      sl_exits  += 1
        elif reason == "TIME":    time_exits += 1

        t = t + 1 + n_held + 1

    return dict(t=t, n_pushed=n_pushed, n_trades=n_trades,
                 pnls=pnls_chunk,
                 rl_exits=rl_exits, tp_exits=tp_exits,
                 sl_exits=sl_exits, time_exits=time_exits)


# ── policy wrappers ──────────────────────────────────────────────────────────

class _UniformRandomExit:
    """HOLD-biased random: 10% EXIT_NOW, 90% HOLD. Ensures warmup buffer
    contains full trade trajectories with rule-fired terminal rewards
    (rather than 50% bar-1 exits that dominate naive uniform random)."""
    EXIT_PROB = 0.10
    def __init__(self, rng): self.rng = rng
    def __call__(self, s):
        return 1 if self.rng.random() < self.EXIT_PROB else 0


class _AlwaysHold:
    def __call__(self, s): return 0


class _EpsilonGreedyExit:
    """ε-greedy with HOLD-biased random arm (10% EXIT). With 2 actions,
    pure-uniform exploration would terminate every other trade at bar 1,
    starving the buffer of long trajectories."""
    EXIT_PROB = 0.10
    def __init__(self, net, rng, step_ref):
        self.net = net; self.rng = rng; self.step_ref = step_ref
    def __call__(self, s):
        eps = epsilon(self.step_ref["step"])
        if self.rng.random() < eps:
            return 1 if self.rng.random() < self.EXIT_PROB else 0
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            return int(self.net(sb).argmax(dim=-1).item())


class _GreedyExit:
    def __init__(self, net): self.net = net
    def __call__(self, s):
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            return int(self.net(sb).argmax(dim=-1).item())


def epsilon(step: int) -> float:
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + (EPS_END - EPS_START) * frac


def per_beta(step: int) -> float:
    frac = min(1.0, step / TOTAL_GRAD_STEPS)
    return PER_BETA_START + (PER_BETA_END - PER_BETA_START) * frac


# ── Bellman loss (no action mask, both actions always valid) ─────────────────

def bellman_loss(online: ExitDQN, target: ExitDQN, batch: dict, gamma: float,
                  is_w: torch.Tensor):
    s      = torch.from_numpy(batch["state"]).float()
    a      = torch.from_numpy(batch["action"].astype(np.int64))
    r      = torch.from_numpy(batch["reward"]).float()
    dur    = torch.from_numpy(batch["duration"].astype(np.int64))
    s_next = torch.from_numpy(batch["state_next"]).float()
    done   = torch.from_numpy(batch["done"]).bool()

    q_sa = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        q_next_max = target(s_next).max(dim=-1).values
        bootstrap  = (gamma ** dur.float()) * q_next_max
        bootstrap[done] = 0.0
        tgt        = r + bootstrap
    td_err = q_sa - tgt
    loss = (is_w * F.huber_loss(q_sa, tgt, reduction="none", delta=HUBER_DELTA)).mean()
    return loss, td_err.detach().cpu().numpy()


# ── evaluator ────────────────────────────────────────────────────────────────

def evaluate_policy(
    policy_fn, state, signals_strat, prices,
    tp, sl, trail, tab, be, ts_bars, fee: float,
    strat_filter: int = -1,
):
    """Sequentially simulate trades through the entire `state` length, applying
    `policy_fn` per in-trade bar. Returns Sharpe + per-trade stats."""
    n_bars = len(prices)
    K      = signals_strat.shape[1]
    equity = 1.0; peak = 1.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []
    trade_durs = []
    rl_exits = tp_exits = sl_exits = time_exits = eod_exits = 0

    t = 0
    while t < n_bars - 2:
        # find next entry
        k_use = -1
        while t < n_bars - 2:
            if strat_filter >= 0:
                if signals_strat[t, strat_filter] != 0:
                    k_use = strat_filter; break
            else:
                for k in range(K):
                    if signals_strat[t, k] != 0:
                        k_use = k; break
                if k_use >= 0:
                    break
            t += 1
        if k_use < 0:
            break

        direction = int(signals_strat[t, k_use])
        pnl, n_held, reason = simulate_trade_episode(
            state, prices, entry_bar=t + 1, direction=direction,
            tp_pct=float(tp[t, k_use]), sl_pct=float(sl[t, k_use]),
            trail_pct=float(trail[t, k_use]), tab_pct=float(tab[t, k_use]),
            breakeven_pct=float(be[t, k_use]),
            time_stop_bars=int(ts_bars[t, k_use]),
            fee=fee, policy_fn=policy_fn,
            transitions_out=None,
        )
        if reason == "INVALID":
            t += 1
            continue

        t_close = t + 1 + n_held
        if t_close >= n_bars: t_close = n_bars - 1
        eq_arr[t:t_close + 1] = equity
        equity *= (1.0 + pnl)
        eq_arr[t_close + 1:] = equity
        peak = max(peak, equity)
        trade_pnls.append(pnl)
        trade_durs.append(n_held)
        if   reason == "RL_EXIT": rl_exits += 1
        elif reason == "TP":      tp_exits += 1
        elif reason == "SL":      sl_exits += 1
        elif reason == "TIME":    time_exits += 1
        elif reason == "EOD":     eod_exits += 1

        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    trade_pnls_arr = np.array(trade_pnls, dtype=np.float64)
    win_rate = (trade_pnls_arr > 0).mean() if len(trade_pnls_arr) else 0.0

    eq_for_dd = eq_arr / np.maximum.accumulate(eq_arr)
    max_dd = float(eq_for_dd.min() - 1.0) if len(eq_arr) else 0.0

    return dict(
        sharpe       = float(sharpe),
        equity_final = float(equity),
        equity_peak  = float(peak),
        max_dd       = max_dd,
        n_trades     = int(len(trade_pnls_arr)),
        win_rate     = float(win_rate),
        mean_pnl_pct = float(trade_pnls_arr.mean() * 100) if len(trade_pnls_arr) else 0.0,
        mean_duration= float(np.mean(trade_durs)) if trade_durs else 0.0,
        rl_exit_pct  = float(rl_exits / max(1, len(trade_pnls_arr)) * 100),
        exit_breakdown = dict(
            RL_EXIT=int(rl_exits), TP=int(tp_exits), SL=int(sl_exits),
            TIME=int(time_exits), EOD=int(eod_exits)),
    )


# ── main training ────────────────────────────────────────────────────────────

def train(ticker: str = "btc", seed: int = 42, tag: str = "B1",
           fee: float = 0.0008, strat_filter: int = -1):
    torch.manual_seed(seed); np.random.seed(seed)
    rng_warm = np.random.default_rng(seed)
    rng_eps  = np.random.default_rng(seed + 1)

    t_start = time.perf_counter()
    print(f"\n{'='*78}\n  GROUP B — EXIT-TIMING DQN  ({ticker.upper()})  tag={tag}\n"
          f"  fee={fee:.4f}  strat_filter="
          f"{STRAT_KEYS[strat_filter] if strat_filter >= 0 else 'ALL'}\n"
          f"  state_dim={EXIT_STATE_DIM}  n_actions={N_ACTIONS}  γ={GAMMA}  "
          f"reward_scale={REWARD_SCALE}\n{'='*78}")

    # ── load DQN-train and DQN-val arrays (same as entry DQN) ───────────────
    sp_tr = np.load(CACHE / "state" / f"{ticker}_dqn_state_train.npz")
    sp_v  = np.load(CACHE / "state" / f"{ticker}_dqn_state_val.npz")
    print(f"  DQN-train: state {sp_tr['state'].shape}")
    print(f"  DQN-val  : state {sp_v['state'].shape}")

    vol = np.load(CACHE / "preds" / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])

    # ── build per-bar exit-param arrays ─────────────────────────────────────
    print("  building exit param arrays for train + val ...")
    tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts = _build_exit_arrays(
        sp_tr["price"], sp_tr["atr"], atr_median)
    v_tp, v_sl, v_tr, v_tab, v_be, v_ts = _build_exit_arrays(
        sp_v["price"], sp_v["atr"], atr_median)

    # ── networks + optimizer + buffer ───────────────────────────────────────
    online = ExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=64)
    target = ExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=64)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    buf = ReplayBuffer(capacity=BUFFER_SIZE, state_dim=EXIT_STATE_DIM,
                        n_actions=N_ACTIONS)
    print(f"  ExitDQN params: {online.n_params():,}")

    # ── baseline (no RL exit) eval on val for reference ─────────────────────
    print(f"\n  Computing rule-only baseline on DQN-val (always HOLD) ...")
    t0 = time.perf_counter()
    baseline = evaluate_policy(
        _AlwaysHold(), sp_v["state"], sp_v["signals"], sp_v["price"],
        v_tp, v_sl, v_tr, v_tab, v_be, v_ts, fee=fee, strat_filter=strat_filter)
    print(f"    rule-only:  Sharpe {baseline['sharpe']:+.3f}  "
          f"trades {baseline['n_trades']:,}  win {baseline['win_rate']*100:.1f}%  "
          f"meanPnL {baseline['mean_pnl_pct']:+.3f}%  eq {baseline['equity_final']:.3f}  "
          f"[{time.perf_counter()-t0:.1f}s]")
    print(f"    exit reasons: {baseline['exit_breakdown']}")

    # ── warmup with random policy ───────────────────────────────────────────
    print(f"\n  Warmup: {WARMUP_STEPS:,} random transitions ...")
    cursor_tr = dict(t=0)
    rand_pol = _UniformRandomExit(rng_warm)
    t_warm = time.perf_counter()
    while len(buf) < WARMUP_STEPS:
        info = rollout_chunk_exit(
            sp_tr["state"], sp_tr["signals"], sp_tr["price"],
            tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
            policy_fn=rand_pol, buffer=buf, cursor=cursor_tr,
            max_transitions=min(2000, WARMUP_STEPS - len(buf)),
            fee=fee, strat_filter=strat_filter,
        )
        cursor_tr = dict(t=info["t"])
    n_done_in_buf = int(buf.done[:len(buf)].sum())
    print(f"    buffer={len(buf):,}  ({n_done_in_buf:,} terminal) "
          f"[{time.perf_counter()-t_warm:.1f}s]")

    # ── main loop ───────────────────────────────────────────────────────────
    step_ref = {"step": 0}
    eps_pol  = _EpsilonGreedyExit(online, rng_eps, step_ref)

    history = []
    best_val_sharpe = -np.inf
    best_step       = 0
    losses          = []

    print(f"\n  Training: {TOTAL_GRAD_STEPS:,} grad steps "
          f"(refresh M={REFRESH_M}/{REFRESH_EVERY}, target sync {TARGET_SYNC_EVERY}, "
          f"val every {VAL_EVERY})")
    print()
    t_loop = time.perf_counter()

    for step in range(1, TOTAL_GRAD_STEPS + 1):
        step_ref["step"] = step

        if step % REFRESH_EVERY == 0:
            info = rollout_chunk_exit(
                sp_tr["state"], sp_tr["signals"], sp_tr["price"],
                tr_tp, tr_sl, tr_tr, tr_tab, tr_be, tr_ts,
                policy_fn=eps_pol, buffer=buf, cursor=cursor_tr,
                max_transitions=REFRESH_M,
                fee=fee, strat_filter=strat_filter,
            )
            cursor_tr = dict(t=info["t"])

        # gradient step (prioritized, no stratification — both actions present in episodic data)
        batch, idx, is_w_np = buf.sample_prioritized(
            BATCH_SIZE, alpha=PER_ALPHA, beta=per_beta(step))
        is_w = torch.from_numpy(is_w_np)
        loss, td_errs = bellman_loss(online, target, batch, GAMMA, is_w)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
        optimizer.step()
        buf.update_priorities(idx, td_errs)
        losses.append(float(loss.item()))

        if step % TARGET_SYNC_EVERY == 0:
            target.load_state_dict(online.state_dict())

        if step % VAL_EVERY == 0:
            t_eval = time.perf_counter()
            greedy = _GreedyExit(online)
            val = evaluate_policy(
                greedy, sp_v["state"], sp_v["signals"], sp_v["price"],
                v_tp, v_sl, v_tr, v_tab, v_be, v_ts, fee=fee,
                strat_filter=strat_filter)
            improved = val["sharpe"] > best_val_sharpe
            marker = "★" if improved else " "
            mean_loss = float(np.mean(losses[-VAL_EVERY:]))
            elapsed = time.perf_counter() - t_loop
            print(f"  [step {step:>6,}] {marker} ε={epsilon(step):.3f}  "
                  f"loss={mean_loss:.4f}  Sharpe={val['sharpe']:>+6.3f} "
                  f"(rule-only {baseline['sharpe']:+.3f})  "
                  f"trd={val['n_trades']:>4} win={val['win_rate']*100:.0f}% "
                  f"eq={val['equity_final']:.3f} dd={val['max_dd']*100:>+5.1f}% "
                  f"RLexit={val['rl_exit_pct']:.0f}% "
                  f"[loop {elapsed:>4.0f}s + val {time.perf_counter()-t_eval:.1f}s]")
            history.append(dict(
                step=step, eps=epsilon(step), mean_loss=mean_loss,
                val_sharpe=val["sharpe"], val_trades=val["n_trades"],
                val_winrate=val["win_rate"], val_equity=val["equity_final"],
                val_max_dd=val["max_dd"],
                rl_exit_pct=val["rl_exit_pct"],
                exit_breakdown=val["exit_breakdown"],
            ))
            if improved:
                best_val_sharpe = val["sharpe"]
                best_step       = step
                torch.save(online.state_dict(),
                            CACHE / "policies" / f"{ticker}_exit_dqn_policy_{tag}.pt")
            if step - best_step > EARLY_STOP_PATIENCE:
                print(f"\n  Early stop at step {step} "
                      f"(best Sharpe={best_val_sharpe:+.3f} @ {best_step:,})")
                break

    elapsed_total = time.perf_counter() - t_start

    # ── summary ─────────────────────────────────────────────────────────────
    print(f"\n\n{'='*78}\n  EXIT-DQN TRAINING SUMMARY  ({tag})\n{'='*78}")
    print(f"  total wall time      : {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  best val Sharpe      : {best_val_sharpe:+.3f}  (step {best_step:,})")
    print(f"  rule-only baseline   : {baseline['sharpe']:+.3f}")
    print(f"  Δ vs rule-only       : {best_val_sharpe - baseline['sharpe']:+.3f}")
    print(f"  policy saved to      : cache/{ticker}_exit_dqn_policy_{tag}.pt")

    hist_path = CACHE / "policies" / f"{ticker}_exit_dqn_history_{tag}.json"
    hist_path.write_text(json.dumps(dict(
        ticker=ticker, run_at=datetime.utcnow().isoformat(),
        tag=tag, fee=fee, strat_filter=strat_filter,
        baseline=baseline,
        config=dict(
            gamma=GAMMA, lr=LR, batch=BATCH_SIZE, buffer=BUFFER_SIZE,
            warmup=WARMUP_STEPS, total_steps=TOTAL_GRAD_STEPS,
            refresh_every=REFRESH_EVERY, refresh_m=REFRESH_M,
            target_sync_every=TARGET_SYNC_EVERY, val_every=VAL_EVERY,
            patience=EARLY_STOP_PATIENCE,
            eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_STEPS,
            per_alpha=PER_ALPHA, reward_scale=REWARD_SCALE,
            state_dim=EXIT_STATE_DIM, n_actions=N_ACTIONS,
            max_trade_bars=MAX_TRADE_BARS,
        ),
        best_val_sharpe=best_val_sharpe, best_step=best_step,
        elapsed_seconds=elapsed_total,
        history=history,
    ), indent=2, default=str))
    print(f"  history saved to     : {hist_path.name}")
    return dict(best_val_sharpe=best_val_sharpe, best_step=best_step,
                 baseline=baseline, history=history, elapsed=elapsed_total)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--tag", default="B1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee", type=float, default=0.0008,
                     help="per-side fee (B1=0.0008 taker, B2=0.0004 maker, B3=0)")
    ap.add_argument("--strat-filter", type=int, default=-1, dest="strat_filter",
                     help="restrict to single strategy index 0..8 (B4 cells)")
    args = ap.parse_args()
    train(args.ticker, seed=args.seed, tag=args.tag, fee=args.fee,
           strat_filter=args.strat_filter)
