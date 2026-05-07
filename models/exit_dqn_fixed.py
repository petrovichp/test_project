"""
Fixed-window exit-timing DQN (Group B5).

Differences vs Group B (exit_dqn.py):
  - Episodes are FIXED LENGTH (N bars). DQN must pick exit bar within window
    or accept forced-close at bar N.
  - Rule-based exits (TP/SL/BE/trail/time-stop) DISABLED during training.
    Only the DQN's EXIT_NOW or the window edge terminate an episode.
  - State is 53-dim and includes: in-trade trajectory scalars, price-path
    window (last 20 bars cum-return-from-entry), volatility window
    (last 10 bars |log-return|), time-of-day cyclic, entry-time static
    context, plus current market aggregates.

Rationale: B4 had emergent variable-length episodes where rule-fired
terminals dominated the buffer. The DQN never observed "what if I held
longer than the rule allowed", and its credit-assignment was confounded
by the rule-vs-DQN race. Here every episode is N bars, every terminal is
DQN-fired or window-fired, and the state has direct visibility into the
trade's price path.

State (53 dims):
   0  unrealized_pnl_pct                     clip ±10
   1  bars_in_trade / N                      ∈ [0, 1]
   2  bars_remaining / N                     ∈ [0, 1]
   3  entry_direction                        ±1
   4  max_unrealized_pnl_so_far              clip 0..10
   5  min_unrealized_pnl_so_far              clip -10..0
   6  bars_since_peak / N                    ∈ [0, 1]
   7  realized_vol_in_trade                  clip 0..5 (sqrt-Σ-bar-ret²)
   8  hour_of_day_sin                        ∈ [-1, 1]
   9  hour_of_day_cos                        ∈ [-1, 1]
  10..29  PRICE PATH (20 dims, last 20 bars cum-return-from-entry × 100)
                  padded with 0 for bars before entry
  30..39  VOLATILITY WINDOW (10 dims, |log_ret| at last 10 bars, standardized)
  40  vol_pred at entry bar                  standardized (sliced from base state)
  41  bb_width at entry bar                  standardized
  42  regime_id at entry / 4.0               ∈ [0, 1]
  43..48  log_return × 6 lags [60,30,15,5,1,0]   sliced from base state[t]
  49..52  taker_net_60_z × 4 lags [15,5,1,0]      sliced from base state[t]

Actions: 0 = HOLD, 1 = EXIT_NOW
Reward : 0 on HOLD, realized PnL net of fees at terminal (DQN-fired OR window edge).
γ      : 1.0 (undiscounted within bounded episode).

Run: python3 -m models.exit_dqn_fixed btc --tag B5_fix120_fee0_S3 --fee 0 \
                                            --strat-filter 3 --window-n 120
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
from models.dqn_rollout import STRAT_KEYS

CACHE = ROOT / "cache"

# ── architecture ─────────────────────────────────────────────────────────────
EXIT_STATE_DIM       = 53
N_ACTIONS            = 2
DEFAULT_WINDOW_N     = 120
PRICE_PATH_K         = 20
VOL_WINDOW_K         = 10

# ── hyperparameters (mostly mirror exit_dqn.py) ──────────────────────────────
GAMMA                = 1.0
LR                   = 1e-3
BATCH_SIZE           = 128
BUFFER_SIZE          = 80_000
WARMUP_STEPS         = 5_000
TOTAL_GRAD_STEPS     = 80_000
REFRESH_EVERY        = 500
REFRESH_M            = 2_000
TARGET_SYNC_EVERY    = 1_000
VAL_EVERY            = 4_000
EARLY_STOP_PATIENCE  = 16_000
EPS_START, EPS_END   = 1.0, 0.05
EPS_DECAY_STEPS      = 40_000
PER_ALPHA            = 0.6
PER_BETA_START, PER_BETA_END = 0.4, 1.0
HUBER_DELTA          = 1.0
GRAD_CLIP            = 10.0
REWARD_SCALE         = 100.0
RANDOM_EXIT_PROB     = 0.10                    # HOLD-biased exploration

# split constants (must match dqn_state.py)
WARMUP_BARS = 1440
VOL_TRAIN_BARS = 100_000   # vol-train slice within sliced arrays


# ── network ──────────────────────────────────────────────────────────────────

class FixedExitDQN(nn.Module):
    def __init__(self, state_dim: int = EXIT_STATE_DIM, n_actions: int = N_ACTIONS,
                  hidden: int = 96):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_actions)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── one-time precompute (per training run) ───────────────────────────────────

def precompute_aux_arrays(prices: np.ndarray, ts: np.ndarray):
    """Compute |log_return|-standardized array and hour-of-day sin/cos.

    The standardize stats for |log_return| are fitted on the vol-train slice
    (first VOL_TRAIN_BARS bars of the input — note the input is already past
    WARMUP per dqn_state.py)."""
    # |log_return| at each bar (bar-i return = log(p[i]/p[i-1]))
    safe_p     = np.maximum(prices, 1e-8)
    log_ret    = np.diff(np.log(safe_p), prepend=np.log(safe_p[0])).astype(np.float32)
    abs_log_ret = np.abs(log_ret)

    # standardize on first VOL_TRAIN_BARS bars (vol-train slice)
    n_fit = min(VOL_TRAIN_BARS, len(abs_log_ret))
    med = float(np.median(abs_log_ret[:n_fit]))
    iqr = float(np.percentile(abs_log_ret[:n_fit], 75)
                  - np.percentile(abs_log_ret[:n_fit], 25))
    iqr = max(iqr, 1e-8)
    abs_log_ret_std = np.clip((abs_log_ret - med) / iqr, -10.0, 10.0).astype(np.float32)

    # hour-of-day cyclic encoding
    sec_in_day = ts % 86400
    angle = (2.0 * np.pi * sec_in_day / 86400.0).astype(np.float64)
    hour_sin = np.sin(angle).astype(np.float32)
    hour_cos = np.cos(angle).astype(np.float32)

    return dict(abs_log_ret_std=abs_log_ret_std,
                 hour_sin=hour_sin, hour_cos=hour_cos,
                 abs_log_ret_med=med, abs_log_ret_iqr=iqr)


# ── state builder ────────────────────────────────────────────────────────────

def build_fixed_exit_state(
    base_state_arr,           # (n_bars, 50)
    prices,                   # (n_bars,)
    aux,                      # dict from precompute_aux_arrays
    regime_id,                # (n_bars,) int8
    entry_bar: int, current_bar: int, direction: int,
    entry_price: float,
    max_unrealized: float, min_unrealized: float,
    bars_since_peak: int, running_vol: float,
    N: int, fee: float,
    vol_at_entry: float, bb_at_entry: float, regime_at_entry: float,
    out: np.ndarray = None,
) -> np.ndarray:
    if out is None:
        out = np.zeros(EXIT_STATE_DIM, dtype=np.float32)

    n_in    = current_bar - entry_bar
    cur_p   = prices[current_bar]
    unreal  = direction * (cur_p / entry_price - 1.0) - 2.0 * fee

    # In-trade scalars (8)
    out[0] = max(-10.0, min(10.0, unreal * 100.0))
    out[1] = n_in / N
    out[2] = max(0.0, min(1.0, (N - n_in) / N))
    out[3] = float(direction)
    out[4] = max(0.0, min(10.0, max_unrealized * 100.0))
    out[5] = max(-10.0, min(0.0, min_unrealized * 100.0))
    out[6] = bars_since_peak / N
    out[7] = max(0.0, min(5.0, running_vol * 100.0))

    # Cyclic time (2) — precomputed, just slice
    out[8] = aux["hour_sin"][current_bar]
    out[9] = aux["hour_cos"][current_bar]

    # Price path (20) — last K=20 bars cumulative-return-from-entry × 100
    # bars in window: [current_bar - 19, current_bar]
    # padded with 0 for bars at or before entry_bar
    K = PRICE_PATH_K
    for i in range(K):
        bar_idx = current_bar - (K - 1 - i)
        if bar_idx <= entry_bar:
            out[10 + i] = 0.0
        else:
            cum_ret = direction * (prices[bar_idx] / entry_price - 1.0) - 2.0 * fee
            out[10 + i] = max(-10.0, min(10.0, cum_ret * 100.0))

    # Volatility window (10) — last 10 bars |log_return| (precomputed, standardized)
    K_v = VOL_WINDOW_K
    start = current_bar - (K_v - 1)
    if start >= 0:
        out[30:30 + K_v] = aux["abs_log_ret_std"][start:current_bar + 1]
    else:
        # head-padded: leave zeros for missing prefix
        out[30:30 + K_v] = 0.0
        valid_n = current_bar + 1
        out[30 + K_v - valid_n:30 + K_v] = aux["abs_log_ret_std"][:current_bar + 1]

    # Entry-time static context (3)
    out[40] = vol_at_entry
    out[41] = bb_at_entry
    out[42] = regime_at_entry / 4.0

    # Current market aggregates (10) — sliced from cached entry-DQN state
    out[43:49] = base_state_arr[current_bar, 20:26]   # log_return × 6 lags
    out[49:53] = base_state_arr[current_bar, 28:32]   # taker_net_60_z × 4 (lags 15,5,1,0)
    return out


# ── per-trade simulator (NO rule-based exits during training) ────────────────

def simulate_fixed_episode(
    base_state_arr, prices, aux, regime_id,
    entry_bar: int, direction: int, fee: float, N: int,
    policy_fn,
    transitions_out: list = None,
):
    """Walk fixed N-bar window after entry. NO rule-based exits — only DQN's
    EXIT_NOW or window-edge force-close.

    Returns (pnl, n_held, exit_reason).  exit_reason ∈ {RL_EXIT, WINDOW, EOD}.
    """
    n_bars = len(prices)
    entry_price = prices[entry_bar] * (1.0 + direction * fee)

    # Slice entry-time static once
    base_at_entry  = base_state_arr[entry_bar]
    vol_at_entry   = float(base_at_entry[0])
    bb_at_entry    = float(base_at_entry[16])
    regime_at_e    = float(regime_id[entry_bar])

    # Running stats
    max_unrealized  = 0.0
    min_unrealized  = 0.0
    bars_since_peak = 0
    sum_sq_returns  = 0.0

    end = min(n_bars, entry_bar + 1 + N)
    # if end <= entry_bar + 1: degenerate — shouldn't happen since rollout filters
    if end <= entry_bar + 1:
        return 0.0, 0, "INVALID"

    prev_s = None; prev_a = None

    for i in range(entry_bar + 1, end):
        n_in   = i - entry_bar
        cur_p  = prices[i]
        unreal = direction * (cur_p / entry_price - 1.0) - 2.0 * fee

        # Update running stats BEFORE state-build so they reflect the current bar
        if unreal > max_unrealized:
            max_unrealized   = unreal
            bars_since_peak  = 0
        else:
            bars_since_peak += 1
        if unreal < min_unrealized:
            min_unrealized = unreal

        # Per-bar return (for running vol)
        if i > entry_bar + 1:
            prev_p = max(prices[i - 1], 1e-8)
            bar_ret = (cur_p - prev_p) / prev_p
            sum_sq_returns += bar_ret * bar_ret
        running_vol = float(np.sqrt(sum_sq_returns))

        s_t = build_fixed_exit_state(
            base_state_arr, prices, aux, regime_id,
            entry_bar=entry_bar, current_bar=i, direction=direction,
            entry_price=entry_price,
            max_unrealized=max_unrealized, min_unrealized=min_unrealized,
            bars_since_peak=bars_since_peak, running_vol=running_vol,
            N=N, fee=fee,
            vol_at_entry=vol_at_entry, bb_at_entry=bb_at_entry,
            regime_at_entry=regime_at_e,
        )

        action_intended = int(policy_fn(s_t))

        # Determine outcome
        is_window_edge = (i == end - 1)
        is_data_end    = (i >= n_bars - 1)

        if action_intended == 1 and not is_data_end:
            # DQN exits → close at NEXT bar's price (1-bar exec lag)
            j   = i + 1
            pnl = direction * (prices[j] / entry_price - 1.0) - 2.0 * fee
            done = True; reason = "RL_EXIT"; effective_action = 1
        elif is_window_edge or is_data_end:
            # Force-close at this bar (last in window or last in data)
            pnl = unreal
            done = True
            reason = "EOD" if is_data_end and not is_window_edge else "WINDOW"
            effective_action = 0
        else:
            pnl = 0.0; done = False; reason = ""; effective_action = 0

        # Delayed-write transition push
        if transitions_out is not None and prev_s is not None:
            transitions_out.append(dict(
                state=prev_s, action=prev_a, reward=0.0,
                state_next=s_t, done=False,
            ))

        if done:
            if transitions_out is not None:
                transitions_out.append(dict(
                    state=s_t, action=effective_action, reward=pnl,
                    state_next=s_t, done=True,
                ))
            return pnl, n_in, reason

        prev_s = s_t; prev_a = effective_action

    # safety; should be caught by is_window_edge above
    return 0.0, end - 1 - entry_bar, "?"


# ── chunked rollout ──────────────────────────────────────────────────────────

def rollout_chunk_fixed(
    base_state_arr, signals_strat, prices, aux, regime_id,
    policy_fn, buffer: ReplayBuffer, cursor: dict,
    max_transitions: int,
    fee: float, N: int, strat_filter: int = -1,
    reward_scale: float = REWARD_SCALE,
):
    n_bars   = len(prices)
    K_strat  = signals_strat.shape[1]
    valid_next = np.array([True, True], dtype=np.bool_)

    t        = cursor["t"]
    n_pushed = 0
    n_trades = 0
    rl_exits = window_exits = eod_exits = 0
    pnls_chunk = []

    while n_pushed < max_transitions:
        if t >= n_bars - 2:
            t = 0

        # find next entry event
        k_use = -1
        while t < n_bars - 2:
            if strat_filter >= 0:
                if signals_strat[t, strat_filter] != 0:
                    k_use = strat_filter; break
            else:
                for k in range(K_strat):
                    if signals_strat[t, k] != 0:
                        k_use = k; break
                if k_use >= 0:
                    break
            t += 1
        if k_use < 0:
            break

        direction = int(signals_strat[t, k_use])
        transitions = []
        pnl, n_held, reason = simulate_fixed_episode(
            base_state_arr, prices, aux, regime_id,
            entry_bar=t + 1, direction=direction, fee=fee, N=N,
            policy_fn=policy_fn, transitions_out=transitions,
        )
        if reason == "INVALID" or len(transitions) == 0:
            t += 1
            continue

        # push
        for tr in transitions:
            r_buf = tr["reward"] * reward_scale
            buffer.push(tr["state"], tr["action"], r_buf, 1,
                          tr["state_next"], valid_next, tr["done"])
            n_pushed += 1
            if n_pushed >= max_transitions:
                break

        n_trades += 1
        pnls_chunk.append(pnl)
        if   reason == "RL_EXIT":  rl_exits     += 1
        elif reason == "WINDOW":   window_exits += 1
        elif reason == "EOD":      eod_exits    += 1

        # advance cursor: jump past this trade entirely (sequential, non-overlapping)
        t = t + 1 + n_held + 1

    return dict(t=t, n_pushed=n_pushed, n_trades=n_trades,
                 pnls=pnls_chunk,
                 rl_exits=rl_exits, window_exits=window_exits, eod_exits=eod_exits)


# ── policy wrappers ──────────────────────────────────────────────────────────

class _UniformRandomFixed:
    EXIT_PROB = RANDOM_EXIT_PROB
    def __init__(self, rng): self.rng = rng
    def __call__(self, s):
        return 1 if self.rng.random() < self.EXIT_PROB else 0

class _AlwaysHold:
    def __call__(self, s): return 0

class _EpsilonGreedyFixed:
    EXIT_PROB = RANDOM_EXIT_PROB
    def __init__(self, net, rng, step_ref):
        self.net = net; self.rng = rng; self.step_ref = step_ref
    def __call__(self, s):
        eps = epsilon(self.step_ref["step"])
        if self.rng.random() < eps:
            return 1 if self.rng.random() < self.EXIT_PROB else 0
        with torch.no_grad():
            sb = torch.from_numpy(s).float().unsqueeze(0)
            return int(self.net(sb).argmax(dim=-1).item())

class _GreedyFixed:
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


# ── Bellman loss ─────────────────────────────────────────────────────────────

def bellman_loss(online, target, batch, gamma, is_w):
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
    policy_fn, base_state_arr, signals_strat, prices, aux, regime_id,
    fee: float, N: int, strat_filter: int = -1,
):
    n_bars = len(prices)
    K_strat = signals_strat.shape[1]
    equity = 1.0; peak = 1.0
    eq_arr = np.full(n_bars, 1.0, dtype=np.float64)
    trade_pnls = []
    trade_durs = []
    rl_exits = window_exits = eod_exits = 0

    t = 0
    while t < n_bars - 2:
        k_use = -1
        while t < n_bars - 2:
            if strat_filter >= 0:
                if signals_strat[t, strat_filter] != 0:
                    k_use = strat_filter; break
            else:
                for k in range(K_strat):
                    if signals_strat[t, k] != 0:
                        k_use = k; break
                if k_use >= 0:
                    break
            t += 1
        if k_use < 0:
            break

        direction = int(signals_strat[t, k_use])
        pnl, n_held, reason = simulate_fixed_episode(
            base_state_arr, prices, aux, regime_id,
            entry_bar=t + 1, direction=direction, fee=fee, N=N,
            policy_fn=policy_fn, transitions_out=None,
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
        if   reason == "RL_EXIT": rl_exits     += 1
        elif reason == "WINDOW":  window_exits += 1
        elif reason == "EOD":     eod_exits    += 1

        t = t_close + 1

    rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-12)
    sharpe = (rets.mean() / rets.std() * np.sqrt(525_960)
              if rets.std() > 1e-12 else 0.0)
    pnls_arr = np.array(trade_pnls, dtype=np.float64)
    win_rate = (pnls_arr > 0).mean() if len(pnls_arr) else 0.0
    eq_dd = eq_arr / np.maximum.accumulate(eq_arr)
    max_dd = float(eq_dd.min() - 1.0) if len(eq_arr) else 0.0

    return dict(
        sharpe       = float(sharpe),
        equity_final = float(equity),
        equity_peak  = float(peak),
        max_dd       = max_dd,
        n_trades     = int(len(pnls_arr)),
        win_rate     = float(win_rate),
        mean_pnl_pct = float(pnls_arr.mean() * 100) if len(pnls_arr) else 0.0,
        mean_duration= float(np.mean(trade_durs)) if trade_durs else 0.0,
        rl_exit_pct  = float(rl_exits / max(1, len(pnls_arr)) * 100),
        exit_breakdown = dict(RL_EXIT=int(rl_exits), WINDOW=int(window_exits),
                                EOD=int(eod_exits)),
    )


# ── main ─────────────────────────────────────────────────────────────────────

def train(ticker: str = "btc", seed: int = 42, tag: str = "B5_fix120_S0",
           fee: float = 0.0, strat_filter: int = -1, N: int = DEFAULT_WINDOW_N):
    torch.manual_seed(seed); np.random.seed(seed)
    rng_warm = np.random.default_rng(seed)
    rng_eps  = np.random.default_rng(seed + 1)

    t_start = time.perf_counter()
    label_strat = STRAT_KEYS[strat_filter] if strat_filter >= 0 else "ALL"
    print(f"\n{'='*78}\n  GROUP B5 — FIXED-WINDOW EXIT DQN  ({ticker.upper()})\n"
          f"  tag={tag}  fee={fee:.4f}  strat={label_strat}  window N={N} bars\n"
          f"  state_dim={EXIT_STATE_DIM}  γ={GAMMA}  rule_exits=DISABLED\n{'='*78}")

    # ── load arrays ─────────────────────────────────────────────────────────
    sp_tr = np.load(CACHE / f"{ticker}_dqn_state_train.npz")
    sp_v  = np.load(CACHE / f"{ticker}_dqn_state_val.npz")
    print(f"  DQN-train: {sp_tr['state'].shape}   DQN-val: {sp_v['state'].shape}")

    aux_tr = precompute_aux_arrays(sp_tr["price"], sp_tr["ts"])
    aux_v  = precompute_aux_arrays(sp_v["price"],  sp_v["ts"])
    print(f"  precomputed |log_ret| stats train: med={aux_tr['abs_log_ret_med']:.5f} "
          f"iqr={aux_tr['abs_log_ret_iqr']:.5f}")

    # ── networks + buffer ───────────────────────────────────────────────────
    online = FixedExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=96)
    target = FixedExitDQN(EXIT_STATE_DIM, N_ACTIONS, hidden=96)
    target.load_state_dict(online.state_dict()); target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    buf = ReplayBuffer(capacity=BUFFER_SIZE, state_dim=EXIT_STATE_DIM,
                        n_actions=N_ACTIONS)
    print(f"  FixedExitDQN params: {online.n_params():,}")

    # ── baseline (always-HOLD = always run to window edge) ──────────────────
    print(f"\n  Baseline (always-HOLD = run to N={N} bar window edge) on DQN-val:")
    t0 = time.perf_counter()
    base = evaluate_policy(
        _AlwaysHold(), sp_v["state"], sp_v["signals"], sp_v["price"],
        aux_v, sp_v["regime_id"], fee=fee, N=N, strat_filter=strat_filter)
    print(f"    Sharpe {base['sharpe']:+.3f}  trades {base['n_trades']:,}  "
          f"win {base['win_rate']*100:.1f}%  meanPnL {base['mean_pnl_pct']:+.3f}%  "
          f"eq {base['equity_final']:.3f}  [{time.perf_counter()-t0:.1f}s]")
    print(f"    exit reasons: {base['exit_breakdown']}")

    # ── warmup ──────────────────────────────────────────────────────────────
    print(f"\n  Warmup: {WARMUP_STEPS:,} HOLD-biased random transitions ...")
    cursor_tr = dict(t=0)
    rand_pol = _UniformRandomFixed(rng_warm)
    t_warm = time.perf_counter()
    while len(buf) < WARMUP_STEPS:
        info = rollout_chunk_fixed(
            sp_tr["state"], sp_tr["signals"], sp_tr["price"], aux_tr, sp_tr["regime_id"],
            policy_fn=rand_pol, buffer=buf, cursor=cursor_tr,
            max_transitions=min(2000, WARMUP_STEPS - len(buf)),
            fee=fee, N=N, strat_filter=strat_filter,
        )
        cursor_tr = dict(t=info["t"])
    n_done = int(buf.done[:len(buf)].sum())
    print(f"    buffer={len(buf):,}  ({n_done:,} terminal) "
          f"[{time.perf_counter()-t_warm:.1f}s]")

    # ── training loop ───────────────────────────────────────────────────────
    step_ref = {"step": 0}
    eps_pol  = _EpsilonGreedyFixed(online, rng_eps, step_ref)

    history = []
    best_val_sharpe = -np.inf
    best_step       = 0
    losses          = []

    print(f"\n  Training: {TOTAL_GRAD_STEPS:,} grad steps  "
          f"refresh M={REFRESH_M}/{REFRESH_EVERY}  val every {VAL_EVERY}\n")
    t_loop = time.perf_counter()

    for step in range(1, TOTAL_GRAD_STEPS + 1):
        step_ref["step"] = step

        if step % REFRESH_EVERY == 0:
            info = rollout_chunk_fixed(
                sp_tr["state"], sp_tr["signals"], sp_tr["price"], aux_tr, sp_tr["regime_id"],
                policy_fn=eps_pol, buffer=buf, cursor=cursor_tr,
                max_transitions=REFRESH_M, fee=fee, N=N, strat_filter=strat_filter,
            )
            cursor_tr = dict(t=info["t"])

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
            greedy = _GreedyFixed(online)
            val = evaluate_policy(
                greedy, sp_v["state"], sp_v["signals"], sp_v["price"],
                aux_v, sp_v["regime_id"], fee=fee, N=N, strat_filter=strat_filter)
            improved = val["sharpe"] > best_val_sharpe
            marker = "★" if improved else " "
            mean_loss = float(np.mean(losses[-VAL_EVERY:]))
            elapsed = time.perf_counter() - t_loop
            print(f"  [step {step:>6,}] {marker} ε={epsilon(step):.3f}  "
                  f"loss={mean_loss:.4f}  Sharpe={val['sharpe']:>+6.3f} "
                  f"(rule-only {base['sharpe']:+.3f})  "
                  f"trd={val['n_trades']:>4} win={val['win_rate']*100:.0f}% "
                  f"eq={val['equity_final']:.3f} dd={val['max_dd']*100:>+5.1f}% "
                  f"RLexit={val['rl_exit_pct']:.0f}%  [{elapsed:>4.0f}s+{time.perf_counter()-t_eval:.1f}s]")
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
                            CACHE / f"{ticker}_exit_dqn_fixed_policy_{tag}.pt")
            if step - best_step > EARLY_STOP_PATIENCE:
                print(f"\n  Early stop at step {step} "
                      f"(best Sharpe={best_val_sharpe:+.3f} @ {best_step:,})")
                break

    elapsed_total = time.perf_counter() - t_start

    print(f"\n\n{'='*78}\n  B5 TRAINING SUMMARY  ({tag})\n{'='*78}")
    print(f"  total time           : {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"  best val Sharpe      : {best_val_sharpe:+.3f}  (step {best_step:,})")
    print(f"  rule-only baseline   : {base['sharpe']:+.3f}")
    print(f"  Δ vs baseline        : {best_val_sharpe - base['sharpe']:+.3f}")
    print(f"  policy saved to      : cache/{ticker}_exit_dqn_fixed_policy_{tag}.pt")

    hist_path = CACHE / f"{ticker}_exit_dqn_fixed_history_{tag}.json"
    hist_path.write_text(json.dumps(dict(
        ticker=ticker, run_at=datetime.utcnow().isoformat(),
        tag=tag, fee=fee, strat_filter=strat_filter, N=N,
        baseline=base,
        config=dict(
            gamma=GAMMA, lr=LR, batch=BATCH_SIZE, buffer=BUFFER_SIZE,
            warmup=WARMUP_STEPS, total_steps=TOTAL_GRAD_STEPS,
            refresh_every=REFRESH_EVERY, refresh_m=REFRESH_M,
            target_sync_every=TARGET_SYNC_EVERY, val_every=VAL_EVERY,
            patience=EARLY_STOP_PATIENCE,
            eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY_STEPS,
            per_alpha=PER_ALPHA, reward_scale=REWARD_SCALE,
            state_dim=EXIT_STATE_DIM, n_actions=N_ACTIONS,
            window_N=N, random_exit_prob=RANDOM_EXIT_PROB,
        ),
        best_val_sharpe=best_val_sharpe, best_step=best_step,
        elapsed_seconds=elapsed_total,
        history=history,
    ), indent=2, default=str))
    print(f"  history saved to     : {hist_path.name}")
    return dict(best_val_sharpe=best_val_sharpe, best_step=best_step,
                 baseline=base, history=history, elapsed=elapsed_total)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--tag", default="B5_fix120_fee0_S0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee", type=float, default=0.0)
    ap.add_argument("--strat-filter", type=int, default=-1, dest="strat_filter")
    ap.add_argument("--window-n", type=int, default=DEFAULT_WINDOW_N, dest="window_n")
    args = ap.parse_args()
    train(args.ticker, seed=args.seed, tag=args.tag, fee=args.fee,
           strat_filter=args.strat_filter, N=args.window_n)
