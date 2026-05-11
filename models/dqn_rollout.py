"""
Rollout driver — env-like loop that walks through DQN-train bars, picks an
action per decision point, simulates trades via single_trade.simulate_one_trade,
and emits transitions into a replay buffer.

Phase 3.2: random-policy rollout, no learning, used to verify data flow:
  - action distribution matches mask coverage
  - reward histogram looks bimodal (TP wins / SL losses) plus a NO_TRADE spike at 0
  - throughput is fast enough for Phase 3.3 ε-greedy rollout
  - buffer fills smoothly across the DQN-train pass

Pre-computed per-bar arrays (built once outside the loop):
  signals[t, k]     ∈ {-1, 0, +1}   — direction sign for strategy k
  tp_arr[t, k]                       — ATR-scaled take-profit fraction
  sl_arr[t, k]                       — ATR-scaled stop-loss fraction
  trail_arr[t, k]                    — immediate trailing SL pct (always 0 for ComboExit)
  tab_arr[t, k]                      — trail-after-breakeven pct
  be_arr[t, k]                       — breakeven trigger pct
  ts_arr[t, k]                       — time stop (bars)

Stateful state-vector fields (indices 18, 19) are filled in here:
  state[t, 18] = last_trade_pnl_pct   * 100 → standardized, clip [-10, 10]
  state[t, 19] = current_dd_pct       * 100 → standardized, clip [-20, 0]

Equity tracking uses no position-size scaling (DQN reward = raw trade PnL)
to match the Phase 3 reward function `r = trade.pnl_pct - 2·TAKER_FEE`
(fees are already in `pnl_pct` from `simulate_one_trade`).

Run: python3 -m models.dqn_rollout [ticker]
"""

import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.loader            import load_meta
from strategy.agent         import STRATEGIES
from execution.config       import EXECUTION_CONFIG
from backtest.single_trade  import simulate_one_trade, EXIT_NAMES
from models.dqn_replay      import ReplayBuffer

CACHE        = ROOT / "cache"
WARMUP       = 1440
VOL_TRAIN_E  = 101_440
DQN_TRAIN_E  = 281_440
DQN_VAL_E    = 332_307
STRAT_KEYS   = ["S1_VolDir", "S2_Funding", "S3_BBRevert", "S4_MACDTrend",
                "S6_TwoSignal", "S7_OIDiverg", "S8_TakerFlow",
                "S10_Squeeze", "S12_VWAPVol",
                # Z3 Step 4 additions (2026-05-11):
                "S11_Basis", "S13_OBDiv"]


# ── exit-array builder ───────────────────────────────────────────────────────

def _build_exit_arrays(prices: np.ndarray, atr_arr: np.ndarray,
                        atr_train_median: float):
    """Build (n_bars, 9) arrays of tp, sl, trail, tab, be, ts for each strategy.

    Mirrors backtest/run.py:319-385. Uses ComboExit.arrays() for ATR scaling
    plus ComboExit.plan() at median ATR for static fields (trail/tab/be/ts)."""
    n_bars = len(prices)
    K      = len(STRAT_KEYS)
    tp     = np.zeros((n_bars, K), dtype=np.float32)
    sl     = np.zeros((n_bars, K), dtype=np.float32)
    trail  = np.zeros((n_bars, K), dtype=np.float32)
    tab    = np.zeros((n_bars, K), dtype=np.float32)
    be     = np.zeros((n_bars, K), dtype=np.float32)
    ts     = np.zeros((n_bars, K), dtype=np.int32)

    for k, key in enumerate(STRAT_KEYS):
        cfg = EXECUTION_CONFIG[key]
        if hasattr(cfg.exit, "arrays"):
            tp_arr, sl_arr = cfg.exit.arrays(atr_arr, prices, atr_train_median)
            plan0 = cfg.exit.plan(atr_train_median, float(np.median(prices)),
                                    atr_train_median)
            tp[:, k]    = tp_arr
            sl[:, k]    = sl_arr
            trail[:, k] = 0.0
            tab[:, k]   = plan0.tab_pct
            be[:, k]    = plan0.breakeven_pct
            ts[:, k]    = plan0.time_stop_bars
        else:
            # fallback (not used by current EXECUTION_CONFIG)
            tp[:, k]    = 0.020
            sl[:, k]    = 0.007
    return tp, sl, trail, tab, be, ts


# ── generalized chunked rollout (used by 3.2 random + 3.3 ε-greedy) ─────────

def rollout_chunk(
    state, valid, signals_strat, prices,
    tp, sl, trail, tab, be, ts_bars,
    policy_fn,                       # callable(state_t (50,), valid_t (10,)) -> int
    buffer, cursor: dict,
    max_transitions: int,
    reward_scale: float = 1.0,       # multiplier for buffer reward only (eq tracking uses raw)
    valid_mask_override: np.ndarray = None,   # optional (n_actions,) bool to AND with per-bar mask
    fee: float = None,               # if None, uses TAKER_FEE; else parameterized
    trade_penalty: float = 0.0,      # fixed cost subtracted from buffer reward per trade entry
):
    """Continuous rollout that pushes up to `max_transitions` new transitions
    into `buffer` and returns updated cursor (t, equity, peak, last_pnl).

    cursor wraps to t=0 (and resets episode equity) when reaching end of bars.
    """
    # lazy import to avoid circular dependency
    from models.diagnostics_ab import _simulate_one_trade_fee
    from backtest.costs        import TAKER_FEE
    if fee is None:
        fee = TAKER_FEE

    n_bars   = len(state)
    t        = cursor["t"]
    equity   = cursor["equity"]
    peak     = cursor["peak"]
    last_pnl = cursor["last_pnl"]

    n_pushed = 0
    chunk_rewards: list = []
    chunk_trades = 0
    chunk_actions = np.zeros(valid.shape[1], dtype=np.int64)

    while n_pushed < max_transitions:
        if t >= n_bars - 2:
            t = 0
            equity, peak, last_pnl = 1.0, 1.0, 0.0

        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        valid_t = valid[t] if valid_mask_override is None else (valid[t] & valid_mask_override)
        if not valid_t.any():
            valid_t = valid[t].copy()
            valid_t[1:] = False              # ensure NO_TRADE is the fallback
            valid_t[0]  = True
        action  = int(policy_fn(s_t, valid_t))
        chunk_actions[action] += 1

        if action == 0:
            raw_pnl, duration, t_next = 0.0, 1, t + 1
        else:
            k = action - 1
            direction = int(signals_strat[t, k])
            if direction == 0:
                raw_pnl, duration, t_next = 0.0, 1, t + 1
            else:
                pnl, n_held = _simulate_one_trade_fee(
                    prices, t + 1, direction,
                    float(tp[t, k]), float(sl[t, k]),
                    float(trail[t, k]), float(tab[t, k]),
                    float(be[t, k]),   int(ts_bars[t, k]),
                    0, fee,
                )
                raw_pnl  = float(pnl)
                duration = int(n_held + 1)
                t_next   = t + duration + 1
                chunk_trades += 1

        if action != 0:
            equity   *= (1.0 + raw_pnl)      # equity uses raw PnL (with fee already applied)
            peak      = max(peak, equity)
            last_pnl  = raw_pnl
        chunk_rewards.append(raw_pnl)        # diagnostics in raw units

        done = t_next >= n_bars - 1
        if done:
            t_next = n_bars - 1
        s_next = state[t_next].copy()
        s_next[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_next[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        v_next = valid[t_next] if valid_mask_override is None else (valid[t_next] & valid_mask_override)
        v_next[0] = True                      # NO_TRADE always valid in next-state mask too

        # Buffer reward = scaled raw PnL minus fixed per-trade penalty (action != 0 only).
        # The penalty is a learning-signal modifier, NOT a real cost (equity tracks raw_pnl above).
        buffer_reward = raw_pnl * reward_scale
        if action != 0:
            buffer_reward -= trade_penalty
        buffer.push(s_t, action, buffer_reward, duration, s_next, v_next, done)
        n_pushed += 1

        if done:
            t = 0
            equity, peak, last_pnl = 1.0, 1.0, 0.0
        else:
            t = t_next

    return dict(t=t, equity=equity, peak=peak, last_pnl=last_pnl,
                 n_pushed=n_pushed, rewards=chunk_rewards, trades=chunk_trades,
                 actions=chunk_actions)


# ── rollout driver ───────────────────────────────────────────────────────────

def random_rollout(
    state:        np.ndarray,         # (n_bars, 50)  base state (stateful 18,19 = 0)
    valid:        np.ndarray,         # (n_bars, 10)  bool action mask
    signals_strat:np.ndarray,         # (n_bars, 9)   {-1, 0, +1}
    prices:       np.ndarray,         # (n_bars,)
    tp:           np.ndarray,         # (n_bars, 9)
    sl:           np.ndarray,
    trail:        np.ndarray,
    tab:          np.ndarray,
    be:           np.ndarray,
    ts_bars:      np.ndarray,
    buffer:       ReplayBuffer,
    seed:         int = 42,
    verbose:      bool = True,
):
    """Walk through DQN-train bars with a uniform-random policy over valid
    actions. Push transitions into `buffer`. Returns diagnostics."""
    rng       = np.random.default_rng(seed)
    n_bars    = len(state)
    state_dim = state.shape[1]
    n_actions = valid.shape[1]

    action_counts = np.zeros(n_actions, dtype=np.int64)
    rewards_log   = []
    durations_log = []
    exit_reasons  = np.zeros(6, dtype=np.int64)

    equity   = 1.0
    peak     = 1.0
    last_pnl = 0.0

    n_pushed = 0
    t = 0
    t0 = time.perf_counter()

    while t < n_bars - 2:
        # ── stateful fields update on s_t ─────────────────────────────────────
        s_t = state[t].copy()
        s_t[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_t[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))

        # ── pick action uniformly from valid set ─────────────────────────────
        valid_t    = valid[t]
        valid_idxs = np.where(valid_t)[0]
        action     = int(rng.choice(valid_idxs))
        action_counts[action] += 1

        # ── act ──────────────────────────────────────────────────────────────
        if action == 0:                                          # NO_TRADE
            reward    = 0.0
            duration  = 1
            t_next    = t + 1
        else:
            k = action - 1
            direction = int(signals_strat[t, k])
            if direction == 0:
                # Defensive: should not happen since valid mask filters this.
                reward, duration, t_next = 0.0, 1, t + 1
            else:
                pnl, n_held, exit_id = simulate_one_trade(
                    prices, t + 1, direction,
                    float(tp[t, k]), float(sl[t, k]),
                    float(trail[t, k]), float(tab[t, k]),
                    float(be[t, k]),   int(ts_bars[t, k]),
                    0,                                            # no max_lookahead cap
                )
                reward    = float(pnl)
                duration  = int(n_held + 1)                       # 1 entry-lag bar + n_held
                t_next    = t + duration + 1
                exit_reasons[exit_id] += 1

        # ── update accumulators ──────────────────────────────────────────────
        if action != 0:
            equity   *= (1.0 + reward)
            peak      = max(peak, equity)
            last_pnl  = reward
        rewards_log.append(reward)
        durations_log.append(duration)

        # ── bound t_next; build s_next; push ─────────────────────────────────
        done = t_next >= n_bars - 1
        if done:
            t_next = n_bars - 1
        s_next = state[t_next].copy()
        s_next[18] = float(np.clip(last_pnl       * 100.0, -10.0, 10.0))
        s_next[19] = float(np.clip((equity - peak) / peak * 100.0, -20.0, 0.0))
        v_next = valid[t_next]

        buffer.push(s_t, action, reward, duration, s_next, v_next, done)
        n_pushed += 1

        if verbose and n_pushed % 25_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    t={t:>7,}/{n_bars-2:,}  pushed={n_pushed:,}  "
                  f"equity={equity:.4f}  dd={(equity-peak)/peak*100:>+6.3f}%  "
                  f"[{elapsed:.1f}s, {n_pushed/elapsed:,.0f}/s]")

        if done:
            break
        t = t_next

    elapsed = time.perf_counter() - t0
    return dict(
        n_pushed       = n_pushed,
        action_counts  = action_counts,
        rewards        = np.array(rewards_log,   dtype=np.float32),
        durations      = np.array(durations_log, dtype=np.int32),
        exit_reasons   = exit_reasons,
        elapsed        = elapsed,
        final_equity   = equity,
        final_peak     = peak,
    )


# ── diagnostics ──────────────────────────────────────────────────────────────

def _diagnostics(stats: dict, ticker: str):
    n         = stats["n_pushed"]
    rewards   = stats["rewards"]
    durations = stats["durations"]
    cnt       = stats["action_counts"]
    er        = stats["exit_reasons"]

    print(f"\n  ── ROLLOUT DIAGNOSTICS ──")
    print(f"    transitions pushed   : {n:,}")
    print(f"    elapsed              : {stats['elapsed']:.1f}s  "
          f"({n/stats['elapsed']:,.0f} transitions/s)")
    print(f"    final equity         : {stats['final_equity']:.4f}  "
          f"(peak {stats['final_peak']:.4f})")

    print(f"\n    Action distribution:")
    actions = ["NO_TRADE"] + STRAT_KEYS
    for i, name in enumerate(actions):
        pct = cnt[i] / n * 100 if n else 0
        bar = "█" * int(pct / 2)
        print(f"      {i:>2} {name:<14} {cnt[i]:>7,}  {pct:>5.2f}%  {bar}")

    nz_rewards = rewards[rewards != 0]
    if len(nz_rewards):
        wins = nz_rewards[nz_rewards > 0]
        losses = nz_rewards[nz_rewards <= 0]
        print(f"\n    Reward histogram (non-zero only, {len(nz_rewards):,} trades):")
        print(f"      wins   : {len(wins):>6,} ({len(wins)/len(nz_rewards)*100:.1f}%)  "
              f"mean={wins.mean()*100:>+.3f}%  max={wins.max()*100:>+.3f}%")
        print(f"      losses : {len(losses):>6,} ({len(losses)/len(nz_rewards)*100:.1f}%)  "
              f"mean={losses.mean()*100:>+.3f}%  min={losses.min()*100:>+.3f}%")
        print(f"      mean trade PnL    = {nz_rewards.mean()*100:>+.4f}%")
        print(f"      total trade PnL   = {nz_rewards.sum()*100:>+.2f}%  (random policy)")
        # crude bins
        edges = np.array([-np.inf, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, np.inf])
        labels = ["<-2%", "-2..-1%", "-1..-0.5%", "-0.5..0%", "0..0.5%", "0.5..1%", "1..2%", ">2%"]
        h, _ = np.histogram(nz_rewards, bins=edges)
        print(f"\n      bin            count       %")
        for lbl, c in zip(labels, h):
            pct = c / len(nz_rewards) * 100
            print(f"      {lbl:<10}    {c:>6,}    {pct:>5.1f}%")

    print(f"\n    Exit reasons (trades only):")
    total_trades = er.sum()
    for i, name in enumerate(EXIT_NAMES):
        if total_trades:
            print(f"      {name:<5}  {er[i]:>6,}  ({er[i]/total_trades*100:>5.1f}%)")

    print(f"\n    Trade duration (bars from decision to next decision):")
    trade_durs = durations[durations > 1]
    if len(trade_durs):
        print(f"      min={trade_durs.min()}  median={int(np.median(trade_durs))}  "
              f"mean={trade_durs.mean():.1f}  max={trade_durs.max()}")


# ── main ─────────────────────────────────────────────────────────────────────

def run(ticker: str = "btc", buffer_size: int = 80_000, seed: int = 42,
        skill_level: str = "random"):
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  PHASE 3.2 — RANDOM-POLICY ROLLOUT  ({ticker.upper()})\n{'='*70}")

    # ── load DQN-train state arrays ──────────────────────────────────────────
    sp = np.load(CACHE / f"{ticker}_dqn_state_train.npz")
    state    = sp["state"]                    # (180000, 50)
    valid    = sp["valid_actions"]            # (180000, 10)
    sigs     = sp["signals"]                  # (180000, 9)  ∈ {-1,0,+1}
    prices   = sp["price"]                    # (180000,)
    atr      = sp["atr"]                      # (180000,)
    print(f"  loaded train: state {state.shape}  valid {valid.shape}  "
          f"prices {prices.shape}")
    n_bars = len(state)

    # ── ATR train-median (from vol_v4 npz) ───────────────────────────────────
    vol = np.load(CACHE / f"{ticker}_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    print(f"  atr_train_median = {atr_median:.4f}")

    # ── pre-compute exit arrays ──────────────────────────────────────────────
    print(f"  building per-bar TP/SL/trail/tab/be/ts arrays ...")
    t1 = time.perf_counter()
    tp_arr, sl_arr, trail_arr, tab_arr, be_arr, ts_arr = _build_exit_arrays(
        prices, atr, atr_median)
    print(f"    done in {time.perf_counter()-t1:.1f}s  "
          f"shapes: tp={tp_arr.shape}")
    print(f"    sample (S1): tp={tp_arr[0,0]:.4f}  sl={sl_arr[0,0]:.4f}  "
          f"tab={tab_arr[0,0]:.4f}  be={be_arr[0,0]:.4f}  ts={ts_arr[0,0]}")

    # ── DQN MLP forward sanity (untrained) ──────────────────────────────────
    print(f"\n  loading + sanity-checking DQN MLP ...")
    import torch
    from models.dqn_network import DQN, masked_argmax
    net = DQN(state_dim=50, n_actions=10, hidden=64)
    print(f"    network params: {net.n_params():,}")
    s_batch = torch.from_numpy(state[:64]).float()
    v_batch = torch.from_numpy(valid[:64])
    q       = net(s_batch)
    a       = masked_argmax(net, s_batch, v_batch)
    print(f"    Q-values shape: {tuple(q.shape)}  "
          f"Q range: [{q.min().item():+.3f}, {q.max().item():+.3f}]")
    print(f"    masked argmax: {a.tolist()[:10]} (first 10)")

    # ── rollout ──────────────────────────────────────────────────────────────
    print(f"\n  initializing replay buffer (capacity={buffer_size:,}) ...")
    buf = ReplayBuffer(capacity=buffer_size, state_dim=50, n_actions=10)

    print(f"\n  running random-policy rollout through DQN-train ({n_bars:,} bars) ...")
    stats = random_rollout(
        state=state, valid=valid, signals_strat=sigs, prices=prices,
        tp=tp_arr, sl=sl_arr, trail=trail_arr, tab=tab_arr,
        be=be_arr, ts_bars=ts_arr, buffer=buf, seed=seed,
    )

    print(f"\n  buffer state: size={len(buf):,} / {buf.capacity:,}  "
          f"({len(buf)/buf.capacity*100:.1f}% full)")

    # ── diagnostics ──────────────────────────────────────────────────────────
    _diagnostics(stats, ticker)

    # ── sanity sample ───────────────────────────────────────────────────────
    print(f"\n  ── BUFFER SAMPLE TEST ──")
    batch, idx, w = buf.sample_uniform(8)
    print(f"    batch state shape   : {batch['state'].shape}")
    print(f"    batch action sample : {batch['action'].tolist()}")
    print(f"    batch reward sample : {[round(float(r), 4) for r in batch['reward']]}")
    print(f"    batch duration smpl : {batch['duration'].tolist()}")
    print(f"    is_weights          : all 1.0 (uniform sampling)")

    print(f"\n  ── PHASE 3.2 GATES ──")
    n_trades  = int(sum(stats["action_counts"][1:]))
    n_no_trd  = int(stats["action_counts"][0])
    pass_thru = (stats["n_pushed"] / stats["elapsed"]) > 5_000
    pass_act  = 0.05 < (n_trades / max(1, stats["n_pushed"])) < 0.95
    pass_rew  = (stats["rewards"] != 0).sum() > 100

    thr = int(stats["n_pushed"] / stats["elapsed"])
    pct = n_trades / stats["n_pushed"] * 100
    nz  = int((stats["rewards"] != 0).sum())
    print(f"    throughput >5k transitions/s : {'✓ ' + str(thr) + '/s'  if pass_thru else '✗ ' + str(thr) + '/s'}")
    print(f"    action mix (5%..95% trades)  : {'✓ ' + f'{pct:.1f}%' if pass_act  else '✗ ' + f'{pct:.1f}%'}")
    print(f"    nonzero-reward count >100    : {'✓ ' + str(nz)        if pass_rew  else '✗ ' + str(nz)}")

    if pass_thru and pass_act and pass_rew:
        print(f"\n  ✓ Phase 3.2 PASS — data flow verified, ready for Phase 3.3")
    else:
        print(f"\n  ✗ Phase 3.2 FAIL — investigate before proceeding to Phase 3.3")

    print(f"\n  total time {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "btc")
