"""
Z4.2 — Transformer-DQN training loop.

Drop-in trainer for TransformerDQN using the same loss + replay machinery
as dqn_selector. Mirrors hyperparameters; differs only in net class.
"""
import argparse, sys, time, json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.transformer_network import TransformerDQN
from models.dqn_network import masked_argmax, masked_max
from models.dqn_replay  import ReplayBuffer
from models.dqn_rollout import rollout_chunk, _build_exit_arrays
from models.dqn_selector import (
    bellman_loss, _UniformRandom, _EpsilonGreedy, evaluate_policy,
    GAMMA, LR, BATCH_SIZE, BUFFER_SIZE, WARMUP_STEPS, TOTAL_GRAD_STEPS,
    REFRESH_EVERY, REFRESH_M, TARGET_SYNC_EVERY, VAL_EVERY,
    EARLY_STOP_PATIENCE, REWARD_SCALE, USE_STRATIFIED_PER, GRAD_CLIP,
    epsilon, per_beta, PER_ALPHA,
)
from config.cache_paths import POLICIES, STATE, PREDS

CACHE = ROOT / "cache"


def train(ticker="btc", seed=42, tag="XFMR_v8", state_version="v8_s11s13",
           d_model=16, n_heads=4, n_layers=2, hidden=128, trade_penalty=0.001):
    torch.manual_seed(seed); np.random.seed(seed)
    rng_warm = np.random.default_rng(seed)
    rng_eps  = np.random.default_rng(seed + 1)
    t_start = time.perf_counter()

    print(f"\n{'='*70}\n  Transformer-DQN — {tag}  seed={seed}\n"
          f"  d_model={d_model}  n_heads={n_heads}  n_layers={n_layers}  hidden={hidden}\n"
          f"  state={state_version}\n{'='*70}")

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

    online = TransformerDQN(state_dim, n_actions, d_model, n_heads, n_layers, hidden)
    target = TransformerDQN(state_dim, n_actions, d_model, n_heads, n_layers, hidden)
    target.load_state_dict(online.state_dict()); target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    buf = ReplayBuffer(capacity=BUFFER_SIZE, state_dim=state_dim, n_actions=n_actions)

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
    eps_pol = _EpsilonGreedy(online, rng_eps, epsilon, step_ref)
    best_val_sharpe = -np.inf; best_step = 0
    history = []
    losses = []

    print(f"  training: max {TOTAL_GRAD_STEPS:,} grad steps")
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
        batch, idx, is_w_np = sampler(BATCH_SIZE, alpha=PER_ALPHA, beta=per_beta(step))
        is_w = torch.from_numpy(is_w_np)
        loss, td_errs = bellman_loss(online, target, batch, GAMMA, is_w, double=True)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
        optimizer.step()
        buf.update_priorities(idx, td_errs)
        losses.append(float(loss.item()))

        if step % TARGET_SYNC_EVERY == 0:
            target.load_state_dict(online.state_dict())

        if step % VAL_EVERY == 0:
            val = evaluate_policy(
                online, sp_v["state"], sp_v["valid_actions"],
                sp_v["signals"], sp_v["price"],
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

    hist_path = POLICIES / f"{ticker}_dqn_train_history_{tag}_seed{seed}.json"
    hist_path.write_text(json.dumps(dict(
        tag=tag, seed=seed, arch="transformer",
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, hidden=hidden,
        run_at=datetime.utcnow().isoformat(),
        best_val_sharpe=best_val_sharpe, best_step=best_step,
        elapsed_seconds=elapsed, history=history,
    ), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?", default="btc")
    ap.add_argument("--tag", default="XFMR_v8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=16, dest="d_model")
    ap.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    ap.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--trade-penalty", type=float, default=0.001, dest="trade_penalty")
    ap.add_argument("--state-version", default="v8_s11s13", dest="state_version")
    args = ap.parse_args()
    train(args.ticker, args.seed, args.tag, args.state_version,
           args.d_model, args.n_heads, args.n_layers, args.hidden, args.trade_penalty)
