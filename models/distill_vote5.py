"""
Path C2 — train single net to mimic VOTE5_v8_H256_DD plurality vote.

Supervised distillation: student DuelingDQN (same arch as teachers) trained
on per-bar teacher action labels with masked cross-entropy. Stratified
sampling balances the ~95% NO_TRADE vs ~5% trade classes.

Validation: action accuracy on val split + masked-argmax greedy Sharpe at
periodic intervals (rule-based exits, fee=0).

Best checkpoint by val Sharpe saved to cache/btc_dqn_policy_DISTILL_v8_seed{N}.pt.

Run:
  python3 -m models.distill_vote5 --seed 42 --tag DISTILL_v8
"""
import argparse, json, pathlib, time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

from models.dqn_network import DuelingDQN
from models.dqn_rollout import _build_exit_arrays
from models.dqn_selector import evaluate_policy

CACHE = pathlib.Path("cache")
STATE_SUFFIX = "_v8_s11s13"

# hyperparameters
BATCH_SIZE     = 256
LR             = 1e-3
EPOCHS         = 12
VAL_EVERY      = 1                  # epochs between full validation
GRAD_CLIP      = 10.0
SOFT_WEIGHT    = 0.0                # set >0 to add MSE-to-q_mean term


def stratified_batch_indices(actions_tr: np.ndarray, batch: int, rng) -> np.ndarray:
    """Half NO_TRADE, half trade. Compensates for ~95/5 imbalance."""
    no_idx = np.where(actions_tr == 0)[0]
    tr_idx = np.where(actions_tr != 0)[0]
    half = batch // 2
    a = rng.choice(no_idx, size=half, replace=False)
    b = rng.choice(tr_idx, size=batch - half, replace=len(tr_idx) < batch - half)
    out = np.concatenate([a, b]); rng.shuffle(out)
    return out


def train_student(seed: int, tag: str, soft_weight: float = SOFT_WEIGHT):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed + 1000)
    t0 = time.perf_counter()

    print(f"\n{'='*70}\n  C2 self-distillation — student tag={tag}  seed={seed}\n"
          f"  soft_weight={soft_weight}  (0 = pure CE; >0 mixes MSE-to-Q_teacher)\n{'='*70}")

    sp_tr = np.load(CACHE / "state" / f"btc_dqn_state_train{STATE_SUFFIX}.npz")
    sp_v  = np.load(CACHE / "state" / f"btc_dqn_state_val{STATE_SUFFIX}.npz")
    tgt_tr = np.load(CACHE / "distill" / "btc_distill_targets_train_v8.npz")
    tgt_v  = np.load(CACHE / "distill" / "btc_distill_targets_val_v8.npz")

    s_tr = torch.from_numpy(sp_tr["state"]).float()
    v_tr = torch.from_numpy(sp_tr["valid_actions"]).bool()
    v_tr[:, 0] = True
    y_tr = torch.from_numpy(tgt_tr["action"]).long()
    q_tr = torch.from_numpy(tgt_tr["q_mean"]).float() if soft_weight > 0 else None
    s_v  = torch.from_numpy(sp_v["state"]).float()
    v_v  = torch.from_numpy(sp_v["valid_actions"]).bool()
    v_v[:, 0] = True
    y_v  = torch.from_numpy(tgt_v["action"]).long()

    actions_tr_np = tgt_tr["action"]
    n_no = int((actions_tr_np == 0).sum())
    n_tr = int((actions_tr_np != 0).sum())
    print(f"  train: {len(s_tr):,} bars  NO_TRADE={n_no:,}  trade={n_tr:,}")
    print(f"  val  : {len(s_v):,} bars   NO_TRADE={int((y_v==0).sum()):,}  "
          f"trade={int((y_v!=0).sum()):,}")

    student = DuelingDQN(52, 12, 256)
    print(f"  student: DuelingDQN(52, 12, 256)  params={student.n_params():,}")
    opt = torch.optim.Adam(student.parameters(), lr=LR)

    # build val exit arrays for greedy Sharpe eval
    vol = np.load(CACHE / "preds" / "btc_pred_vol_v4.npz")
    atr_median = float(vol["atr_train_median"])
    v_tp, v_sl, v_tr_, v_tab, v_be, v_ts = _build_exit_arrays(
        sp_v["price"], sp_v["atr"], atr_median)

    # iteration budget: pass roughly the "trade" examples 4× per epoch
    steps_per_epoch = max(1, (n_tr * 4) // (BATCH_SIZE // 2))
    print(f"  steps_per_epoch = {steps_per_epoch:,}  (trade-class oversampling)")

    best_val_sharpe = -np.inf
    best_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        student.train()
        losses, ce_losses, sl_losses = [], [], []
        for _ in range(steps_per_epoch):
            idx = stratified_batch_indices(actions_tr_np, BATCH_SIZE, rng)
            sb = s_tr[idx]
            vb = v_tr[idx]
            yb = y_tr[idx]

            logits = student(sb).masked_fill(~vb, -1e9)
            ce = F.cross_entropy(logits, yb)
            loss = ce
            if soft_weight > 0:
                qt = q_tr[idx]
                mse = F.mse_loss(student(sb), qt)
                loss = ce + soft_weight * mse
                sl_losses.append(float(mse.item()))
            ce_losses.append(float(ce.item()))
            losses.append(float(loss.item()))

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
            opt.step()

        if epoch % VAL_EVERY == 0:
            student.eval()
            with torch.no_grad():
                # action accuracy on val
                preds = []
                for i in range(0, len(s_v), 4096):
                    q = student(s_v[i:i+4096]).masked_fill(~v_v[i:i+4096], -1e9)
                    preds.append(q.argmax(dim=1))
                preds = torch.cat(preds)
                acc_all = (preds == y_v).float().mean().item()
                # trade-class accuracy
                tr_mask = y_v != 0
                acc_trade = (preds[tr_mask] == y_v[tr_mask]).float().mean().item() if tr_mask.any() else 0.0

            # greedy Sharpe eval on val (rule-based exits)
            val = evaluate_policy(
                student, sp_v["state"], sp_v["valid_actions"],
                sp_v["signals"], sp_v["price"],
                v_tp, v_sl, v_tr_, v_tab, v_be, v_ts, fee=0.0,
            )

            improved = val["sharpe"] > best_val_sharpe
            marker = "★" if improved else " "
            elapsed = time.perf_counter() - t0
            print(f"  epoch {epoch:>2}{marker} loss={np.mean(losses):.4f}  ce={np.mean(ce_losses):.4f}  "
                  f"acc_all={acc_all*100:>5.1f}%  acc_trade={acc_trade*100:>5.1f}%  "
                  f"val_sharpe={val['sharpe']:>+6.3f}  trades={val['n_trades']:>4}  "
                  f"eq={val['equity_final']:.3f}  [{elapsed:.0f}s]")

            history.append(dict(
                epoch=epoch, mean_loss=float(np.mean(losses)),
                ce_loss=float(np.mean(ce_losses)),
                acc_all=acc_all, acc_trade=acc_trade,
                val_sharpe=val["sharpe"], val_trades=val["n_trades"],
                val_equity=val["equity_final"],
            ))

            if improved:
                best_val_sharpe = val["sharpe"]
                best_acc = acc_all
                best_epoch = epoch
                torch.save(student.state_dict(), CACHE / "policies" / f"btc_dqn_policy_{tag}_seed{seed}.pt")

    elapsed = time.perf_counter() - t0
    print(f"\n  best val Sharpe={best_val_sharpe:+.3f} at epoch {best_epoch} "
          f"(acc={best_acc*100:.1f}%)  [{elapsed:.0f}s]")
    print(f"  policy → cache/btc_dqn_policy_{tag}_seed{seed}.pt")

    hist_path = CACHE / "policies" / f"btc_dqn_train_history_{tag}_seed{seed}.json"
    hist_path.write_text(json.dumps(dict(
        tag=tag, seed=seed, run_at=datetime.utcnow().isoformat(),
        soft_weight=soft_weight,
        best_val_sharpe=best_val_sharpe, best_epoch=best_epoch, best_acc=best_acc,
        elapsed_seconds=elapsed, history=history,
    ), indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag",  default="DISTILL_v8")
    ap.add_argument("--soft-weight", type=float, default=0.0, dest="soft_weight",
                    help="weight on MSE-to-teacher-Q (0 = pure CE distillation)")
    args = ap.parse_args()
    train_student(args.seed, args.tag, args.soft_weight)
