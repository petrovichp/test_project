"""
Z4.2 distill — precompute teacher labels from XFMR_v8 K=5 plurality vote.

Mirrors models/distill_targets.py but uses TransformerDQN teachers
(d_model=8, n_heads=2, n_layers=1, hidden=128) instead of the
DuelingDQN H256 VOTE5_v8 teachers.

Output: cache/distill/btc_distill_targets_{split}_xfmr.npz
        (action, votes_count, q_mean) — same schema as the v8 version.

Run: python3 -m models.distill_targets_xfmr
"""
import pathlib, time
from collections import Counter
import numpy as np
import torch

from models.transformer_network import TransformerDQN

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]
STATE_SUFFIX = "_v8_s11s13"


def load_teachers():
    nets = []
    for s in SEEDS:
        net = TransformerDQN(52, 12, d_model=8, n_heads=2, n_layers=1, hidden=128)
        net.load_state_dict(torch.load(
            CACHE / "policies" / f"btc_dqn_policy_XFMR_v8_seed{s}.pt", map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def plurality(votes_row: np.ndarray) -> tuple[int, int]:
    c = Counter(votes_row.tolist())
    top = c.most_common(2)
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return 0, top[0][1]
    return int(top[0][0]), int(top[0][1])


def main():
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Precompute distillation targets (XFMR_v8 teachers)\n{'='*70}")
    nets = load_teachers()
    print(f"  loaded {len(nets)} TransformerDQN teachers")

    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}{STATE_SUFFIX}.npz")
        s = torch.from_numpy(sp["state"]).float()
        v = torch.from_numpy(sp["valid_actions"]).bool()
        v_eff = v.clone()
        v_eff[:, 0] = True

        votes = np.zeros((len(nets), len(s)), dtype=np.int64)
        with torch.no_grad():
            for k, net in enumerate(nets):
                for i in range(0, len(s), 8192):
                    sb = s[i:i+8192]
                    vb = v_eff[i:i+8192]
                    q = net(sb).masked_fill(~vb, -1e9)
                    votes[k, i:i+8192] = q.argmax(dim=1).numpy()

        actions = np.zeros(votes.shape[1], dtype=np.int64)
        votes_count = np.zeros(votes.shape[1], dtype=np.int64)
        for i in range(votes.shape[1]):
            a, c = plurality(votes[:, i])
            actions[i] = a; votes_count[i] = c

        with torch.no_grad():
            qs = torch.zeros(len(s), 12)
            for net in nets:
                for i in range(0, len(s), 8192):
                    qs[i:i+8192] += net(s[i:i+8192])
            qs /= len(nets)
            q_mean = qs.numpy().astype(np.float32)

        out = CACHE / "distill" / f"btc_distill_targets_{split}_xfmr.npz"
        np.savez(out, action=actions, votes_count=votes_count, q_mean=q_mean)

        counts = Counter(actions.tolist())
        no_trade_pct = counts[0] / len(actions) * 100
        trade_pct = 100 - no_trade_pct
        n5 = int((votes_count == 5).sum())
        n4 = int((votes_count == 4).sum())
        n3 = int((votes_count == 3).sum())
        nT = int((votes_count == 2).sum())
        print(f"  {split:<6} n={len(actions):>7,}  NO_TRADE {no_trade_pct:>5.1f}%  "
              f"trade {trade_pct:>5.2f}%  |  votes 5/4/3/tie: {n5}/{n4}/{n3}/{nT}  "
              f"→ {out.name}")

    print(f"\n  total [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
