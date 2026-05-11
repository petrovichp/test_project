"""
Path C2 — precompute teacher labels from VOTE5_v8_H256_DD plurality vote.

Saves per-bar argmax action (plurality, tie → 0) for each split. Student net
will be supervised on these in models/distill_vote5.py.

Run: python3 -m models.distill_targets
"""
import pathlib, time
from collections import Counter
import numpy as np
import torch

from models.dqn_network import DuelingDQN

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]
STATE_SUFFIX = "_v8_s11s13"


def load_teachers():
    nets = []
    for s in SEEDS:
        net = DuelingDQN(52, 12, 256)
        net.load_state_dict(torch.load(
            CACHE / f"btc_dqn_policy_VOTE5_v8_H256_DD_seed{s}.pt", map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def plurality(votes_row: np.ndarray) -> tuple[int, int]:
    """votes_row shape (K,). Returns (action, votes_for_top). Tie → 0."""
    c = Counter(votes_row.tolist())
    top = c.most_common(2)
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return 0, top[0][1]
    return int(top[0][0]), int(top[0][1])


def main():
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Precompute distillation targets (VOTE5_v8_H256_DD teachers)\n{'='*70}")
    nets = load_teachers()
    print(f"  loaded {len(nets)} teacher nets (DuelingDQN 52→256→128→12)")

    for split in ("train", "val", "test"):
        sp = np.load(CACHE / f"btc_dqn_state_{split}{STATE_SUFFIX}.npz")
        s = torch.from_numpy(sp["state"]).float()
        v = torch.from_numpy(sp["valid_actions"]).bool()
        # ensure NO_TRADE always valid
        v_eff = v.clone()
        v_eff[:, 0] = True

        votes = np.zeros((len(nets), len(s)), dtype=np.int64)
        with torch.no_grad():
            for k, net in enumerate(nets):
                # chunked to keep memory manageable
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

        # also store ensemble-mean Q-values for soft-distillation option
        with torch.no_grad():
            qs = torch.zeros(len(s), 12)
            for net in nets:
                for i in range(0, len(s), 8192):
                    qs[i:i+8192] += net(s[i:i+8192])
            qs /= len(nets)
            q_mean = qs.numpy().astype(np.float32)

        out = CACHE / f"btc_distill_targets_{split}_v8.npz"
        np.savez(out, action=actions, votes_count=votes_count, q_mean=q_mean)

        counts = Counter(actions.tolist())
        no_trade_pct = counts[0] / len(actions) * 100
        trade_pct = 100 - no_trade_pct
        n_consensus5 = int((votes_count == 5).sum())
        n_consensus4 = int((votes_count == 4).sum())
        n_consensus3 = int((votes_count == 3).sum())
        n_tie = int((votes_count == 2).sum())   # tie → action=0 enforced
        print(f"  {split:<6} n={len(actions):>7,}  NO_TRADE {no_trade_pct:>5.1f}%  "
              f"trade {trade_pct:>5.2f}%  "
              f"|  votes 5/4/3/tie: {n_consensus5}/{n_consensus4}/{n_consensus3}/{n_tie}  "
              f"→ {out.name}")

    print(f"\n  total [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
