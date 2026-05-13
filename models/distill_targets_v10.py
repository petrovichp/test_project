"""
Generate distillation targets from VOTE5_v10_H256_DD K=5 plurality vote.

Output: cache/distill/btc_distill_targets_{split}_v10.npz

Run: python3 -m models.distill_targets_v10
"""
import pathlib, time
from collections import Counter
import numpy as np
import torch

from models.dqn_network import DuelingDQN

CACHE = pathlib.Path("cache")
SEEDS = [42, 7, 123, 0, 99]
STATE_SUFFIX = "_v10_regimeoff"


def load_teachers():
    nets = []
    for s in SEEDS:
        net = DuelingDQN(52, 12, 256)
        net.load_state_dict(torch.load(
            CACHE / "policies" / f"btc_dqn_policy_VOTE5_v10_H256_DD_seed{s}.pt",
            map_location="cpu"))
        net.eval()
        nets.append(net)
    return nets


def plurality(votes_row):
    c = Counter(votes_row.tolist())
    top = c.most_common(2)
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return 0, top[0][1]
    return int(top[0][0]), int(top[0][1])


def main():
    t0 = time.perf_counter()
    print(f"\n{'='*70}\n  Distillation targets — VOTE5_v10_H256_DD teachers\n{'='*70}")
    nets = load_teachers()

    for split in ("train", "val", "test"):
        sp = np.load(CACHE / "state" / f"btc_dqn_state_{split}{STATE_SUFFIX}.npz")
        s = torch.from_numpy(sp["state"]).float()
        v = torch.from_numpy(sp["valid_actions"]).bool()
        v_eff = v.clone(); v_eff[:, 0] = True

        votes = np.zeros((len(nets), len(s)), dtype=np.int64)
        with torch.no_grad():
            for k, net in enumerate(nets):
                for i in range(0, len(s), 8192):
                    sb = s[i:i+8192]; vb = v_eff[i:i+8192]
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

        out = CACHE / "distill" / f"btc_distill_targets_{split}_v10.npz"
        np.savez(out, action=actions, votes_count=votes_count, q_mean=q_mean)
        n_no = int((actions == 0).sum()); n_tr = int((actions != 0).sum())
        print(f"  {split:<6} n={len(actions):>7,}  NO_TRADE {n_no/len(actions)*100:>5.1f}%  "
              f"trade {n_tr/len(actions)*100:>5.2f}%  → {out.name}")

    print(f"\n  total [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
