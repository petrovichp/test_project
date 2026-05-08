"""
DQN network — 3-layer MLP, ~5,674 params.

Topology (matches v5 spec):
  state(50) → Linear→ReLU(64) → Linear→ReLU(32) → Linear(10) → Q-values

Action masking happens AFTER the network forward, before any argmax/max/softmax:
  q[~valid] = -1e9  (avoids NaN propagation that -inf would cause)

Two networks are used in training: online + target. Hard sync every K steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim: int = 50, n_actions: int = 10, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, n_actions)
        # Kaiming-uniform default fits ReLU; no special init needed.
        # No BatchNorm, no Dropout — small net, off-policy sampling.

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                # raw logits (Q-values), shape (B, n_actions)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DuelingDQN(nn.Module):
    """Dueling architecture: shared trunk → value head V(s) + advantage head A(s,a).
    Output Q(s,a) = V(s) + A(s,a) − mean_a' A(s,a').

    Often more stable than vanilla DQN when the policy is highly selective
    (many states have similar V; advantages discriminate the action choice).
    """
    def __init__(self, state_dim: int = 50, n_actions: int = 10, hidden: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        # shared trunk (same depth as DQN: 50→64→32)
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        # heads
        self.v_head = nn.Linear(hidden // 2, 1)
        self.a_head = nn.Linear(hidden // 2, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        v = self.v_head(x)                                # (B, 1)
        a = self.a_head(x)                                # (B, n_actions)
        return v + a - a.mean(dim=1, keepdim=True)        # (B, n_actions)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class EnsembleDQN(nn.Module):
    """Wraps K DQNs; forward returns averaged Q-values across all members.

    Same call signature as DQN — drop-in for masked_argmax / evaluate_policy.
    """
    def __init__(self, nets: list):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.state_dim = nets[0].state_dim
        self.n_actions = nets[0].n_actions

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        qs = torch.stack([net(s) for net in self.nets], dim=0)   # (K, B, n_actions)
        return qs.mean(dim=0)


def masked_argmax(net: DQN, state: torch.Tensor,
                   valid_actions_mask: torch.Tensor) -> torch.Tensor:
    """Inference path. valid_actions_mask: (B, n_actions) bool."""
    with torch.no_grad():
        q = net(state)
        q = q.masked_fill(~valid_actions_mask, -1e9)
        return q.argmax(dim=-1)


def masked_max(target_net: DQN, state_next: torch.Tensor,
                valid_mask_next: torch.Tensor) -> torch.Tensor:
    """Bellman-target path. Returns max Q over valid actions only."""
    with torch.no_grad():
        q_next = target_net(state_next)
        q_next = q_next.masked_fill(~valid_mask_next, -1e9)
        return q_next.max(dim=-1).values
