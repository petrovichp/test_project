"""
Quantile-Regression DQN (QR-DQN) network — outputs a discrete distribution
of Q(s, a) over N quantiles per action.

Paper: Distributional Reinforcement Learning with Quantile Regression
(Dabney et al. 2017). https://arxiv.org/abs/1710.10044

Topology (same shared trunk as DuelingDQN, output expanded to N quantiles):
  state(state_dim) → fc1(hidden) → fc2(hidden/2) → fc3(n_actions × n_quantiles)

The forward returns a (B, n_actions, n_quantiles) tensor. The mean over
quantiles is the standard Q-value; the lower-tail mean is CVaR.

We use a Dueling decomposition over quantiles too (V over quantiles
+ A over (actions, quantiles) - mean A over actions), following common QR
implementations. Gives more stable value/advantage separation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QRDuelingDQN(nn.Module):
    """Quantile-Regression Dueling DQN."""

    def __init__(self, state_dim: int = 52, n_actions: int = 12,
                  hidden: int = 256, n_quantiles: int = 32):
        super().__init__()
        self.state_dim   = state_dim
        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        # Heads output one value per quantile (and per action for A)
        self.v_head = nn.Linear(hidden // 2, n_quantiles)
        self.a_head = nn.Linear(hidden // 2, n_actions * n_quantiles)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_actions, n_quantiles)."""
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        v = self.v_head(x).view(-1, 1, self.n_quantiles)                      # (B, 1, Q)
        a = self.a_head(x).view(-1, self.n_actions, self.n_quantiles)         # (B, A, Q)
        q = v + a - a.mean(dim=1, keepdim=True)                                # (B, A, Q)
        return q

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def quantile_huber_loss(theta: torch.Tensor, target_theta: torch.Tensor,
                          tau: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile Huber loss between online quantiles `theta` and target
    quantiles `target_theta`.

    theta:        (B, N)   — online quantile estimates for the taken action
    target_theta: (B, N')  — target distribution (Bellman target)
    tau:          (N,)     — quantile fractions for `theta`, in (0, 1)
    kappa:        Huber threshold (paper uses 1.0)

    Returns scalar loss (mean over batch).
    """
    # u[b, i, j] = target_theta[b, j] - theta[b, i]
    u = target_theta.unsqueeze(1) - theta.unsqueeze(2)            # (B, N, N')
    abs_u = u.abs()
    huber = torch.where(abs_u <= kappa,
                         0.5 * u.pow(2),
                         kappa * (abs_u - 0.5 * kappa))           # (B, N, N')
    tau_b = tau.view(1, -1, 1).to(theta.device)                   # (1, N, 1)
    loss = (tau_b - (u < 0).float()).abs() * huber / kappa        # (B, N, N')
    # Sum over target quantile dim, mean over source quantile and batch.
    return loss.mean(dim=2).sum(dim=1).mean()


def cvar_action(net: QRDuelingDQN, state: torch.Tensor,
                 valid_mask: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    """CVaR-α action: average of the bottom α fraction of quantiles per action.
    Then argmax over actions.

    alpha = 1.0 → standard mean-Q greedy.
    alpha < 1.0 → more risk-averse (worse-case tail averaging).
    """
    with torch.no_grad():
        q = net(state)                                            # (B, A, Q)
        n_q = q.size(-1)
        k = max(1, int(round(alpha * n_q)))
        # Sort along quantile dim ascending; take lowest k
        sorted_q, _ = q.sort(dim=-1)
        cvar = sorted_q[..., :k].mean(dim=-1)                     # (B, A)
        cvar = cvar.masked_fill(~valid_mask, -1e9)
        return cvar.argmax(dim=-1)


def mean_q_action(net: QRDuelingDQN, state: torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
    """Standard greedy: argmax over mean Q across quantiles."""
    with torch.no_grad():
        q = net(state).mean(dim=-1)                               # (B, A)
        q = q.masked_fill(~valid_mask, -1e9)
        return q.argmax(dim=-1)
