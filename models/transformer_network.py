"""
Transformer-DQN — multi-head self-attention over state dimensions.

Hypothesis: the 52-dim state has structure that an MLP treats as a flat
unordered vector. Among those 52 dims:
  [0..3]   direction probabilities (4 tokens)
  [4..6]   vol prediction features (3 tokens)
  [7..15]  9 strategy signal flags (9 tokens)
  [16..17] hour-of-day sin/cos (2 tokens)
  [18..20] equity / drawdown / last-pnl (3 tokens)
  [21..29] 9 per-strategy rank features (9 tokens)
  [30..49] 20 orderbook + microstructure features (20 tokens)
  [50..51] S11/S13 strategy flags (v8 addition, 2 tokens)

A transformer block can learn pairwise interactions between these
sub-groups (e.g. "is the basis dislocation interacting with funding
extreme?"). The MLP must implicitly learn these via fc1's weights.

Architecture: project each scalar feature to d_model=16 with a learned
positional embedding per feature, run 2 layers of multi-head attention,
mean-pool over feature dim, then a Dueling head.

Output is standard DuelingDQN-style Q-values (B, n_actions). Drop-in
compatible with masked_argmax / masked_max / vote ensembling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDQN(nn.Module):
    def __init__(self, state_dim: int = 52, n_actions: int = 12,
                  d_model: int = 16, n_heads: int = 4, n_layers: int = 2,
                  hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.d_model   = d_model

        # Per-feature scalar→d_model embedding + learned positional embedding
        self.feature_embed = nn.Linear(1, d_model, bias=False)
        self.pos_embed     = nn.Parameter(torch.randn(state_dim, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Dueling head over pooled representation
        pooled = d_model
        self.fc1 = nn.Linear(pooled, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.v_head = nn.Linear(hidden // 2, 1)
        self.a_head = nn.Linear(hidden // 2, n_actions)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, state_dim) → Q: (B, n_actions)."""
        # (B, state_dim, 1) → (B, state_dim, d_model)
        x = self.feature_embed(s.unsqueeze(-1)) + self.pos_embed.unsqueeze(0)
        x = self.encoder(x)                                 # (B, state_dim, d_model)
        x = x.mean(dim=1)                                   # mean-pool (B, d_model)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.v_head(x)                                  # (B, 1)
        a = self.a_head(x)                                  # (B, n_actions)
        return v + a - a.mean(dim=1, keepdim=True)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
