"""
Replay buffer for DQN.

Stores fixed-shape transitions in numpy arrays (circular). Each transition:
  state         : (state_dim,) float32
  action        : ()           int8       — chosen action 0..n_actions-1
  reward        : ()           float32    — total trade PnL (or 0 for NO_TRADE)
  duration      : ()           int16      — n bars from decision to next decision
                                             (1 for NO_TRADE; 1+n_held for trade)
  state_next    : (state_dim,) float32    — state at the next decision point
  valid_next    : (n_actions,) bool       — action mask at state_next
  done          : ()           bool       — true if episode end (no bootstrap)
  priority      : ()           float32    — abs TD-error + ε  (for PER, set later)

For Phase 3.2 this implements **uniform sampling**. Phase 3.3 swaps in
prioritized sampling (`sample_prioritized`) using the same storage.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, n_actions: int,
                 priority_eps: float = 1e-3):
        self.capacity   = int(capacity)
        self.state_dim  = state_dim
        self.n_actions  = n_actions
        self._eps       = priority_eps

        self.state      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action     = np.zeros((capacity,),           dtype=np.int8)
        self.reward     = np.zeros((capacity,),           dtype=np.float32)
        self.duration   = np.zeros((capacity,),           dtype=np.int16)
        self.state_next = np.zeros((capacity, state_dim), dtype=np.float32)
        self.valid_next = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.done       = np.zeros((capacity,),           dtype=np.bool_)
        self.priority   = np.full((capacity,), priority_eps, dtype=np.float32)

        self.ptr  = 0
        self.size = 0

    def push(self, s, a, r, dur, s_next, v_next, done, priority=None):
        i = self.ptr
        self.state[i]      = s
        self.action[i]     = a
        self.reward[i]     = r
        self.duration[i]   = dur
        self.state_next[i] = s_next
        self.valid_next[i] = v_next
        self.done[i]       = done
        # Initialize priority high so new transitions are sampled at least once
        self.priority[i]   = max(self.priority[: self.size + 1].max(), self._eps) \
                             if priority is None else priority
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ── uniform sampling (Phase 3.2) ─────────────────────────────────────────
    def sample_uniform(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._gather(idx), idx, np.ones(batch_size, dtype=np.float32)

    # ── prioritized sampling (Phase 3.3) ─────────────────────────────────────
    def sample_prioritized(self, batch_size: int, alpha: float = 0.6,
                            beta: float = 0.4):
        prios = self.priority[: self.size] ** alpha
        probs = prios / prios.sum()
        idx   = np.random.choice(self.size, size=batch_size, p=probs)
        # Importance-sampling correction
        is_w  = (self.size * probs[idx]) ** (-beta)
        is_w  = (is_w / is_w.max()).astype(np.float32)
        return self._gather(idx), idx, is_w

    # ── stratified-prioritized sampling (Path A) ─────────────────────────────
    def sample_stratified_prioritized(self, batch_size: int,
                                       alpha: float = 0.6, beta: float = 0.4):
        """50/50 split between action=0 (NO_TRADE) and action!=0 (trades).
        Within each stratum: prioritized sampling via |TD error|^alpha.

        Counters the buffer's natural action skew (~92% NO_TRADE under realistic
        coverage). Used in Path A to give trade transitions stronger gradient
        signal."""
        n_trade = batch_size // 2
        n_notrd = batch_size - n_trade

        action_arr = self.action[: self.size]
        no_idx = np.where(action_arr == 0)[0]
        tr_idx = np.where(action_arr != 0)[0]

        # Fallback if one stratum is empty
        if len(tr_idx) == 0 or len(no_idx) == 0:
            return self.sample_prioritized(batch_size, alpha, beta)

        no_p = (self.priority[no_idx] ** alpha)
        no_p = no_p / no_p.sum()
        tr_p = (self.priority[tr_idx] ** alpha)
        tr_p = tr_p / tr_p.sum()

        no_pos = np.random.choice(len(no_idx), size=n_notrd, p=no_p)
        tr_pos = np.random.choice(len(tr_idx), size=n_trade, p=tr_p)
        no_picks = no_idx[no_pos]
        tr_picks = tr_idx[tr_pos]
        idx      = np.concatenate([no_picks, tr_picks])

        # IS weights computed relative to within-stratum sampling probability.
        # This corrects bias from non-uniform priority sampling within each
        # stratum but DOES NOT correct the (intentional) 50/50 stratification.
        is_w = np.empty(batch_size, dtype=np.float32)
        is_w[:n_notrd] = (len(no_idx) * no_p[no_pos]) ** -beta
        is_w[n_notrd:] = (len(tr_idx) * tr_p[tr_pos]) ** -beta
        is_w = (is_w / is_w.max()).astype(np.float32)

        return self._gather(idx), idx, is_w

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray):
        self.priority[idx] = np.abs(td_errors).astype(np.float32) + self._eps

    def _gather(self, idx):
        return dict(
            state      = self.state[idx],
            action     = self.action[idx],
            reward     = self.reward[idx],
            duration   = self.duration[idx],
            state_next = self.state_next[idx],
            valid_next = self.valid_next[idx],
            done       = self.done[idx],
        )

    def __len__(self):
        return self.size
