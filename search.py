"""
Search components: ReplayBuffer and PUCT state selector.
"""

import math
import logging
from typing import Optional

from pydantic import BaseModel
from environments import State   # flat import — same directory

logger = logging.getLogger(__name__)


class BufferEntry(BaseModel):
    state: State
    action_code: str
    next_state: State
    reward: float
    thinking: str = ""
    step: int = 0

    class Config:
        arbitrary_types_allowed = True


class ReplayBuffer:
    """Stores candidate solutions and tracks parent-child reward relationships."""

    def __init__(self):
        self.states: list[State] = []
        self.expansion_counts: list[int] = []
        self.child_rewards: list[list[float]] = []
        self.total_expansions: int = 0
        self.entries: list[BufferEntry] = []
        self.best_state: Optional[State] = None
        self.best_reward: float = -math.inf

    def add_initial_state(self, state: State) -> int:
        idx = len(self.states)
        self.states.append(state)
        self.expansion_counts.append(0)
        self.child_rewards.append([])
        if state.reward > self.best_reward:
            self.best_reward = state.reward
            self.best_state = state
        return idx

    def add_entry(self, entry: BufferEntry, parent_idx: int):
        self.entries.append(entry)
        if 0 <= parent_idx < len(self.states):
            self.expansion_counts[parent_idx] += 1
            self.child_rewards[parent_idx].append(entry.reward)
        self.total_expansions += 1
        if entry.reward > self.best_reward:
            self.best_reward = entry.reward
            self.best_state = entry.next_state

    def get_max_child_reward(self, idx: int) -> float:
        children = self.child_rewards[idx]
        return max(children) if children else self.states[idx].reward

    def size(self) -> int:
        return len(self.states)

    def summary(self) -> str:
        lines = [f"Buffer: {self.size()} states, {len(self.entries)} entries",
                 f"Best reward: {self.best_reward:.8f}"]
        if self.best_state and self.best_state.metadata.get("bound"):
            lines.append(f"Best bound:  {self.best_state.metadata['bound']:.8f}")
        return "\n".join(lines)


class PUCTSelector:
    """
    PUCT-inspired selection (paper Section 3.2).

    score(s) = Q(s) + c * P(s) * sqrt(1 + T) / (1 + n(s))

    Q(s) = max child reward (not mean — we care about best outcome)
    P(s) = rank-based prior (higher reward -> higher P)
    T    = total expansions so far
    n(s) = times s has been expanded
    c    = exploration coefficient
    """

    def __init__(self, exploration_coeff: float = 1.0):
        self.c = exploration_coeff

    def select(self, buffer: ReplayBuffer) -> tuple[int, State]:
        n = buffer.size()
        if n == 0:
            raise ValueError("Buffer is empty.")
        if n == 1:
            return 0, buffer.states[0]

        T = buffer.total_expansions
        rewards = [buffer.states[i].reward for i in range(n)]
        sorted_idx = sorted(range(n), key=lambda i: rewards[i], reverse=True)
        rank = [0] * n
        for r, idx in enumerate(sorted_idx):
            rank[idx] = r

        scores = []
        for i in range(n):
            Q = buffer.get_max_child_reward(i)
            P = (n - rank[i]) / n
            bonus = self.c * P * math.sqrt(1 + T) / (1 + buffer.expansion_counts[i])
            scores.append(Q + bonus)

        best = max(range(n), key=lambda i: scores[i])
        return best, buffer.states[best]
