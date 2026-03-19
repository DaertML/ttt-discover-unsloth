"""
Discovery problem environments for TTT-Discover.

Contains:
- BaseEnvironment   abstract interface
- State             Pydantic model for a candidate solution
- ErdosEnvironment  Erdos Minimum Overlap Problem
"""

import json
import math
import textwrap
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


# ─── Core data models ─────────────────────────────────────────────────────────────

class EnvironmentConfig(BaseModel):
    name: str
    description: str
    maximize: bool = True


class State(BaseModel):
    content: Any
    reward: float = 0.0
    valid: bool = True
    metadata: dict = {}

    class Config:
        arbitrary_types_allowed = True


# ─── Abstract base ────────────────────────────────────────────────────────────────

class BaseEnvironment(ABC):

    def __init__(self, config: EnvironmentConfig):
        self.config = config

    @abstractmethod
    def get_problem_description(self) -> str: ...

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_initial_state(self) -> State: ...

    @abstractmethod
    def reward(self, state: State) -> float: ...

    @abstractmethod
    def transition(self, action_code: str, current_state: State) -> State: ...

    @abstractmethod
    def get_sota(self) -> tuple[State | None, float]: ...

    def format_state_for_prompt(self, state: State) -> str:
        return str(state.content)

    def is_better(self, r1: float, r2: float) -> bool:
        return r1 > r2 if self.config.maximize else r1 < r2


# ─── Erdos helpers ────────────────────────────────────────────────────────────────

def _compute_upper_bound(f_values: list[float]) -> float:
    """
    Certified upper bound on c = lim M(n)/n for the Erdos problem.

    bound = max_k { (1/N) * sum_i f[i] * (1 - f[(i+k) % N]) }
          = mean(f) - min_k autocorr[k]

    Computed in O(N log N) via FFT.
    """
    f = np.array(f_values, dtype=np.float64)
    n = len(f)

    if np.any(f < -1e-9) or np.any(f > 1 + 1e-9):
        return float("inf")
    if abs(np.mean(f) - 0.5) > 1e-3:
        return float("inf")

    f = np.clip(f, 0, 1)
    F = np.fft.rfft(f)
    autocorr = np.fft.irfft(F * np.conj(F), n=n) / n
    C = np.mean(f) - autocorr
    return float(np.max(C))


def _execute_action_code(code: str, current_f: list[float]) -> list[float] | None:
    local_vars = {
        "current_f": list(current_f),
        "np": np,
        "math": math,
        "f": None,
    }
    try:
        exec(code, {"__builtins__": __builtins__}, local_vars)
        result = local_vars.get("f")
        if result is None:
            return None
        return list(np.array(result, dtype=np.float64))
    except Exception:
        return None


# ─── Erdos environment ────────────────────────────────────────────────────────────

class ErdosEnvironment(BaseEnvironment):
    """
    Erdos Minimum Overlap Problem.

    Find a step function f: [0,1] -> [0,1] (GRID_SIZE values, mean=0.5)
    that minimises:
        bound = max_k { (1/N) * sum_i f[i] * (1 - f[(i+k) % N]) }

    Reward = 1 / bound  (so maximising reward <=> minimising bound).

    Known results:
        Human SOTA (Haugland 1996) : 0.380927
        AlphaEvolve                : 0.380924
        TTT-Discover (paper, 120B) : 0.380876
    """

    GRID_SIZE = 100  # 100 sufficient; 500 causes LLMs to emit literal arrays and truncate
    KNOWN_SOTA_BOUND = 0.380924

    def __init__(self):
        super().__init__(EnvironmentConfig(
            name="erdos_minimum_overlap",
            description="Erdos Minimum Overlap Problem",
            maximize=True,
        ))

    def get_problem_description(self) -> str:
        return textwrap.dedent(f"""
        # Erdos Minimum Overlap Problem

        Partition {{1,...,2n}} into sets A and B of equal size n.
        Define Mk = number of solutions to ai - bj = k  (ai in A, bj in B).
        Let M(n) = min_{{A,B}} max_k Mk.  Goal: bound c = lim M(n)/n.

        ## Representation
        Step function f as a Python list of {self.GRID_SIZE} floats on [0,1].

        ## Constraints (MUST hold)
        - 0 <= f[i] <= 1  for all i
        - mean(f) == 0.5

        ## Certified upper bound (minimise this)
        bound = max_k {{ (1/N) * sum_i f[i] * (1 - f[(i+k) % N]) }}  where N={self.GRID_SIZE}

        ## Your code
        Must set variable `f` to a list of {self.GRID_SIZE} floats.
        `current_f` is available as a starting point.
        `np` (numpy) and `math` are imported.
        """).strip()

    def get_system_prompt(self) -> str:
        return textwrap.dedent(f"""
        You are an expert mathematician specialising in combinatorial optimisation.

        Write Python code that COMPUTES a step function `f` (a numpy array or list
        of exactly {self.GRID_SIZE} floats) that minimises the Erdos upper bound.

        CRITICAL RULES:
        - Output ONLY a ```python ... ``` code block. No prose outside the block.
        - Your code MUST set the variable `f` as the FINAL action.
        - DO NOT write f as a literal list of numbers — write CODE that constructs it
          algorithmically (e.g. using np.linspace, optimisation, perturbation).
        - `current_f` (a list of {self.GRID_SIZE} floats), `np`, and `math` are available.
        - All values of f must be in [0, 1] and np.mean(f) must equal 0.5.
        - scipy.optimize is available if you want to import it inside the block.

        Good approaches: gradient descent on the bound, perturbing current_f,
        Fourier series constructions, scipy.optimize.minimize.
        Bad approach: writing out 100 literal numbers.
        """).strip()

    def get_initial_state(self) -> State:
        rng = np.random.default_rng()
        f = rng.uniform(0, 1, self.GRID_SIZE)
        f = f - np.mean(f) + 0.5
        f = np.clip(f, 0, 1)
        f = f / np.mean(f) * 0.5
        f = np.clip(f, 0, 1).tolist()
        bound = _compute_upper_bound(f)
        reward = 1.0 / bound if (bound > 0 and not math.isinf(bound)) else 0.0
        return State(content=f, reward=reward, valid=not math.isinf(bound),
                     metadata={"bound": bound})

    def reward(self, state: State) -> float:
        if not state.valid:
            return 0.0
        b = _compute_upper_bound(state.content)
        return 0.0 if (math.isinf(b) or b <= 0) else 1.0 / b

    def transition(self, action_code: str, current_state: State) -> State:
        current_f = current_state.content or [0.5] * self.GRID_SIZE
        new_f = _execute_action_code(action_code, current_f)

        if new_f is None:
            return State(content=current_f, reward=0.0, valid=False,
                         metadata={"error": "execution failed"})

        # Resample if wrong length
        if len(new_f) != self.GRID_SIZE:
            old_x = np.linspace(0, 1, len(new_f))
            new_x = np.linspace(0, 1, self.GRID_SIZE)
            new_f = np.interp(new_x, old_x, np.array(new_f)).tolist()

        bound = _compute_upper_bound(new_f)
        if math.isinf(bound) or bound <= 0:
            return State(content=new_f, reward=0.0, valid=False,
                         metadata={"bound": bound, "error": "invalid step function"})

        return State(content=new_f, reward=1.0 / bound, valid=True,
                     metadata={"bound": bound})

    def get_sota(self) -> tuple[State | None, float]:
        return None, 1.0 / self.KNOWN_SOTA_BOUND

    def format_state_for_prompt(self, state: State) -> str:
        if not state.content:
            return "No current solution."
        f = np.array(state.content)
        bound = state.metadata.get("bound", _compute_upper_bound(state.content))
        return (
            f"Step function ({len(state.content)} values)\n"
            f"  Certified bound : {bound:.8f}\n"
            f"  Mean            : {f.mean():.6f}  (target 0.5)\n"
            f"  Min / Max       : {f.min():.4f} / {f.max():.4f}\n"
            f"  First 10        : {f[:10].tolist()}\n"
            f"  Last  10        : {f[-10:].tolist()}\n"
            f"  Full array (current_f) : {json.dumps(state.content)}"
        )
