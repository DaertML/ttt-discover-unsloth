"""
TTT-Discover Trainer — Algorithm 1 with real LoRA gradient updates.

Every step:
  1. PUCT selects initial state s_i
  2. Generate rollouts_per_step responses from current pi_theta
  3. Execute code -> reward
  4. Entropic weights + advantage A = w_beta - 1
  5. loss.backward() + optimizer.step()  ->  theta_{i+1} != theta_i
  6. Update PUCT buffer
  7. Optional LoRA checkpoint
"""

import math
import time
import json
import logging
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from environments import BaseEnvironment, State   # flat import
from unsloth_model import UnslothModel, LLMAction  # flat import
from search import ReplayBuffer, PUCTSelector, BufferEntry  # flat import

logger = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────────

class TTTDiscoverConfig(BaseModel):
    # Training
    n_steps: int = 50
    rollouts_per_step: int = 16      # paper: 512 on cluster; 8-32 practical on 1 GPU
    n_initial_states: int = 4

    # PUCT
    exploration_coeff: float = 1.0

    # Entropic objective
    beta_min: float = 1.0            # low beta -> explore (near-uniform weights)
    beta_max: float = 10.0           # high beta -> exploit (weight on best rollout)

    # Generation temperature schedule
    temperature_start: float = 0.9
    temperature_end: float = 0.4

    # Checkpointing
    save_every_n_steps: int = 10     # 0 = never
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"


# ─── Helpers ──────────────────────────────────────────────────────────────────────

def _beta(step: int, n: int, lo: float, hi: float) -> float:
    return lo + (hi - lo) * step / max(n - 1, 1)


def _temp(step: int, n: int, start: float, end: float) -> float:
    return start - (start - end) * step / max(n - 1, 1)


def _entropic_weights(rewards: list[float], beta: float) -> list[float]:
    """w_beta(a_i) = exp(beta*R_i) / sum_j exp(beta*R_j)  (numerically stable)."""
    r = np.array(rewards, dtype=np.float64)
    e = np.exp(np.clip(beta * (r - r.max()), -60, 0))
    return (e / e.sum()).tolist()


def _build_prompt(
    env: BaseEnvironment,
    initial_state: State,
    best_examples: list[tuple[State, str]],
    beta: float,
    step: int,
    n_steps: int,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for one rollout."""
    progress = step / max(n_steps - 1, 1)
    phase = "EXPLORE" if progress < 0.35 else "REFINE"

    parts = [
        env.get_problem_description(),
        "\n" + "=" * 60 + "\n",
        f"## Step {step+1}/{n_steps}  |  {phase}  |  beta={beta:.1f}\n",
        "## Starting solution (improve this)",
        env.format_state_for_prompt(initial_state),
        "",
    ]

    if best_examples:
        parts.append("## Best solutions found so far")
        for k, (ex, thinking) in enumerate(best_examples):
            bound = ex.metadata.get("bound", "?")
            parts.append(f"\n### Example {k+1}  (bound={bound})")
            if thinking:
                parts.append(f"Reasoning:\n{thinking[:500]}...")
            parts.append(env.format_state_for_prompt(ex))
        parts.append("")

    parts += [
        "## Your task",
        "Write Python code that sets `f` to a list of floats with a smaller upper bound.",
        "Constraints: all values in [0,1], mean(f)==0.5.",
        "Wrap code in ```python ... ```.",
    ]

    return env.get_system_prompt(), "\n".join(parts)


# ─── Trainer ──────────────────────────────────────────────────────────────────────

class TTTDiscoverTrainer:

    def __init__(
        self,
        env: BaseEnvironment,
        model: UnslothModel,
        config: TTTDiscoverConfig | None = None,
    ):
        self.env    = env
        self.model  = model
        self.config = config or TTTDiscoverConfig()
        self.buffer = ReplayBuffer()
        self.puct   = PUCTSelector(self.config.exploration_coeff)
        self.logs: list[dict] = []

    def _seed_buffer(self):
        logger.info(f"Seeding buffer with {self.config.n_initial_states} random states ...")
        for i in range(self.config.n_initial_states):
            s = self.env.get_initial_state()
            self.buffer.add_initial_state(s)
            logger.info(f"  [{i+1}] reward={s.reward:.6f}  bound={s.metadata.get('bound','?')}")

    def _best_examples(self, n: int = 3) -> list[tuple[State, str]]:
        seen, out = set(), []
        for e in sorted(self.buffer.entries, key=lambda x: x.reward, reverse=True):
            if e.reward <= 0:
                continue
            key = round(e.next_state.metadata.get("bound", 0), 6)
            if key not in seen:
                seen.add(key)
                out.append((e.next_state, e.thinking))
            if len(out) >= n:
                break
        return out

    def run(self) -> State:
        cfg = self.config
        logger.info(f"TTT-Discover  env={self.env.config.name}  model={self.model.config.model_name}")
        logger.info(f"steps={cfg.n_steps}  rollouts/step={cfg.rollouts_per_step}  "
                    f"beta {cfg.beta_min}->{cfg.beta_max}  T {cfg.temperature_start}->{cfg.temperature_end}")

        self._seed_buffer()
        _, sota_reward = self.env.get_sota()
        logger.info(f"Known SOTA reward: {sota_reward:.8f}")
        t0 = time.time()

        for step in range(cfg.n_steps):
            step_t = time.time()
            beta = _beta(step, cfg.n_steps, cfg.beta_min, cfg.beta_max)
            temp = _temp(step, cfg.n_steps, cfg.temperature_start, cfg.temperature_end)

            # 1. PUCT: pick initial state
            state_idx, init_state = self.puct.select(self.buffer)
            logger.info(f"\nStep {step+1}/{cfg.n_steps}  beta={beta:.2f}  T={temp:.3f}  "
                        f"state_idx={state_idx}  init_reward={init_state.reward:.6f}")

            # 2. Build prompt
            sys_p, usr_p = _build_prompt(
                self.env, init_state, self._best_examples(3), beta, step, cfg.n_steps
            )

            # 3. Generate rollouts from CURRENT theta_i
            actions: list[LLMAction] = self.model.generate(
                sys_p, usr_p, n=cfg.rollouts_per_step, temperature=temp
            )

            # 4. Execute code -> reward
            rewards, responses, entries = [], [], []
            for action in actions:
                next_state = self.env.transition(action.code, init_state)
                rewards.append(next_state.reward)
                responses.append(action.raw_response)
                entries.append(BufferEntry(
                    state=init_state, action_code=action.code,
                    next_state=next_state, reward=next_state.reward,
                    thinking=action.thinking, step=step,
                ))

            n_valid = sum(e.next_state.valid for e in entries)
            logger.info(f"  {n_valid}/{cfg.rollouts_per_step} valid  "
                        f"max={max(rewards):.6f}  mean={np.mean(rewards):.6f}")

            # 5. Gradient step on LoRA weights (the real TTT-Discover update)
            #    loss = -mean( A_i * log pi_theta(response_i | prompt) )
            #    where A_i = w_beta(a_i) - 1
            train_stats = {}
            if n_valid > 0:
                train_stats = self.model.train_step(sys_p, usr_p, responses, rewards, beta)
                logger.info(f"  grad step: loss={train_stats.get('loss',0):.5f}  "
                            f"valid={train_stats.get('valid_rollouts',0)}")

            # 6. Update PUCT buffer
            for weight, entry in sorted(
                zip(_entropic_weights(rewards, beta), entries),
                key=lambda x: x[0], reverse=True
            ):
                self.buffer.add_entry(entry, state_idx)
                if entry.next_state.valid and entry.reward > 0:
                    self.buffer.add_initial_state(entry.next_state)

            # 7. Checkpoint
            if cfg.save_every_n_steps > 0 and (step + 1) % cfg.save_every_n_steps == 0:
                self.model.save_lora(tag=f"step_{step+1:04d}")

            # Log
            elapsed = time.time() - step_t
            self.logs.append({
                "step": step + 1, "beta": beta, "temperature": temp,
                "n_valid": n_valid,
                "max_reward": max(rewards) if rewards else 0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0,
                "global_best_reward": self.buffer.best_reward,
                "global_best_bound": (self.buffer.best_state.metadata.get("bound")
                                      if self.buffer.best_state else None),
                "buffer_size": self.buffer.size(),
                "step_time_s": elapsed,
                **{f"train_{k}": v for k, v in train_stats.items()},
            })
            logger.info(f"  global_best={self.buffer.best_reward:.8f}  "
                        f"bound={self.logs[-1]['global_best_bound']}  "
                        f"buffer={self.buffer.size()}  t={elapsed:.1f}s")
            if self.buffer.best_reward > sota_reward:
                logger.info(f"  *** NEW SOTA: {1/self.buffer.best_reward:.8f} "
                            f"(was {1/sota_reward:.8f}) ***")

        logger.info(f"\nDone in {time.time()-t0:.0f}s\n{self.buffer.summary()}")
        self.model.save_lora("final")
        self._save()
        return self.buffer.best_state

    def _save(self):
        out = Path(self.config.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = self.env.config.name
        with open(out / f"{name}_log.json", "w") as f:
            json.dump(self.logs, f, indent=2)
        if self.buffer.best_state:
            b = self.buffer.best_state
            with open(out / f"{name}_best.json", "w") as f:
                json.dump({"reward": b.reward, "valid": b.valid,
                           "metadata": b.metadata, "content": b.content}, f, indent=2)
        logger.info(f"Results saved to {out}/")

    def get_logs(self) -> list[dict]:
        return self.logs
