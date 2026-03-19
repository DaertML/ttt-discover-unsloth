"""
Unsloth-based model wrapper for TTT-Discover.

Handles:
- Loading a causal LM with Unsloth (4-bit QLoRA by default)
- LoRA adapter attachment (matching paper: rank 32)
- Text generation (rollouts) under torch.no_grad
- Gradient-based weight updates via the entropic objective (train_step)
- LoRA checkpoint save/load
"""

import re
import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ─── Pydantic response models ─────────────────────────────────────────────────────

class LLMAction(BaseModel):
    thinking: str = Field(default="")
    code: str = Field(...)
    raw_response: str = Field(default="")

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Code block cannot be empty")
        return v.strip()


# ─── Code / thinking extraction ───────────────────────────────────────────────────

def _extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines, in_code = [], False
    for line in text.strip().split("\n"):
        if any(kw in line for kw in ["import ", "def ", "for ", "f =", "np.", "current_f"]):
            in_code = True
        if in_code:
            lines.append(line)
    return "\n".join(lines) if lines else None


def _extract_thinking(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    code_start = text.find("```")
    return text[:code_start].strip() if code_start > 0 else ""


# ─── Model config ─────────────────────────────────────────────────────────────────

class UnslothModelConfig(BaseModel):
    model_name: str = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"

    # LoRA  (paper uses rank 32)
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # Generation
    max_new_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    repetition_penalty: float = 1.1

    # Training
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    kl_coeff: float = 0.01

    # Misc
    max_seq_len: int = 4096
    dtype: str = "auto"
    load_in_4bit: bool = True
    checkpoint_dir: str = "./checkpoints"


# ─── UnslothModel ─────────────────────────────────────────────────────────────────

class UnslothModel:
    """
    Wraps an Unsloth-loaded causal LM with LoRA for TTT-Discover.

    Lifecycle:
      model = UnslothModel(config)
      model.load()            # downloads weights, attaches LoRA, builds optimizer
      actions = model.generate(sys_p, usr_p, n=16)   # sample rollouts
      stats   = model.train_step(sys_p, usr_p, responses, rewards, beta)  # gradient step
      model.save_lora("step_0010")
    """

    def __init__(self, config: UnslothModelConfig | None = None):
        self.config = config or UnslothModelConfig()
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────────

    def load(self):
        """Download weights, attach LoRA adapter, build AdamW optimizer."""
        logger.info(f"Loading: {self.config.model_name}")
        logger.info(f"  4-bit={self.config.load_in_4bit}  LoRA r={self.config.lora_r}")

        from unsloth import FastLanguageModel

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "auto": None}

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_len,
            dtype=dtype_map.get(self.config.dtype),
            load_in_4bit=self.config.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            target_modules=self.config.lora_target_modules,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        logger.info(f"  Trainable LoRA parameters: {n_params:,}")

        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.config.learning_rate, weight_decay=0.0
        )
        self._loaded = True
        logger.info("Model ready.")

    def _assert_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call UnslothModel.load() first.")

    # ── Prompt formatting ─────────────────────────────────────────────────────────

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ── Generation ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        n: int = 1,
        temperature: float | None = None,
    ) -> list[LLMAction]:
        """Sample n rollouts from current policy pi_theta."""
        self._assert_loaded()
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        temp = temperature if temperature is not None else self.config.temperature
        prompt_text = self._format_prompt(system_prompt, user_prompt)

        inputs = self.tokenizer(
            [prompt_text] * n,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len // 2,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=temp,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        actions = []
        for output_ids in outputs:
            raw = self.tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)
            code = _extract_code(raw) or "# no code found"
            thinking = _extract_thinking(raw)
            try:
                actions.append(LLMAction(thinking=thinking, code=code, raw_response=raw))
            except Exception:
                actions.append(LLMAction(thinking=thinking, code="# parse error", raw_response=raw))

        return actions

    # ── Training step ─────────────────────────────────────────────────────────────

    def train_step(
        self,
        system_prompt: str,
        user_prompt: str,
        responses: list[str],
        rewards: list[float],
        beta: float,
    ) -> dict:
        """
        One gradient step of the entropic objective J_beta(theta).

        Derivation:
          J_beta = E_s [ log E_a~pi_theta [ exp(beta * R(a)) ] ]
          grad   = E_a [ w_beta(a) * grad log pi_theta(a|s) ]
          where  w_beta(a) = exp(beta*R(a)) / E[exp(beta*R)]

        Advantage (with baseline E[w_beta] = 1):
          A(a) = w_beta(a) - 1

        Loss (negated for gradient ascent via descent):
          L = -mean_i( A_i * log pi_theta(response_i | prompt) )

        After loss.backward() + optimizer.step(), theta_{i+1} != theta_i.
        The model that generates the next batch of rollouts is genuinely different.
        """
        self._assert_loaded()
        from unsloth import FastLanguageModel
        FastLanguageModel.for_training(self.model)

        if not rewards:
            return {"loss": 0.0, "mean_reward": 0.0}

        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # Entropic weights w_beta — numerically stable
        shifted = beta * (rewards_t - rewards_t.max())
        exp_r = torch.exp(torch.clamp(shifted, -50, 50))
        w_beta = exp_r / exp_r.sum()

        # Advantage: A = w_beta - 1  (baseline = E[w_beta] = 1 by construction)
        advantages = w_beta - 1.0

        prompt_text = self._format_prompt(system_prompt, user_prompt)
        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=self.config.max_seq_len // 2,
        )["input_ids"].to(self.model.device)
        prompt_len = prompt_ids.shape[1]

        total_loss = torch.tensor(0.0, device=self.model.device)
        valid_count = 0

        for response_text, adv in zip(responses, advantages):
            full_text = prompt_text + response_text
            full_ids = self.tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=self.config.max_seq_len,
            )["input_ids"].to(self.model.device)

            if full_ids.shape[1] <= prompt_len + 1:
                continue  # empty response after tokenisation

            # Forward pass — compute logits over the full sequence
            logits = self.model(input_ids=full_ids).logits  # (1, T, V)

            # Shift so we predict token t from token t-1
            # We only care about the response tokens (after the prompt)
            shift_logits = logits[0, prompt_len - 1:-1, :]   # (R, V)
            shift_labels = full_ids[0, prompt_len:]           # (R,)

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_lp = log_probs[
                torch.arange(shift_labels.shape[0], device=self.model.device),
                shift_labels,
            ]  # (R,)

            # Sequence log-prob = mean token log-prob
            seq_lp = token_lp.mean()

            # Policy-gradient loss for this rollout
            rollout_loss = -adv.to(self.model.device) * seq_lp
            total_loss = total_loss + rollout_loss
            valid_count += 1

        if valid_count == 0:
            logger.warning("train_step: no valid rollouts.")
            return {"loss": 0.0, "mean_reward": float(rewards_t.mean()), "valid_rollouts": 0}

        loss = total_loss / valid_count

        # ── Actual gradient step ──────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": float(rewards_t.mean()),
            "max_reward": float(rewards_t.max()),
            "valid_rollouts": valid_count,
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────────

    def save_lora(self, tag: str = "latest"):
        """Save only the LoRA adapter weights (small — a few hundred MB)."""
        self._assert_loaded()
        path = Path(self.config.checkpoint_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        logger.info(f"LoRA saved -> {path}")

    def load_lora(self, checkpoint_path: str):
        """Load a previously saved LoRA adapter by full path."""
        self._assert_loaded()
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        logger.info(f"LoRA loaded <- {checkpoint_path}")
