#!/usr/bin/env python3
"""
TTT-Discover — Unsloth edition.
Run from the same directory as the other .py files.

Examples
--------
# Quick smoke-test
python run.py --steps 5 --rollouts 4 --model unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit

# Paper-like run
python run.py --steps 50 --rollouts 32 --model unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit

# List available environments
python run.py --list-envs

# Resume from checkpoint
python run.py --resume ./checkpoints/step_0020 --steps 50 --rollouts 32
"""

import argparse
import logging
import sys
import os

# Make sure Python finds the other files in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_env_registry() -> dict:
    from environments import ErdosEnvironment
    return {
        "erdos": {
            "class": ErdosEnvironment,
            "description": "Erdos Minimum Overlap Problem (bound minimisation)",
        },
        # Add more environments here:
        # "myenv": {"class": MyEnvironment, "description": "..."},
    }


def main():
    parser = argparse.ArgumentParser(
        description="TTT-Discover (Unsloth): test-time LoRA training for scientific discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--env", default="erdos")
    parser.add_argument("--list-envs", action="store_true")

    parser.add_argument(
        "--model", default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        help="Unsloth model (HuggingFace hub ID or local path)",
    )
    parser.add_argument("--lora-r",     type=int,   default=32)
    parser.add_argument("--lora-alpha", type=int,   default=64)
    parser.add_argument("--no-4bit",    action="store_true")
    parser.add_argument("--max-seq-len",    type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)

    parser.add_argument("--steps",      type=int,   default=50)
    parser.add_argument("--rollouts",   type=int,   default=16)
    parser.add_argument("--init-states",type=int,   default=4)
    parser.add_argument("--beta-min",   type=float, default=1.0)
    parser.add_argument("--beta-max",   type=float, default=10.0)
    parser.add_argument("--temp-start", type=float, default=0.9)
    parser.add_argument("--temp-end",   type=float, default=0.4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--exploration",type=float, default=1.0)
    parser.add_argument("--kl-coeff",   type=float, default=0.01)

    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--save-every",     type=int, default=10)
    parser.add_argument("--results-dir",    default="./results")
    parser.add_argument("--resume",         default=None,
                        help="Path to a LoRA checkpoint directory to resume from")

    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    registry = get_env_registry()

    if args.list_envs:
        print("\nAvailable environments:")
        for name, info in registry.items():
            print(f"  {name:20s}  {info['description']}")
        return

    if args.env not in registry:
        print(f"Unknown environment '{args.env}'. Available: {list(registry)}")
        sys.exit(1)

    from unsloth_model import UnslothModel, UnslothModelConfig
    from trainer import TTTDiscoverTrainer, TTTDiscoverConfig

    model_cfg = UnslothModelConfig(
        model_name=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=not args.no_4bit,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        checkpoint_dir=args.checkpoint_dir,
    )
    train_cfg = TTTDiscoverConfig(
        n_steps=args.steps,
        rollouts_per_step=args.rollouts,
        n_initial_states=args.init_states,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        exploration_coeff=args.exploration,
        save_every_n_steps=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
    )

    print(f"\n{'='*60}")
    print("TTT-Discover  (Unsloth)")
    print(f"{'='*60}")
    print(f"  env       : {args.env}")
    print(f"  model     : {args.model}")
    print(f"  LoRA      : r={args.lora_r}  alpha={args.lora_alpha}  4bit={not args.no_4bit}")
    print(f"  steps     : {args.steps}  rollouts/step: {args.rollouts}")
    print(f"  beta      : {args.beta_min} -> {args.beta_max}")
    print(f"  temp      : {args.temp_start} -> {args.temp_end}")
    print(f"  lr        : {args.lr}")
    print(f"{'='*60}\n")

    model = UnslothModel(model_cfg)
    model.load()

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model.load_lora(args.resume)

    env = registry[args.env]["class"]()
    trainer = TTTDiscoverTrainer(env=env, model=model, config=train_cfg)
    best = trainer.run()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    if best:
        bound = best.metadata.get("bound", "N/A")
        _, sota_reward = env.get_sota()
        sota_bound = round(1.0 / sota_reward, 8) if sota_reward else "N/A"
        print(f"  Best bound  : {bound}")
        print(f"  SOTA bound  : {sota_bound}")
        if isinstance(bound, float) and isinstance(sota_bound, float):
            delta = sota_bound - bound
            if delta > 0:
                print(f"  NEW SOTA by : {delta:.8f}")
            else:
                print(f"  Gap to SOTA : {-delta:.8f}")
    print(f"  Checkpoint  : {args.checkpoint_dir}/final")
    print(f"  Log         : {args.results_dir}/{args.env}_log.json")


if __name__ == "__main__":
    main()
