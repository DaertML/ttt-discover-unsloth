#!/usr/bin/env python3
"""
compare_models.py — compare base vs LoRA-trained model on an environment.

Diagnoses exactly why rollouts fail and auto-repairs common constraint
violations (wrong array length, mean != 0.5) before discarding them.

Usage
-----
python compare_models.py --checkpoint ./checkpoints/final
python compare_models.py --checkpoint ./checkpoints/final --rollouts 8 --show-code
python compare_models.py --checkpoint ./checkpoints/final --output comparison.json
"""

import argparse
import json
import logging
import math
import sys
import os
import time
import traceback
from dataclasses import dataclass, asdict, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Failure taxonomy ─────────────────────────────────────────────────────────────

FAIL_NONE          = "ok"
FAIL_NO_CODE       = "no_code_block"       # LLM didn't produce a ```python block
FAIL_EXEC_ERROR    = "exec_error"          # code raised an exception
FAIL_NO_F          = "f_not_set"           # code ran but never set variable f
FAIL_WRONG_TYPE    = "f_wrong_type"        # f is not numeric/iterable
FAIL_EMPTY         = "f_empty"             # f has zero elements
FAIL_WRONG_LEN     = "f_wrong_length"      # f has wrong number of elements (after repair attempt)
FAIL_OUT_OF_RANGE  = "f_out_of_range"      # values not in [0,1] after clip attempt
FAIL_BAD_MEAN      = "f_bad_mean"          # mean != 0.5 after repair attempt
FAIL_INF_BOUND     = "f_inf_bound"         # bound computation returned inf (validity check failed)


@dataclass
class DiagnosedRollout:
    model_tag: str
    rollout_idx: int

    # Raw output
    raw_response: str = ""
    extracted_code: str = ""
    thinking: str = ""

    # Execution diagnostics
    fail_reason: str = FAIL_NONE
    exec_error_msg: str = ""
    f_len_before_repair: int = 0
    f_mean_before_repair: float = 0.0
    f_min_val: float = 0.0
    f_max_val: float = 0.0
    repair_applied: str = ""        # description of any auto-repair done

    # Final outcome
    reward: float = 0.0
    bound: float | None = None
    valid: bool = False
    latency_s: float = 0.0


# ─── Robust transition with full diagnostics ─────────────────────────────────────

def diagnosed_transition(
    env,
    action,
    seed_state,
    model_tag: str,
    rollout_idx: int,
    show_code: bool = False,
) -> DiagnosedRollout:
    """
    Execute one action through the environment with full diagnosis of failures.

    Auto-repairs two common issues before giving up:
      1. Wrong array length  -> resample via linear interpolation
      2. Mean != 0.5         -> rescale to enforce mean=0.5
    """
    import re

    result = DiagnosedRollout(
        model_tag=model_tag,
        rollout_idx=rollout_idx,
        raw_response=action.raw_response,
        extracted_code=action.code,
        thinking=action.thinking,
    )

    grid_size = env.GRID_SIZE

    # ── 1. Check we actually have code ───────────────────────────────────────────
    if not action.code or action.code.strip() in ("# no code found", "# parse error", ""):
        result.fail_reason = FAIL_NO_CODE
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_NO_CODE}")
        if show_code:
            print(f"\n  --- RAW RESPONSE (no code extracted) ---\n{action.raw_response[:800]}\n")
        return result

    # ── 2. Execute code ───────────────────────────────────────────────────────────
    local_vars = {
        "current_f": list(seed_state.content),
        "np": np,
        "math": math,
        "f": None,
    }
    try:
        exec(action.code, {"__builtins__": __builtins__}, local_vars)
    except Exception as e:
        result.fail_reason = FAIL_EXEC_ERROR
        result.exec_error_msg = f"{type(e).__name__}: {e}"
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_EXEC_ERROR} — {result.exec_error_msg}")
        if show_code:
            print(f"\n  --- CODE that crashed ---\n{action.code}\n  --- ERROR ---\n{traceback.format_exc()}\n")
        return result

    raw_f = local_vars.get("f")

    # ── 3. Check f was set ────────────────────────────────────────────────────────
    if raw_f is None:
        result.fail_reason = FAIL_NO_F
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_NO_F}")
        if show_code:
            print(f"\n  --- CODE that didn't set f ---\n{action.code}\n")
        return result

    # ── 4. Convert to numpy ───────────────────────────────────────────────────────
    try:
        f = np.array(raw_f, dtype=np.float64).flatten()
    except Exception as e:
        result.fail_reason = FAIL_WRONG_TYPE
        result.exec_error_msg = str(e)
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_WRONG_TYPE}")
        return result

    if f.size == 0:
        result.fail_reason = FAIL_EMPTY
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_EMPTY}")
        return result

    result.f_len_before_repair = int(f.size)
    result.f_mean_before_repair = float(f.mean())
    result.f_min_val = float(f.min())
    result.f_max_val = float(f.max())

    # ── 5. Auto-repair: wrong length → resample ───────────────────────────────────
    repairs = []
    if f.size != grid_size:
        old_x = np.linspace(0, 1, f.size)
        new_x = np.linspace(0, 1, grid_size)
        f = np.interp(new_x, old_x, f)
        repairs.append(f"resampled {result.f_len_before_repair}->{grid_size}")

    # ── 6. Clip to [0,1] ─────────────────────────────────────────────────────────
    n_out_of_range = int(np.sum((f < 0) | (f > 1)))
    if n_out_of_range > 0:
        f = np.clip(f, 0.0, 1.0)
        repairs.append(f"clipped {n_out_of_range} values to [0,1]")

    # ── 7. Auto-repair: mean != 0.5 → shift+scale ────────────────────────────────
    mean_err = abs(f.mean() - 0.5)
    if mean_err > 1e-6:
        # Shift to correct mean, then clip, then renormalise one more time
        f = f - f.mean() + 0.5
        f = np.clip(f, 0, 1)
        if abs(f.mean() - 0.5) > 1e-3:
            # Scale attempt
            scale = 0.5 / f.mean() if f.mean() > 0 else 1.0
            f = f * scale
            f = np.clip(f, 0, 1)
        repairs.append(f"mean corrected ({result.f_mean_before_repair:.4f}->~0.5)")

    result.repair_applied = "; ".join(repairs) if repairs else "none"

    # ── 8. Final validity checks ──────────────────────────────────────────────────
    if abs(f.mean() - 0.5) > 1e-3:
        result.fail_reason = FAIL_BAD_MEAN
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_BAD_MEAN} "
                       f"(mean={f.mean():.4f} after repair)")
        return result

    # ── 9. Compute bound ──────────────────────────────────────────────────────────
    F = np.fft.rfft(f)
    autocorr = np.fft.irfft(F * np.conj(F), n=len(f)) / len(f)
    bound = float(np.max(f.mean() - autocorr))

    if math.isinf(bound) or math.isnan(bound) or bound <= 0:
        result.fail_reason = FAIL_INF_BOUND
        logger.warning(f"    [{model_tag}] rollout {rollout_idx+1}: {FAIL_INF_BOUND}")
        return result

    result.bound = bound
    result.reward = 1.0 / bound
    result.valid = True
    result.fail_reason = FAIL_NONE
    return result


# ─── Model evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    model_tag: str,
    model,
    env,
    n_rollouts: int,
    temperature: float,
    seed_state,
    show_code: bool = False,
) -> list[DiagnosedRollout]:
    logger.info(f"  Evaluating [{model_tag}] — {n_rollouts} rollouts ...")
    sys_p = env.get_system_prompt()
    usr_p = "\n\n".join([
        env.get_problem_description(),
        "## Starting solution",
        env.format_state_for_prompt(seed_state),
        "## Your task",
        "Write Python code that sets `f` to a list of floats with a smaller upper bound.",
        "Constraints: all values in [0,1], mean(f)==0.5.",
        "Wrap code in ```python ... ```.",
    ])

    results = []
    for i in range(n_rollouts):
        t0 = time.time()
        actions = model.generate(sys_p, usr_p, n=1, temperature=temperature)
        action = actions[0]
        dr = diagnosed_transition(env, action, seed_state, model_tag, i, show_code)
        dr.latency_s = time.time() - t0

        if dr.valid:
            status = f"bound={dr.bound:.8f}"
            if dr.repair_applied != "none":
                status += f"  [repaired: {dr.repair_applied}]"
        else:
            status = f"INVALID ({dr.fail_reason})"
            if dr.repair_applied != "none":
                status += f"  [tried: {dr.repair_applied}]"

        logger.info(f"    rollout {i+1:3d}/{n_rollouts}: {status}  ({dr.latency_s:.1f}s)")
        results.append(dr)

    return results


# ─── Stats ────────────────────────────────────────────────────────────────────────

@dataclass
class ModelStats:
    model_tag: str
    n_rollouts: int
    n_valid: int
    valid_rate: float
    failure_breakdown: dict = field(default_factory=dict)
    n_repaired: int = 0

    mean_reward: float = 0.0
    std_reward: float = 0.0
    max_reward: float = 0.0

    mean_bound: float | None = None
    std_bound: float | None = None
    best_bound: float | None = None

    sota_bound: float = 0.0
    beats_sota: bool = False

    mean_latency_s: float = 0.0


def compute_stats(results: list[DiagnosedRollout], sota_bound: float) -> ModelStats:
    tag = results[0].model_tag
    rewards = [r.reward for r in results]
    valid = [r for r in results if r.valid]
    bounds = [r.bound for r in valid if r.bound is not None]
    repaired = [r for r in results if r.repair_applied not in ("none", "")]

    failures = {}
    for r in results:
        if not r.valid:
            failures[r.fail_reason] = failures.get(r.fail_reason, 0) + 1

    return ModelStats(
        model_tag=tag,
        n_rollouts=len(results),
        n_valid=len(valid),
        valid_rate=len(valid) / len(results) if results else 0,
        failure_breakdown=failures,
        n_repaired=len(repaired),

        mean_reward=float(np.mean(rewards)) if rewards else 0,
        std_reward=float(np.std(rewards)) if rewards else 0,
        max_reward=float(np.max(rewards)) if rewards else 0,

        mean_bound=float(np.mean(bounds)) if bounds else None,
        std_bound=float(np.std(bounds)) if bounds else None,
        best_bound=float(np.min(bounds)) if bounds else None,

        sota_bound=sota_bound,
        beats_sota=bool(bounds and min(bounds) < sota_bound),

        mean_latency_s=float(np.mean([r.latency_s for r in results])),
    )


# ─── Printing ─────────────────────────────────────────────────────────────────────

def print_failure_diagnosis(results: list[DiagnosedRollout], tag: str):
    invalid = [r for r in results if not r.valid]
    if not invalid:
        return
    print(f"\n  Failure diagnosis [{tag}]:")
    for r in invalid:
        print(f"    rollout {r.rollout_idx+1}: {r.fail_reason}", end="")
        if r.exec_error_msg:
            print(f"  —  {r.exec_error_msg}", end="")
        if r.f_len_before_repair and r.f_len_before_repair != 0:
            print(f"  |  f_len={r.f_len_before_repair}  f_mean={r.f_mean_before_repair:.4f}"
                  f"  f_range=[{r.f_min_val:.3f},{r.f_max_val:.3f}]", end="")
        if r.repair_applied not in ("none", ""):
            print(f"  |  tried repair: {r.repair_applied}", end="")
        print()


def print_per_rollout(base: list[DiagnosedRollout], trained: list[DiagnosedRollout]):
    print()
    print("  PER-ROLLOUT DETAILS")
    print(f"  {'#':>4}  {'Base':>16}  {'Trained':>16}  {'Delta':>12}  "
          f"{'Base fail':>16}  {'Trained fail':>16}")
    print("  " + "-" * 90)
    for b, t in zip(base, trained):
        bb = f"{b.bound:.8f}" if b.valid else f"({b.fail_reason})"
        tb = f"{t.bound:.8f}" if t.valid else f"({t.fail_reason})"
        if b.valid and t.valid:
            delta = f"{b.bound - t.bound:+.8f}"
        else:
            delta = "N/A"
        bf = "" if b.valid else b.fail_reason
        tf = "" if t.valid else t.fail_reason
        print(f"  {b.rollout_idx+1:>4}  {bb:>16}  {tb:>16}  {delta:>12}  {bf:>16}  {tf:>16}")
    print()


def print_comparison(base: ModelStats, trained: ModelStats):
    W = 72

    def row(label, bval, tval, fmt="{}", lower_is_better=False):
        bs = fmt.format(bval) if bval is not None else "N/A"
        ts = fmt.format(tval) if tval is not None else "N/A"
        if bval is not None and tval is not None and bval != tval:
            better = tval < bval if lower_is_better else tval > bval
            indicator = "▲ trained" if better else "▼ base"
        elif bval == tval:
            indicator = "= tie"
        else:
            indicator = ""
        print(f"  {label:<30s}  {bs:>12s}  {ts:>12s}   {indicator}")

    print()
    print("=" * W)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * W)
    print(f"  {'Metric':<30s}  {'Base':>12s}  {'Trained':>12s}   Change")
    print("-" * W)

    row("Valid rollouts",
        f"{base.n_valid}/{base.n_rollouts}",
        f"{trained.n_valid}/{trained.n_rollouts}", fmt="{}")
    row("Valid rate",         base.valid_rate,   trained.valid_rate,   fmt="{:.1%}")
    row("Auto-repaired",      base.n_repaired,   trained.n_repaired,   fmt="{}")

    if base.failure_breakdown or trained.failure_breakdown:
        all_reasons = set(base.failure_breakdown) | set(trained.failure_breakdown)
        print()
        print(f"  {'Failure reason':<30s}  {'Base':>12s}  {'Trained':>12s}")
        print("  " + "-" * 58)
        for reason in sorted(all_reasons):
            b_count = base.failure_breakdown.get(reason, 0)
            t_count = trained.failure_breakdown.get(reason, 0)
            print(f"  {reason:<30s}  {b_count:>12d}  {t_count:>12d}")

    print()
    row("Mean reward",        base.mean_reward,  trained.mean_reward,  fmt="{:.6f}")
    row("Std  reward",        base.std_reward,   trained.std_reward,   fmt="{:.6f}",  lower_is_better=True)
    row("Max  reward",        base.max_reward,   trained.max_reward,   fmt="{:.6f}")

    print()
    row("Best  bound ↓",      base.best_bound,   trained.best_bound,   fmt="{:.8f}",  lower_is_better=True)
    row("Mean  bound ↓",      base.mean_bound,   trained.mean_bound,   fmt="{:.8f}",  lower_is_better=True)
    row("Std   bound",        base.std_bound,    trained.std_bound,    fmt="{:.8f}",  lower_is_better=True)

    print()
    print(f"  {'SOTA bound':<30s}  {base.sota_bound:.8f}")
    row("Beats SOTA",
        "yes" if base.beats_sota else "no",
        "yes" if trained.beats_sota else "no", fmt="{}")
    row("Mean latency/rollout", base.mean_latency_s, trained.mean_latency_s,
        fmt="{:.1f}s", lower_is_better=True)

    print("-" * W)
    print()

    # Verdict
    if base.n_valid == 0 and trained.n_valid == 0:
        print("  ⚠  Both models produced 0 valid rollouts.")
        print("     Common causes:")
        print("       • GRID_SIZE=500 is too large — the LLM returns a short array")
        print("         Fix: reduce ErdosEnvironment.GRID_SIZE to 50 or 100")
        print("       • The model generates explanation text instead of code")
        print("         Fix: use --show-code to inspect raw outputs")
        print("       • scipy/other imports fail inside exec()")
        print("         Fix: add them to the exec locals in _execute_action_code")
    elif trained.n_valid == 0 and base.n_valid > 0:
        print("  ⚠  Base model produced valid rollouts but trained model did not.")
        print("     The LoRA fine-tuning may have caused the model to produce")
        print("     different code structure that hits a different failure mode.")
    elif trained.best_bound is not None and base.best_bound is not None:
        delta = base.best_bound - trained.best_bound
        if delta > 0:
            print(f"  ✓  Trained model improved best bound by {delta:.8f}  "
                  f"({delta/base.best_bound*100:.4f}%)")
        elif delta < 0:
            print(f"  ✗  Base model achieved a better bound (trained worse by {-delta:.8f})")
        else:
            print(f"  =  No difference in best bound.")

        dr = trained.mean_reward - base.mean_reward
        if dr > 1e-9:
            print(f"  ✓  Mean reward: trained +{dr/max(base.mean_reward,1e-12)*100:.2f}% vs base")
        elif dr < -1e-9:
            print(f"  ✗  Mean reward: trained -{abs(dr)/max(base.mean_reward,1e-12)*100:.2f}% vs base")

    print("=" * W)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs LoRA-trained model with full failure diagnostics"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to LoRA checkpoint dir (e.g. ./checkpoints/final)")
    parser.add_argument("--model",
                        default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    parser.add_argument("--env",         default="erdos")
    parser.add_argument("--rollouts",    type=int,   default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-4bit",     action="store_true")
    parser.add_argument("--show-code",   action="store_true",
                        help="Print raw LLM output for every failed rollout")
    parser.add_argument("--output",      default=None,
                        help="Save full results to JSON")
    parser.add_argument("--verbose",     "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Environment ───────────────────────────────────────────────────────────────
    from environments import ErdosEnvironment
    env_map = {"erdos": ErdosEnvironment}
    if args.env not in env_map:
        print(f"Unknown env '{args.env}'. Available: {list(env_map)}")
        sys.exit(1)
    env = env_map[args.env]()
    _, sota_reward = env.get_sota()
    sota_bound = 1.0 / sota_reward

    seed_state = env.get_initial_state()
    logger.info(f"Seed state: bound={seed_state.metadata.get('bound','?'):.6f}")

    # ── Base model ────────────────────────────────────────────────────────────────
    from unsloth_model import UnslothModel, UnslothModelConfig

    cfg = UnslothModelConfig(model_name=args.model, load_in_4bit=not args.no_4bit, lora_r=32)

    print(f"\nLoading BASE model: {args.model}")
    base_model = UnslothModel(cfg)
    base_model.load()
    base_results = evaluate_model(
        "base", base_model, env, args.rollouts, args.temperature, seed_state, args.show_code
    )

    # ── Trained model ─────────────────────────────────────────────────────────────
    print(f"\nLoading TRAINED model: {args.model} + {args.checkpoint}")
    trained_cfg = UnslothModelConfig(model_name=args.model, load_in_4bit=not args.no_4bit, lora_r=32)
    trained_model = UnslothModel(trained_cfg)
    trained_model.load()
    trained_model.load_lora(args.checkpoint)
    trained_results = evaluate_model(
        "trained", trained_model, env, args.rollouts, args.temperature, seed_state, args.show_code
    )

    # ── Diagnose failures ─────────────────────────────────────────────────────────
    print_failure_diagnosis(base_results, "base")
    print_failure_diagnosis(trained_results, "trained")

    # ── Stats + comparison ────────────────────────────────────────────────────────
    base_stats    = compute_stats(base_results,    sota_bound)
    trained_stats = compute_stats(trained_results, sota_bound)

    print_per_rollout(base_results, trained_results)
    print_comparison(base_stats, trained_stats)

    # ── Save ──────────────────────────────────────────────────────────────────────
    if args.output:
        payload = {
            "config": {
                "model": args.model, "checkpoint": args.checkpoint,
                "env": args.env, "rollouts": args.rollouts,
                "temperature": args.temperature, "sota_bound": sota_bound,
                "seed_bound": seed_state.metadata.get("bound"),
            },
            "base_stats":    asdict(base_stats),
            "trained_stats": asdict(trained_stats),
            "base_rollouts":    [asdict(r) for r in base_results],
            "trained_rollouts": [asdict(r) for r in trained_results],
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
