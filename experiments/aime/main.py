"""GEPA reproduction, Phase 1: AIME with a local Ollama model.

Usage:
    python -m experiments.aime.main --smoke          # validate the pipeline (~minutes)
    python -m experiments.aime.main --budget 300     # real run (overnight on a laptop)

Defaults target `ollama_chat/qwen3.5:9b` for both the task model and the
reflection model (self-reflection, matching the paper's Qwen3 runs). Point
--reflection-model at an API model to test the cheap-executor/strong-reflector split.
"""

import argparse
import json
import os
from pathlib import Path

import dspy
from dspy.utils.exceptions import AdapterParseError

from experiments.aime.utils import evaluate_on_dataset, load_math_dataset, math_metric, parse_failure_metric, run_llm
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, SideInfo, optimize_anything

# litellm's default request timeout (600s) is tuned for hosted APIs; a quantized
# model on a laptop CPU/GPU can take longer per rollout, especially with a long
# reflection prompt. Both the task LM and reflection LM forward this to
# litellm.completion(timeout=...).
DEFAULT_TIMEOUT_S = 1800

INITIAL_PROMPT = (
    "Solve the math problem carefully. Break down the steps and provide the final answer as a single number."
)


def evaluate(candidate: str, example) -> tuple[float, SideInfo]:
    # GEPA re-raises evaluator exceptions, so an unparseable rollout (thinking
    # model burns the whole token budget on reasoning and never emits the answer
    # fields) would otherwise kill the entire run. Score it 0 with feedback the
    # reflection model can learn from instead.
    try:
        prediction = run_llm(example, candidate)
    except AdapterParseError as error:
        score, feedback = parse_failure_metric(example, error)
        return score, {
            "score": score,
            "input": example.input,
            "output": "",
            "reasoning": "",
            "execution_feedback": feedback,
        }
    score, feedback = math_metric(example, prediction)
    side_info = {
        "score": score,
        "input": example.input,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }
    return score, side_info


def main():
    parser = argparse.ArgumentParser(description="GEPA on AIME with a local model")
    parser.add_argument("--task-model", default=os.getenv("TASK_MODEL", "ollama_chat/qwen3.5:9b"))
    parser.add_argument("--reflection-model", default=os.getenv("REFLECTION_MODEL", "ollama_chat/qwen3.5:9b"))
    parser.add_argument("--api-base", default=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
    parser.add_argument("--budget", type=int, default=300, help="Max metric calls (paper uses 1839 for AIME)")
    parser.add_argument("--workers", type=int, default=2, help="Parallel evals; match OLLAMA_NUM_PARALLEL")
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.6, help="Paper's Qwen3 setting (App. E.2)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="Per-request LM timeout in seconds, forwarded to litellm (default: %(default)s). "
        "Raise this if slow local inference trips litellm.Timeout.",
    )
    parser.add_argument("--run-dir", default="outputs/aime")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Tiny splits + budget 30 to validate the pipeline")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-only", action="store_true", help="Evaluate the seed prompt on test and exit")
    args = parser.parse_args()

    # LiteLLM resolves ollama_chat/* endpoints from this env var (reflection LM included)
    os.environ.setdefault("OLLAMA_API_BASE", args.api_base)

    solver_lm = dspy.LM(
        args.task_model,
        api_base=args.api_base if args.task_model.startswith("ollama") else None,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    dspy.configure(lm=solver_lm)

    trainset, valset, testset = load_math_dataset(seed=args.seed)
    if args.smoke:
        trainset, valset, testset = trainset[:6], valset[:4], testset[:4]
        args.budget = min(args.budget, 30)
        print(f"SMOKE TEST: {len(trainset)} train / {len(valset)} val / {len(testset)} test, budget {args.budget}")
    print(f"Task model: {args.task_model} | Reflection model: {args.reflection_model}")
    print(f"Data: {len(trainset)} train / {len(valset)} val / {len(testset)} test | budget {args.budget}")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_score = None
    if args.baseline_only or not args.skip_baseline:
        print("\nEvaluating baseline (initial prompt) on test set...")
        baseline_score = evaluate_on_dataset(INITIAL_PROMPT, testset, num_threads=args.workers)
        print(f"Baseline test score: {baseline_score:.2%}")
    if args.baseline_only:
        (run_dir / "results.json").write_text(
            json.dumps({"task_model": args.task_model, "baseline_test_score": baseline_score}, indent=2)
        )
        return

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            max_metric_calls=args.budget,
            track_best_outputs=True,
            parallel=args.workers > 1,
            max_workers=args.workers,
            cache_evaluation=True,
            seed=args.seed,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            reflection_lm_kwargs={"timeout": args.timeout},
        ),
    )

    result = optimize_anything(
        seed_candidate=INITIAL_PROMPT,
        evaluator=evaluate,
        dataset=trainset,
        valset=valset,
        config=config,
    )

    best_prompt = result.best_candidate
    print(f"\nBest prompt found:\n{'-' * 70}\n{best_prompt}\n{'-' * 70}")
    (run_dir / "best_prompt.txt").write_text(str(best_prompt))

    print("\nEvaluating optimized prompt on test set...")
    optimized_score = evaluate_on_dataset(str(best_prompt), testset, num_threads=args.workers)

    results = {
        "task_model": args.task_model,
        "reflection_model": args.reflection_model,
        "budget": args.budget,
        "smoke": args.smoke,
        "baseline_test_score": baseline_score,
        "optimized_test_score": optimized_score,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 70}")
    if baseline_score is not None:
        print(f"Baseline:  {baseline_score:.2%}")
    print(f"Optimized: {optimized_score:.2%}")
    if baseline_score is not None:
        print(f"Delta:     {optimized_score - baseline_score:+.2%}")
    print(f"Artifacts in {run_dir}/ (best_prompt.txt, results.json, GEPA state/logs)")


if __name__ == "__main__":
    main()
