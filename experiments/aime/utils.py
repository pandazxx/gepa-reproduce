"""AIME task definition: dataset, solver module, metric.

Adapted from the official example at gepa-ai/gepa examples/aime_math, which follows
the paper's AIME setup (App. E.1): train/val = AIME 2022-2024 split 45/45,
test = AIME 2025.
"""

import random

import dspy
from datasets import load_dataset


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


predictor = dspy.ChainOfThought(MathSolverSignature)


def run_llm(example, prompt: str):
    predictor.predict.signature.instructions = prompt
    return predictor(input=example.input)


def math_metric(example, prediction):
    """Score (exact integer match) plus feedback text for reflection (the paper's mu_f)."""
    correct_answer, written_solution = int(example.answer), getattr(example, "solution", "")
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\n"
        "Think about what takeaways you can learn from this solution to improve "
        "your future answers and approach to similar problems"
        if written_solution
        else ""
    )

    try:
        llm_answer = int(prediction.answer)
    except (ValueError, TypeError):
        feedback_text = (
            f"The final answer must be a valid integer and nothing else. You responded with "
            f"'{prediction.answer}', which couldn't be parsed as a python integer. "
            f"The correct answer is '{correct_answer}'.{solution_suffix}"
        )
        return 0.0, feedback_text

    score = float(correct_answer == llm_answer)
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


def load_math_dataset(seed: int = 0):
    train_split = []
    test_split = []

    for item in load_dataset("AI-MO/aimo-validation-aime", "default", split="train"):
        train_split.append(
            dspy.Example(input=item["problem"], solution=item["solution"], answer=item["answer"]).with_inputs("input")
        )
    random.Random(seed).shuffle(train_split)

    for item in load_dataset("MathArena/aime_2025", "default", split="train"):
        test_split.append(dspy.Example(input=item["problem"], answer=item["answer"]).with_inputs("input"))

    half = len(train_split) // 2
    return train_split[:half], train_split[half:], test_split


def evaluate_on_dataset(prompt: str, dataset, num_threads: int = 4) -> float:
    predictor.predict.signature.instructions = prompt

    def dspy_metric(example, prediction, trace=None):
        return math_metric(example, prediction)[0]

    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=dspy_metric,
        num_threads=num_threads,
        display_progress=True,
    )
    return evaluator(predictor).score / 100.0
