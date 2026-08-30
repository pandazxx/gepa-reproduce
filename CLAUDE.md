# CLAUDE.md — Agent Instructions for gepa-reproduce

This file is the harness for coding agents (Claude Code, Codex, etc.) working in this repo. It is written so a smaller model can execute tasks correctly without re-deriving context. Follow it exactly; when it conflicts with your instincts, this file wins.

## What this project is

A laptop-scale reproduction of the GEPA paper (arXiv:2507.19457, ICLR 2026 Oral): optimizing LLM prompts via reflective evolution using the authors' `gepa` package. We reproduce the *experiments*, not the optimizer internals — never reimplement GEPA's algorithm; always call `gepa.optimize_anything`.

- Paper PDF + reading guide + study notes: https://github.com/pandazxx/research/tree/master/topics/prompt-learning — read `gepa-study-notes.md` there if you need paper context.
- Target hardware: one MacBook Pro (M3 Pro, 18 GB). Models run through **Ollama** (`qwen3.5:9b`, quantized). This constrains every design decision: small parallelism, modest budgets, overnight wall-clocks.

## Ground rules

1. **Never start a long optimization run yourself.** Runs with budget > 30 take hours and belong to the user's laptop. You may run `just smoke` (budget 30, minutes) to validate changes if an Ollama server is reachable; otherwise validate with `python -m py_compile` and dry imports.
2. **Reproducibility over cleverness.** Every experiment: fixed seed (default 0), splits defined in code, config echoed into `results.json`. Never shuffle without a seeded RNG.
3. **Don't silently change comparability-critical settings**: sampling (temp 0.6, top-p 0.95 — paper §E.2), data splits, the metric, or the seed prompt. If a task requires changing one, call it out in the PR description.
4. **Outputs are git-ignored** (`outputs/`). Never commit run artifacts; summarize them in `RESULTS.md` instead.
5. **Small PRs to `main`**, one phase or fix per branch. Update the phase table in `README.md` when a phase lands.

## Repo map

```
experiments/aime/        Phase 1 (done): utils.py = data/solver/metric, main.py = CLI runner
scripts/preflight.sh     Environment checks (Ollama up, model pulled, datasets reachable)
justfile                 All user-facing entry points (just --list)
RESULTS.md               Append-only log of completed runs
```

Every new benchmark phase copies the `experiments/aime/` pattern: a `utils.py` (dataset loader, dspy program, metric-with-feedback) + `main.py` (argparse CLI, baseline eval, `optimize_anything`, test eval, `results.json`). Keep that structure; a new phase should diff almost entirely in `utils.py`.

## Critical technical gotchas (violating these breaks the run, often silently)

- **Evaluator signature is inspected by name.** The fitness function must be `def evaluate(candidate, example) -> tuple[float, side_info]`. The second parameter MUST be named `example` — GEPA's wrapper forwards kwargs by parameter name; any other name crashes with a missing-argument error.
- **Ollama context window.** GEPA reflection prompts are long. The server must run with `OLLAMA_CONTEXT_LENGTH=16384` (the `just serve` target does this). Symptom of getting it wrong: reflections that ignore most of the feedback, no error raised.
- **Model strings.** Local models are `ollama_chat/<tag>` (LiteLLM), endpoint from `OLLAMA_API_BASE` (default `http://localhost:11434`). The reflection LM in `ReflectionConfig` takes the same LiteLLM string, or any Python callable `(str) -> str`.
- **The dspy predictor is module-global** in `utils.py`; the candidate prompt is injected via `predictor.predict.signature.instructions = prompt`. Concurrent evaluation with *different* prompts on the same predictor is a race — GEPA evaluates one candidate at a time across examples, which is safe; do not "optimize" this into cross-candidate parallelism.
- **Budget accounting.** `max_metric_calls` counts *every* fitness call: minibatch rollouts AND val-set evaluations (a val pass costs `len(valset)` calls per accepted candidate). Smoke mode must keep valset tiny or the budget disappears into one val pass.
- **Metric feedback is the learning signal.** `math_metric`-style functions return `(score, feedback_text)`; the feedback text (correct answer, written solution, failure mode) is what the reflection model learns from. When writing a new metric, invest there — a bare 0/1 starves the optimizer.

## How to run things

```
just --list        # all targets
just preflight     # env sanity
just smoke         # tiny end-to-end validation, ~minutes — run after any pipeline change
just run [budget]  # real run — USER ONLY, do not launch unprompted
just results       # scores + best prompt of last run
```

## Phase roadmap (implementation notes per phase)

**Phase 2 — LiveBench-Math** (`experiments/livebench_math/`): paper §E.1 uses the LiveBench math subset (368 questions), shuffled with python `random` seed 0, split into equal thirds train/val/test. Load from HF `livebench/math`. Answers need normalization (strings, LaTeX) — write the metric to strip formatting before comparison and return what-was-expected in the feedback text. Reuse the AIME CLI unchanged.

**Phase 3 — IFBench** (`experiments/ifbench/`): first multi-module system. Two dspy modules: (1) draft an answer to the query, (2) rewrite the draft to satisfy the output constraints. Train/val from the IF-RLVR training data (HF: `allenai/IF_multi_constraints_upto5`), test = IFBench's own test set (unseen constraint types). Metric = programmatic constraint checkers from the IFBench release; feedback text = which constraints passed/failed, by name. This phase must switch to the dict-candidate API (one component per module prompt) — see `gepa.gepa_launcher.optimize_anything` or the DSPy `dspy.GEPA` integration.

**Phase 4 — MIPROv2 comparison**: same task/data/metric as an existing phase, optimizer swapped to `dspy.MIPROv2(auto="heavy")`. Report the same `results.json` schema so rows are comparable. Also the API-model track (GPT-4.1 Mini) if the user provides a key — budget ≈ $15/benchmark.

**Definition of done for any phase**: `just smoke` equivalent passes; `main.py --help` documents all flags; `results.json` schema matches AIME's; README phase table row flipped; a row appended to `RESULTS.md` after the user's real run.

## When results come in

After the user reports a completed run, append to `RESULTS.md`: date, phase, task model, reflection model, budget, baseline score, optimized score, delta, link/path to best prompt, and one sentence of interpretation against the paper's reference numbers (AIME reference: Qwen3 8B baseline 27.33 → GEPA 32.00, Table 1; note our setup is quantized and a different model, so compare deltas, not absolutes).
