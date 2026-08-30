# gepa-reproduce

Reproduction of [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) (Agrawal et al., ICLR 2026 Oral), scoped to what runs on a laptop.

- Paper code: https://github.com/gepa-ai/gepa (this repo uses the `gepa` package as the optimizer — we reproduce the *experiments*, not the algorithm implementation)
- Study notes / reading guide: [pandazxx/research → topics/prompt-learning](https://github.com/pandazxx/research/tree/master/topics/prompt-learning)

## Scope and plan

Reproduction ladder (from the study notes), cheapest first:

| Phase | What | Status |
|---|---|---|
| 1 | **AIME**: baseline vs GEPA with a local model (this repo, `experiments/aime/`) | bootstrapped |
| 2 | LiveBench-Math (same recipe, different dataset) | todo |
| 3 | IFBench (2-module system + constraint checkers) | todo |
| 4 | MIPROv2 comparison (DSPy), API-model track (GPT-4.1 Mini, ~$15/benchmark) | todo |
| 5 | HotpotQA / HoVer (needs a Wikipedia retrieval index) | maybe |

AIME is Phase 1 because it needs zero extra infra (answer = integer match) and has the smallest budget in the paper (1,839 rollouts). Note it is also the one benchmark where GRPO *beat* GEPA on Qwen3 8B (38.0 vs 32.0, Table 1) — so the interesting local result is GEPA vs baseline and vs the paper's reported deltas, not "GEPA wins everything."

## Hardware reality (M3 Pro, 18 GB)

The target setup is **Ollama + `qwen3.5:9b`** running locally. Honest caveats:

- 18 GB unified memory forces a **quantized** model (Ollama's default q4 ≈ 6 GB; q8 ≈ 10 GB also fits). This is *not* the paper's bf16 checkpoint, and `qwen3.5:9b` is not the paper's Qwen3 8B — treat results as a **qualitative** reproduction (does GEPA's curve climb, does the optimized prompt transfer) rather than expecting Table 1 numbers.
- Expect ~10–20 tok/s. A 300-rollout run with a thinking-style model is an **overnight job**; always run `--smoke` first (a few minutes) to validate the pipeline.
- Keep `max_workers` low (2–4): Ollama parallelism is limited by memory, not cores.

## Quickstart

If you have Nix with flakes enabled, `nix develop` drops you into a shell with Python 3.11, `uv`, `just`, `ollama`, `curl`, and `git` already on `PATH` — skip straight to `just setup`. Copy `.env.example` to `.env` first (`OLLAMA_API_BASE`, optional `OPENAI_API_KEY`) and the devshell auto-loads it on entry.

With [just](https://github.com/casey/just) (see the `justfile` for all shortcuts):

```bash
just setup       # venv + deps (uv)
just pull        # ollama pull qwen3.5:9b
just serve       # ollama serve with OLLAMA_CONTEXT_LENGTH=16384 (separate terminal)
just preflight   # checks Ollama, model, context window, datasets
just smoke       # validate the pipeline in ~minutes (tiny splits, budget 30)
just run         # real run, budget 300 (overnight); just run 600 for a bigger budget
just results     # show scores + the best evolved prompt
```

Other shortcuts: `just baseline` (seed prompt on the test set only), `just run-hybrid` (local rollouts + `gpt-4.1-mini` reflection; needs `OPENAI_API_KEY`), `just clean`. Override the model inline: `just model=<ollama-tag> smoke`.

Or the same steps by hand:

```bash
# 1. Ollama setup (once)
ollama pull qwen3.5:9b
# GEPA's reflection prompts are long — raise the context window (default is too small):
OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=2 ollama serve

# 2. Python setup (uv recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# 3. Preflight — checks Ollama, the model, and the datasets
bash scripts/preflight.sh

# 4. Smoke test (~minutes): tiny splits, budget 30
python -m experiments.aime.main --smoke

# 5. Real run (overnight): budget 300 by default; paper budget is 1839
python -m experiments.aime.main --budget 300
```

Results land in `outputs/aime/`: the GEPA run dir (resumable state, per-candidate logs) plus `best_prompt.txt` and `results.json` (baseline vs optimized test scores).

Useful flags (`python -m experiments.aime.main --help`):

- `--task-model` / `--reflection-model` — any LiteLLM string. Default: `ollama_chat/qwen3.5:9b` for both (self-reflection, like the paper's Qwen runs). If local reflections look weak, point only the reflection model at an API model (e.g. `--reflection-model openai/gpt-4.1-mini`) — the "cheap executor, strong reflector" configuration.
- `--budget` — total metric calls (rollouts), the paper's Eq. 2 budget `B`.
- `--workers` — parallel evaluations; match `OLLAMA_NUM_PARALLEL`.
- `--skip-baseline` — skip the pre-optimization test-set eval.

## Experiment design (matches the paper)

- **Data** (same recipe as the paper §E.1 and the official `examples/aime_math`): train/val = AIME 2022–2024, 90 problems ([AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)) split 45/45; test = AIME 2025, 30 problems ([MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)).
- **System**: single-module ChainOfThought (the paper uses single-step CoT for AIME).
- **Metric μ**: exact integer match. **Feedback μ_f**: correctness + the correct answer + the written solution when available (so reflection can extract *why*).
- **Optimizer**: `gepa.optimize_anything` with Pareto candidate selection — the actual GEPA engine from the paper's authors.
- **Sampling**: temperature 0.6 / top-p 0.95 (paper's Qwen3 settings, §E.2).
