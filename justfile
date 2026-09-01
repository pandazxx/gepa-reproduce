# GEPA reproduction shortcuts. Requires: uv, ollama, just.
# Override models per-invocation, e.g.: just model=qwen3.5:9b smoke

model := env_var_or_default("TASK_MODEL", "qwen3.5:9b")
reflection := env_var_or_default("REFLECTION_MODEL", "ollama_chat/" + model)
ctx := env_var_or_default("OLLAMA_CONTEXT_LENGTH", "16384")
parallel := env_var_or_default("OLLAMA_NUM_PARALLEL", "2")

# List available targets
default:
    @just --list

# Create venv and install dependencies
setup:
    uv venv && uv pip install -e .

# Download the model into Ollama
pull:
    ollama pull {{model}}

# Start the Ollama server with the context window GEPA needs (foreground)
serve:
    OLLAMA_CONTEXT_LENGTH={{ctx}} OLLAMA_NUM_PARALLEL={{parallel}} ollama serve

# Check Ollama, model, context window, and dataset access
preflight:
    TASK_MODEL={{model}} bash scripts/preflight.sh

# Validate the whole pipeline in minutes (tiny splits, budget 30)
smoke:
    uv run python -m experiments.aime.main --smoke --task-model ollama_chat/{{model}} --reflection-model {{reflection}}

# Baseline only: seed prompt on the AIME-2025 test set, no optimization
baseline:
    uv run python -m experiments.aime.main --baseline-only --task-model ollama_chat/{{model}}

# Real run (overnight on a laptop). Usage: just run [budget]
run budget="300":
    uv run python -m experiments.aime.main --budget {{budget}} --task-model ollama_chat/{{model}} --reflection-model {{reflection}}

# Local rollouts + API reflection model (cheap executor, strong reflector). Needs OPENAI_API_KEY.
run-hybrid budget="300" reflector="openai/gpt-4.1-mini":
    uv run python -m experiments.aime.main --budget {{budget}} --task-model ollama_chat/{{model}} --reflection-model {{reflector}}

# Show scores and the best evolved prompt from the last run
results:
    @cat outputs/aime/results.json 2>/dev/null || echo "no results yet - run: just smoke"
    @echo "---"
    @cat outputs/aime/best_prompt.txt 2>/dev/null || true

# Delete all run outputs
clean:
    rm -rf outputs/
