# GEPA reproduction tasks. Requires: uv, ollama.

setup:
    uv venv && uv pip install -e .

preflight:
    bash scripts/preflight.sh

# Validate the whole pipeline in minutes (tiny splits, budget 30)
smoke:
    python -m experiments.aime.main --smoke

# Real run: overnight on an M-series laptop
run budget="300":
    python -m experiments.aime.main --budget {{budget}}

# Baseline only (seed prompt on the test set, no optimization)
baseline:
    python -m experiments.aime.main --baseline-only
