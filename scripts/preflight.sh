#!/usr/bin/env bash
# Preflight checks: Ollama up, model pulled, context window sane, datasets reachable.
set -u
BASE="${OLLAMA_API_BASE:-http://localhost:11434}"
MODEL="${TASK_MODEL:-qwen3.5:9b}"
MODEL="${MODEL#ollama_chat/}"
FAIL=0

echo "1) Ollama server at $BASE ..."
if curl -sf "$BASE/api/tags" >/dev/null; then
  echo "   OK"
else
  echo "   FAIL — start it with: OLLAMA_CONTEXT_LENGTH=16384 OLLAMA_NUM_PARALLEL=2 ollama serve"
  exit 1
fi

echo "2) Model '$MODEL' pulled ..."
if curl -sf "$BASE/api/tags" | grep -q "\"$MODEL\""; then
  echo "   OK"
else
  echo "   FAIL — run: ollama pull $MODEL"
  FAIL=1
fi

echo "3) Generation + context window ..."
RESP=$(curl -sf "$BASE/api/generate" -d "{\"model\":\"$MODEL\",\"prompt\":\"Say OK\",\"stream\":false}" || true)
if [ -n "$RESP" ]; then
  echo "   OK (model responds)"
  echo "   NOTE: GEPA reflection prompts are long. Ensure the server was started with"
  echo "         OLLAMA_CONTEXT_LENGTH=16384 (or set num_ctx in a Modelfile) — silent"
  echo "         truncation at the default context is the #1 cause of bad reflections."
else
  echo "   FAIL — model did not respond"
  FAIL=1
fi

echo "4) HuggingFace datasets reachable ..."
python - <<'EOF' && echo "   OK" || { echo "   FAIL — check network / hf cache"; exit 1; }
from datasets import load_dataset
load_dataset("AI-MO/aimo-validation-aime", "default", split="train[:1]")
load_dataset("MathArena/aime_2025", "default", split="train[:1]")
EOF

exit $FAIL
