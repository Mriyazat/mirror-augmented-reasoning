#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Frontier-LLM evaluation chain.
#
# Runs `predict_with_frontier_llm` against an OpenAI-compatible endpoint for a
# stratified subsample of each evaluation split, using zero-temperature
# decoding for reproducibility. Resumes safely on re-launch.
#
# Usage:
#   export OPENAI_API_KEY=...
#   bash scripts/examples/run_frontier_chain.sh
#
# Environment overrides (all optional):
#   DDI_ROOT             repository root (default: current working dir)
#   OPENAI_MODEL         OpenAI-API-compatible model id (default: gpt-4o)
#   N_PAIRS              pairs per split (default: 500)
#   EVAL_PROMPTS_DIR     directory containing the prompt + manifest JSONL files
# -----------------------------------------------------------------------------
set -euo pipefail

cd "${DDI_ROOT:-$(pwd)}"

# Activate a local venv if one is present; otherwise rely on the active env.
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set." >&2
    exit 1
fi

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
N_PAIRS="${N_PAIRS:-500}"
EVAL_PROMPTS_DIR="${EVAL_PROMPTS_DIR:-outputs/eval_prompts}"

mkdir -p logs "$EVAL_PROMPTS_DIR"

for SP in drug_cold random_full; do
    echo "=== split: $SP ==="
    python -u -m src.inference.predict_with_frontier_llm \
        --prompts   "$EVAL_PROMPTS_DIR/${SP}_test_5000_stratified.with_neighbors.prompts.jsonl" \
        --manifest  "$EVAL_PROMPTS_DIR/${SP}_test_5000_stratified.manifest.jsonl" \
        --output    "$EVAL_PROMPTS_DIR/pred_${OPENAI_MODEL//\//_}_${SP}_${N_PAIRS}.jsonl" \
        --provider openai --model "$OPENAI_MODEL" \
        --n_pairs "$N_PAIRS" --stratified --temperature 0.0 --max_tokens 1024 --resume \
        2>&1 | tee "logs/frontier_${OPENAI_MODEL//\//_}_${SP}_${N_PAIRS}.log"
done

echo "ALL DONE"
