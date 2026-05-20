#!/bin/bash
# ============================================================================
# Teacher generation launcher — run INSIDE a salloc / srun allocation.
#
# Runs ONE teacher model (bring up vLLM server → warmup → generate.py →
# teardown). Safe to invoke multiple times in sequential sallocs: the
# generator's --resume picks up where a previous SIGTERM'd job stopped.
#
# Usage:
#   salloc --time=8:00:00 --cpus-per-task=32 --mem=256G \
#          --gres=gpu:h100:4 --account=${SLURM_ACCOUNT:-your_account}
#   cd /path/to/EMNLP2026_DDI_Verifier_Release
#   source scripts/cluster_examples/activate_env.sh
#   bash scripts/cluster_examples/run_teacher.sh llama-3.3-70b
#
# Args:
#   $1  teacher_id   — one of the IDs in configs/prm_rubric.yaml:teacher_models
#                      (llama-3.3-70b | qwen-2.5-72b | deepseek-r1-distill-70b)
#   $2  split        — optional; defaults to subset25k
#   $3  limit        — optional; cap the number of pairs (dev/debug). Empty = all.
#
# Handles:
#   - vLLM server lifecycle (start in background, wait until /health, tail logs)
#   - SIGTERM propagation to the Python generator (triggers clean per-pair exit)
#   - Automatic server teardown on any exit path
# ============================================================================
set -euo pipefail

# Raise the open-files soft limit. vLLM warns at 51200; at 4 x H100 TP sharding
# plus many concurrent HTTP requests, long-running jobs can flirt with that
# limit. 65535 is safely under the hard limit on most large clusters.
ulimit -n 65535 2>/dev/null || true

TEACHER_ID="${1:?need teacher id (llama-3.3-70b | qwen-2.5-72b | deepseek-r1-distill-70b)}"
SPLIT="${2:-subset25k}"
LIMIT="${3:-}"

# Map teacher_id → HF model path (keep in sync with configs/prm_rubric.yaml)
case "$TEACHER_ID" in
  llama-3.3-70b)             MODEL_PATH="meta-llama/Llama-3.3-70B-Instruct" ;;
  qwen-2.5-72b)              MODEL_PATH="Qwen/Qwen2.5-72B-Instruct" ;;
  deepseek-r1-distill-70b)   MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Llama-70B" ;;
  *) echo "[run_teacher] unknown teacher_id: $TEACHER_ID"; exit 1 ;;
esac

# Read N from the rubric so we don't drift (single source of truth).
N_CAND=$(python - <<'PY'
from src.teacher.schema import load_rubric
print(load_rubric()["teacher_generation"]["n_candidates_per_pair"])
PY
)

# Per-teacher seed (decorrelated; matches generate.py's offset logic).
case "$TEACHER_ID" in
  llama-3.3-70b)             SEED=42 ;;
  qwen-2.5-72b)              SEED=10042 ;;
  deepseek-r1-distill-70b)   SEED=20042 ;;
esac

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_LOG="${DDI_OUTPUTS:-outputs}/teacher/vllm_${TEACHER_ID}.log"
mkdir -p "$(dirname "$VLLM_LOG")"

echo "============================================================"
echo "[run_teacher] teacher_id:  $TEACHER_ID"
echo "[run_teacher] model_path:  $MODEL_PATH"
echo "[run_teacher] split:       $SPLIT"
echo "[run_teacher] n_candidates:$N_CAND  (from rubric)"
echo "[run_teacher] seed:        $SEED"
echo "[run_teacher] vllm_port:   $VLLM_PORT"
echo "[run_teacher] vllm_log:    $VLLM_LOG"
if [ -n "${MANIFEST_FILE:-}" ]; then
    echo "[run_teacher] manifest:    $MANIFEST_FILE"
fi
if [ -n "${GUIDANCE_FILE:-}" ]; then
    echo "[run_teacher] guidance:    $GUIDANCE_FILE"
fi
if [ -n "${PROMPT_VERSION_SUFFIX:-}" ]; then
    echo "[run_teacher] prompt_suffix: $PROMPT_VERSION_SUFFIX"
fi
echo "============================================================"

# ---------- bring up vLLM server ----------
# TP=4 assumes 4×H100 from the salloc. bfloat16 fits 70-72B comfortably.
# --served-model-name must match what the OpenAI client sends.
echo "[run_teacher] starting vLLM server…"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_PATH" \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --port "$VLLM_PORT" \
    --disable-log-requests \
    --disable-custom-all-reduce \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Teardown helper — runs on EXIT/INT/TERM
cleanup() {
    local ec=$?
    echo "[run_teacher] tearing down (exit=$ec)…"
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        # Wait up to 60s for graceful shutdown, then force.
        for _ in $(seq 1 60); do
            kill -0 "$VLLM_PID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
    wait "$VLLM_PID" 2>/dev/null || true
    echo "[run_teacher] vllm teardown complete (exit=$ec)"
}
trap cleanup EXIT INT TERM

# ---------- wait for server /health ----------
# 70B models take 5-10 min to load from $SCRATCH; be patient.
echo -n "[run_teacher] waiting for vLLM /health "
for i in $(seq 1 180); do   # up to 30 min
    if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo ""
        echo "[run_teacher] vLLM is ready after ${i}×10s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "[run_teacher] ERROR: vLLM died during startup; see $VLLM_LOG"
        tail -n 50 "$VLLM_LOG"
        exit 1
    fi
    echo -n "."
    sleep 10
done
if ! curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    echo ""
    echo "[run_teacher] ERROR: vLLM did not become healthy within 30 minutes"
    tail -n 50 "$VLLM_LOG"
    exit 1
fi

# ---------- run the generator ----------
# --resume is on by default; safe to re-launch in a new salloc.
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
MANIFEST_ARG=""
[ -n "${MANIFEST_FILE:-}" ] && MANIFEST_ARG="--manifest-file $MANIFEST_FILE"
GUIDANCE_ARG=""
[ -n "${GUIDANCE_FILE:-}" ] && GUIDANCE_ARG="--guidance-file $GUIDANCE_FILE"
PROMPT_SUFFIX_ARG=""
[ -n "${PROMPT_VERSION_SUFFIX:-}" ] && PROMPT_SUFFIX_ARG="--prompt-version-suffix $PROMPT_VERSION_SUFFIX"

# SLURM sends SIGTERM at timeout; propagate to the python child so our
# signal handler gets a chance to finish the current pair cleanly.
python -m src.teacher.generate \
    --split "$SPLIT" \
    --provider openai \
    --model "$MODEL_PATH" \
    --base-url "http://127.0.0.1:$VLLM_PORT/v1" \
    --n "$N_CAND" \
    --seed "$SEED" \
    --tensor-parallel-size 4 \
    $MANIFEST_ARG \
    $GUIDANCE_ARG \
    $PROMPT_SUFFIX_ARG \
    $LIMIT_ARG

echo "[run_teacher] generation done for $TEACHER_ID"
