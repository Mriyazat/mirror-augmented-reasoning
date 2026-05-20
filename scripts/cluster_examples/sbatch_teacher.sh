#!/bin/bash
#SBATCH --account=${SLURM_ACCOUNT:-your_account}
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --signal=B:SIGTERM@120
#SBATCH --open-mode=append
#SBATCH --requeue
# ============================================================================
# Teacher generation — example SLURM submission wrapper.
#
# This is a portable example. Replace the `--account` placeholder with your
# own allocation identifier (e.g. via `sbatch -A <account> ...` or by
# exporting SLURM_ACCOUNT before submission). All paths use relative
# references from the repository root.
#
# Resource sizing notes (4 x H100 80GB node):
#   - 16 cores + ~124 GB per H100 covers vLLM tokenizer, async prefetch,
#     and the Python generator threadpool.
#   - --mem=0 requests the whole node's memory.
#
# Walltime:
#   - 25k pairs x 24 candidates at ~60-100 s/pair on 4 H100s is ~16-25 h
#     of run time. A single 72 h request finishes in one shot. Adjust to
#     your cluster's queue limits and policies.
#
# Signal handling:
#   --signal=B:SIGTERM@120 instructs SLURM to send SIGTERM to the batch
#   script 120 seconds before the time limit, allowing graceful teardown
#   of the vLLM server and a clean checkpoint in `generate.py`.
#
# Auto-requeue:
#   --requeue lets SLURM resubmit on time-limit or node failure. Combined
#   with the resume logic in `generate.py`, runs continue from the last
#   completed pair.
#
# Usage (submit one job per teacher checkpoint):
#   cd /path/to/EMNLP2026_DDI_Verifier_Release
#   sbatch -J teacher-llama -o logs/teacher-llama-%j.out \
#          scripts/cluster_examples/sbatch_teacher.sh llama-3.3-70b   subset25k
#   sbatch -J teacher-qwen  -o logs/teacher-qwen-%j.out \
#          scripts/cluster_examples/sbatch_teacher.sh qwen-2.5-72b    subset25k
#   sbatch -J teacher-ds    -o logs/teacher-ds-%j.out \
#          scripts/cluster_examples/sbatch_teacher.sh deepseek-r1-distill-70b subset25k
#
# Self-chaining: after each job completes, if the output file has not
# reached the expected pair count, a follow-up job is submitted. Combined
# with `--requeue`, this lets one launch carry through to completion.
# ============================================================================
set -euo pipefail

TEACHER_ID="${1:?teacher id required}"
SPLIT="${2:-subset25k}"
EXPECTED_PAIRS="${3:-25000}"

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
source scripts/cluster_examples/activate_env.sh

echo "========================================================================"
echo "[sbatch] job_id=$SLURM_JOB_ID  teacher=$TEACHER_ID  split=$SPLIT"
echo "[sbatch] host=$(hostname)  started=$(date -Is)"
echo "========================================================================"

# Run the inner driver in the background so we can trap SIGTERM cleanly.
bash scripts/cluster_examples/run_teacher.sh "$TEACHER_ID" "$SPLIT" &
RUN_PID=$!

# Forward SIGTERM from SLURM to the inner pipeline. The inner script
# already has a cleanup trap that shuts down vLLM, and generate.py
# has its own SIGTERM handler that finishes the current pair first.
trap 'echo "[sbatch] received SIGTERM, forwarding to run_teacher.sh"; kill -TERM "$RUN_PID" 2>/dev/null || true; wait "$RUN_PID" 2>/dev/null || true; exit 0' SIGTERM

wait "$RUN_PID"
RC=$?
echo "[sbatch] run_teacher.sh exited with code=$RC"

# Self-chain: if we haven't finished the split, submit a follow-up job.
# Completion is determined by counting unique pair_ids in the raw output.
RAW="${DDI_OUTPUTS:-outputs}/teacher/raw_${SPLIT}"
case "$TEACHER_ID" in
  llama-3.3-70b)           RAW="${RAW}_openai-meta-llama_Llama-3.3-70B-Instruct.jsonl" ;;
  qwen-2.5-72b)            RAW="${RAW}_openai-Qwen_Qwen2.5-72B-Instruct.jsonl" ;;
  deepseek-r1-distill-70b) RAW="${RAW}_openai-deepseek-ai_DeepSeek-R1-Distill-Llama-70B.jsonl" ;;
esac

if [ -f "$RAW" ]; then
    DONE_PAIRS=$(python -c "
import json
pairs=set()
for line in open('$RAW'):
    try: pairs.add(json.loads(line)['pair_id'])
    except: pass
print(len(pairs))
")
else
    DONE_PAIRS=0
fi

echo "[sbatch] completed pairs: $DONE_PAIRS / $EXPECTED_PAIRS"

if [ "$DONE_PAIRS" -lt "$EXPECTED_PAIRS" ]; then
    echo "[sbatch] chaining next job for $TEACHER_ID"
    sbatch -J "$SLURM_JOB_NAME" -o "logs/${SLURM_JOB_NAME}-%j.out" \
        scripts/cluster_examples/sbatch_teacher.sh "$TEACHER_ID" "$SPLIT" "$EXPECTED_PAIRS"
else
    echo "[sbatch] $TEACHER_ID COMPLETE — no more chaining."
fi
