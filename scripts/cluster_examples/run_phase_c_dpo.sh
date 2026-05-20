#!/bin/bash
# -----------------------------------------------------------------------------
# Mirror-IPO / DPO launcher (preference optimization on top of the SFT
# adapter, using PRM-weighted hard-negative and mirror-pair preferences).
#
# Run only after SFT has produced an adapter to start from.
#
# Example:
#   SFT_ADAPTER=$DDI_CKPT/student/sft_mirror \
#   bash scripts/cluster_examples/run_phase_c_dpo.sh
# -----------------------------------------------------------------------------
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PREF_DIR="${PREF_DIR:-outputs/preferences}"
OUT_BASE="${OUT_BASE:-${DDI_CKPT:-$PWD/ddi_checkpoints}/student}"
SFT_ADAPTER="${SFT_ADAPTER:-$OUT_BASE/sft_mirror}"

PREF_FILE="${PREF_FILE:-$PREF_DIR/mirror_preferences.reasoning_safe.train.jsonl}"
VAL_PREF_FILE="${VAL_PREF_FILE:-$PREF_DIR/mirror_preferences.reasoning_safe.val.jsonl}"
OUT_DIR="${OUT_DIR:-$OUT_BASE/dpo_mirror}"

EPOCHS="${EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"

BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-5e-6}"
MAX_STEPS="${MAX_STEPS:--1}"
LOSS_TYPE="${LOSS_TYPE:-ipo}"
BETA="${BETA:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3072}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
RESUME="${RESUME:-auto}"
SEED="${SEED:-42}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_qwen.yaml}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
USE_PRM_WEIGHT="${USE_PRM_WEIGHT:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUT_DIR"

echo "[dpo] pref=$PREF_FILE"
echo "[dpo] val =$VAL_PREF_FILE"
echo "[dpo] sft =$SFT_ADAPTER"
echo "[dpo] out =$OUT_DIR"
echo "[dpo] lr=$LR max_steps=$MAX_STEPS loss=$LOSS_TYPE beta=$BETA gradient_checkpointing=$GRADIENT_CHECKPOINTING"
echo "[dpo] use_prm_weight=$USE_PRM_WEIGHT"

python - <<PY
from pathlib import Path
for p in ["$PREF_FILE", "$VAL_PREF_FILE", "$SFT_ADAPTER"]:
    if not Path(p).exists():
        raise SystemExit(f"missing required path: {p}")
PY

GRADIENT_CHECKPOINTING_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ] || [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"
fi

PRM_WEIGHT_FLAG=""
if [ "$USE_PRM_WEIGHT" = "1" ] || [ "$USE_PRM_WEIGHT" = "true" ]; then
    PRM_WEIGHT_FLAG="--use_prm_weight"
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" \
    -m src.training.dpo_mirror \
    --pref_file "$PREF_FILE" \
    --val_pref_file "$VAL_PREF_FILE" \
    --output_dir "$OUT_DIR" \
    --model_name "$MODEL_NAME" \
    --sft_adapter "$SFT_ADAPTER" \
    --loss_type "$LOSS_TYPE" \
    --beta "$BETA" \
    --epochs "$EPOCHS" \
    --batch_per_gpu "$BATCH_PER_GPU" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --seed "$SEED" \
    --resume_from_checkpoint "$RESUME" \
    $PRM_WEIGHT_FLAG \
    $GRADIENT_CHECKPOINTING_FLAG
