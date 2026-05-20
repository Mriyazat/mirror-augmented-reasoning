#!/bin/bash
# -----------------------------------------------------------------------------
# Student SFT launcher (LoRA on Qwen2.5-7B-Instruct by default).
#
# Activates the environment via scripts/cluster_examples/activate_env.sh and
# launches src.training.sft_train through `accelerate launch`.
#
# Usage (inside a GPU allocation):
#   source scripts/cluster_examples/activate_env.sh
#   bash scripts/cluster_examples/run_phase_c_sft.sh
#
# All knobs are environment variables (see defaults below). The mirror-
# augmented corpus is recommended when training with the symmetry-KL term
# enabled — build it first with:
#
#   python -m src.data.build_mirror_sft_corpus \
#       --input  outputs/teacher_clean/reasoning_safe.train.jsonl \
#       --output outputs/sft_corpus/reasoning_safe.train.mirror.jsonl
#
# Mirror-augmented training expects an even per-device batch size so that
# each (AB, BA) pair fits in a single micro-batch on every device.
# -----------------------------------------------------------------------------
set -euo pipefail

MIRROR_AUG="${MIRROR_AUG:-1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
CORPUS_DIR="${CORPUS_DIR:-outputs/sft_corpus}"
OUT_BASE="${OUT_BASE:-${DDI_CKPT:-$PWD/ddi_checkpoints}/student}"

# Required input files; override via environment.
if [ "$MIRROR_AUG" = "1" ] || [ "$MIRROR_AUG" = "true" ]; then
    TRAIN_FILE="${TRAIN_FILE:-$CORPUS_DIR/reasoning_safe.train.mirror.jsonl}"
    OUT_DIR="${OUT_DIR:-$OUT_BASE/sft_mirror}"
    MIRROR_PAIR_SAMPLER_FLAG="--mirror_pair_sampler"
    BATCH_PER_GPU="${BATCH_PER_GPU:-2}"
    GRAD_ACCUM="${GRAD_ACCUM:-16}"
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
    if [ $((BATCH_PER_GPU % 2)) -ne 0 ]; then
        echo "[sft] ERROR: BATCH_PER_GPU must be even when MIRROR_AUG=1 (got $BATCH_PER_GPU)." >&2
        exit 2
    fi
else
    TRAIN_FILE="${TRAIN_FILE:-$CORPUS_DIR/reasoning_safe.train.jsonl}"
    OUT_DIR="${OUT_DIR:-$OUT_BASE/sft}"
    MIRROR_PAIR_SAMPLER_FLAG=""
    BATCH_PER_GPU="${BATCH_PER_GPU:-1}"
    GRAD_ACCUM="${GRAD_ACCUM:-32}"
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
fi
VAL_FILE="${VAL_FILE:-$CORPUS_DIR/reasoning_safe.val.jsonl}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
FAITHFULNESS_WEIGHT="${FAITHFULNESS_WEIGHT:-0.5}"
SYMMETRY_WEIGHT="${SYMMETRY_WEIGHT:-0.3}"
CLASS_BALANCE_MODE="${CLASS_BALANCE_MODE:-off}"
CLASS_BALANCE_MAX="${CLASS_BALANCE_MAX:-4.0}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-none}"
LOAD_BEST="${LOAD_BEST:-0}"
METRIC_FOR_BEST="${METRIC_FOR_BEST:-eval_loss}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
RESUME="${RESUME:-auto}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_qwen.yaml}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# --load_best_at_end requires eval_steps == save_steps in HF Trainer.
if [ "$LOAD_BEST" = "1" ] || [ "$LOAD_BEST" = "true" ]; then
    SAVE_STEPS="$EVAL_STEPS"
fi

mkdir -p "$OUT_DIR"

echo "[sft] mirror_aug=$MIRROR_AUG"
echo "[sft] train=$TRAIN_FILE"
echo "[sft] val  =$VAL_FILE"
echo "[sft] model=$MODEL_NAME"
echo "[sft] out  =$OUT_DIR"
echo "[sft] batch_per_gpu=$BATCH_PER_GPU  grad_accum=$GRAD_ACCUM  pair_sampler=${MIRROR_PAIR_SAMPLER_FLAG:-off}"
echo "[sft] lr=$LR  epochs=$EPOCHS  lora_r=$LORA_R  lora_alpha=$LORA_ALPHA"
echo "[sft] faith_w=$FAITHFULNESS_WEIGHT  sym_w=$SYMMETRY_WEIGHT"
echo "[sft] eval_steps=$EVAL_STEPS  save_steps=$SAVE_STEPS  load_best=$LOAD_BEST"
echo "[sft] max_length=$MAX_LENGTH  max_steps=$MAX_STEPS  gradient_checkpointing=$GRADIENT_CHECKPOINTING"

python - <<PY
from pathlib import Path
for p in ["$TRAIN_FILE", "$VAL_FILE"]:
    if not Path(p).exists():
        raise SystemExit(f"missing required file: {p}")
PY

LOAD_BEST_FLAG="--save_total_limit $SAVE_TOTAL_LIMIT"
if [ "$LOAD_BEST" = "1" ] || [ "$LOAD_BEST" = "true" ]; then
    LOAD_BEST_FLAG="--load_best_at_end --metric_for_best_model $METRIC_FOR_BEST --save_total_limit $SAVE_TOTAL_LIMIT"
fi

GRADIENT_CHECKPOINTING_FLAG=""
if [ "$GRADIENT_CHECKPOINTING" = "1" ] || [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" \
    -m src.training.sft_train \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUT_DIR" \
    --model_name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch_per_gpu "$BATCH_PER_GPU" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --max_steps "$MAX_STEPS" \
    --max_length "$MAX_LENGTH" \
    --torch_dtype "$TORCH_DTYPE" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --faithfulness_weight "$FAITHFULNESS_WEIGHT" \
    --symmetry_weight "$SYMMETRY_WEIGHT" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --class_balance_mode "$CLASS_BALANCE_MODE" \
    --class_balance_max "$CLASS_BALANCE_MAX" \
    --seed "$SEED" \
    --report_to "$REPORT_TO" \
    --resume_from_checkpoint "$RESUME" \
    $LOAD_BEST_FLAG \
    $GRADIENT_CHECKPOINTING_FLAG \
    $MIRROR_PAIR_SAMPLER_FLAG
