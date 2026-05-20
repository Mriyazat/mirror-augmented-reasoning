#!/bin/bash
# -----------------------------------------------------------------------------
# Student SFT hyperparameter sweep.
#
# Trains a short (typically 1-epoch) SFT run for each combination of
# learning-rate and LoRA-rank in {LRS} x {LORA_RS}, writing each
# configuration to its own subfolder under OUT_BASE/sweeps. The final
# step calls src.training.summarize_sweep to produce a comparison table.
#
# Usage:
#   bash scripts/cluster_examples/run_phase_c_sft_sweep.sh
#
# Override candidate lists:
#   LRS="1e-4 2e-4 3e-4" LORA_RS="32 64" \
#       bash scripts/cluster_examples/run_phase_c_sft_sweep.sh
# -----------------------------------------------------------------------------
set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-outputs/sft_corpus}"
OUT_BASE="${OUT_BASE:-${DDI_CKPT:-$PWD/ddi_checkpoints}/student}"
SWEEP_DIR="${SWEEP_DIR:-$OUT_BASE/sweeps}"
LRS="${LRS:-1e-4 2e-4 3e-4 5e-4}"
LORA_RS="${LORA_RS:-64}"
EPOCHS="${EPOCHS:-1}"
BATCH_PER_GPU="${BATCH_PER_GPU:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SAVE_STEPS="${SAVE_STEPS:-40}"
EVAL_STEPS="${EVAL_STEPS:-40}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
FAITHFULNESS_WEIGHT="${FAITHFULNESS_WEIGHT:-0.5}"
SYMMETRY_WEIGHT="${SYMMETRY_WEIGHT:-0.3}"
SEED="${SEED:-42}"
REPORT_TO="${REPORT_TO:-none}"
LOAD_BEST="${LOAD_BEST:-1}"
METRIC_FOR_BEST="${METRIC_FOR_BEST:-eval_loss}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_qwen.yaml}"
# RESUME=no  -> do NOT resume from any prior checkpoint (recommended for sweeps).
# RESUME=auto -> resume from the latest checkpoint if one is present.
RESUME="${RESUME:-no}"

TRAIN_FILE="${TRAIN_FILE:-$CORPUS_DIR/reasoning_safe.train.jsonl}"
VAL_FILE="${VAL_FILE:-$CORPUS_DIR/reasoning_safe.val.jsonl}"

# --load_best_at_end requires eval_steps == save_steps; keep them locked.
SAVE_STEPS="$EVAL_STEPS"

mkdir -p "$SWEEP_DIR"

echo "[sft_sweep] train=$TRAIN_FILE"
echo "[sft_sweep] val  =$VAL_FILE"
echo "[sft_sweep] LRs  =$LRS"
echo "[sft_sweep] LoRA r=$LORA_RS"
echo "[sft_sweep] epochs=$EPOCHS"
echo "[sft_sweep] eval_steps=$EVAL_STEPS  save_steps=$SAVE_STEPS  logging_steps=$LOGGING_STEPS"
echo "[sft_sweep] load_best=$LOAD_BEST  metric=$METRIC_FOR_BEST"
echo "[sft_sweep] out  =$SWEEP_DIR"

python - <<PY
from pathlib import Path
for p in ["$TRAIN_FILE", "$VAL_FILE"]:
    if not Path(p).exists():
        raise SystemExit(f"missing required file: {p}")
PY

LOAD_BEST_FLAG=""
if [ "$LOAD_BEST" = "1" ] || [ "$LOAD_BEST" = "true" ]; then
    LOAD_BEST_FLAG="--load_best_at_end --metric_for_best_model $METRIC_FOR_BEST"
fi

for LR in $LRS; do
    for R in $LORA_RS; do
        TAG="lr${LR}_r${R}"
        OUT_DIR="$SWEEP_DIR/$TAG"
        mkdir -p "$OUT_DIR"
        ALPHA=$((R * 2))
        echo ""
        echo "============================================================"
        echo "[sft_sweep] training $TAG -> $OUT_DIR"
        echo "============================================================"
        accelerate launch --config_file "$ACCELERATE_CONFIG" \
            -m src.training.sft_train \
            --train_file "$TRAIN_FILE" \
            --val_file "$VAL_FILE" \
            --output_dir "$OUT_DIR" \
            --model_name "${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" \
            --epochs "$EPOCHS" \
            --batch_per_gpu "$BATCH_PER_GPU" \
            --grad_accum "$GRAD_ACCUM" \
            --lr "$LR" \
            --max_length "$MAX_LENGTH" \
            --torch_dtype "$TORCH_DTYPE" \
            --save_steps "$SAVE_STEPS" \
            --eval_steps "$EVAL_STEPS" \
            --logging_steps "$LOGGING_STEPS" \
            --faithfulness_weight "$FAITHFULNESS_WEIGHT" \
            --symmetry_weight "$SYMMETRY_WEIGHT" \
            --lora_r "$R" \
            --lora_alpha "$ALPHA" \
            --save_total_limit "$SAVE_TOTAL_LIMIT" \
            --seed "$SEED" \
            --report_to "$REPORT_TO" \
            --resume_from_checkpoint "$RESUME" \
            $LOAD_BEST_FLAG
    done
done

echo ""
echo "[sft_sweep] all configs complete. Summarizing..."
python -m src.training.summarize_sweep --sweep_dir "$SWEEP_DIR" \
    --out "$SWEEP_DIR/sweep_summary.md" \
    --out_json "$SWEEP_DIR/sweep_summary.json" || true
echo "[sft_sweep] see $SWEEP_DIR/sweep_summary.md"
