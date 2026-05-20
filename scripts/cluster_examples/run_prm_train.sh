#!/bin/bash
# -----------------------------------------------------------------------------
# DDI-PRM (Process Reward Model) fine-tune launcher for a 4 x H100 node.
#
# Pre-reqs (run once on a login node with internet access):
#   bash scripts/cluster_examples/setup_env.sh
#   python scripts/cluster_examples/download_models.py --only Llama-3.1-8B-Instruct
#
# Then request a GPU node and run:
#   salloc --time=8:00:00 --cpus-per-task=32 --mem=256G \
#          --gres=gpu:h100:4 --account=${SLURM_ACCOUNT:-your_account}
#   cd /path/to/EMNLP2026_DDI_Verifier_Release
#   source scripts/cluster_examples/activate_env.sh
#   bash scripts/cluster_examples/run_prm_train.sh
# -----------------------------------------------------------------------------
set -euo pipefail

TRAIN_FILE="${TRAIN_FILE:-outputs/teacher/prm_train.jsonl}"
EVAL_FILE="${EVAL_FILE:-outputs/teacher/prm_eval.jsonl}"
OUT_DIR="${OUT_DIR:-${DDI_CKPT:-$PWD/ddi_checkpoints}/ddi_prm_v1}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

mkdir -p "$OUT_DIR"
echo "[run_prm_train] train=$TRAIN_FILE"
echo "[run_prm_train] eval =$EVAL_FILE"
echo "[run_prm_train] base =$BASE_MODEL"
echo "[run_prm_train] out  =$OUT_DIR"

accelerate launch --config_file configs/accelerate_fsdp_prm.yaml \
    -m src.teacher.prm_train \
    --base_model "$BASE_MODEL" \
    --train_file "$TRAIN_FILE" \
    --eval_file  "$EVAL_FILE" \
    --output_dir "$OUT_DIR" \
    --epochs 2 --lr 1e-4 --batch_size 4 --grad_accum 4 \
    --lora_r 16 --lora_alpha 32 --max_len 3072 \
    --resume
