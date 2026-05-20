#!/bin/bash
# -----------------------------------------------------------------------------
# Post-SFT pipeline: predict -> augment -> (optional abstention fit +
# apply) -> run_full_eval.
#
# Designed to run on a 4-H100 allocation (same allocation you used for SFT).
# AB and BA predict passes run in parallel on 2 GPUs each. That cuts total
# predict wall time from ~5.5 h (1 GPU) to ~1.5 h.
#
# Env knobs (all optional):
#   ADAPTER               path to the SFT adapter directory
#                         (default: $DDI_CKPT/student/ddi_v4_sft_reasoning_safe)
#   BASE_MODEL            base HF model (default: Qwen/Qwen2.5-7B-Instruct)
#   INPUT_JSONL           teacher-format val/test jsonl
#                         (default: outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl)
#   OUT_DIR               directory for predictions + eval artifacts
#                         (default: outputs/phase_c)
#   RUN_NAME              label for the run_full_eval output folder
#                         (default: ddi_v4_sft_reasoning_safe)
#   LABELS                data_processed/labels_hierarchical.parquet
#   MAX_NEW_TOKENS        generation cap (default: 1024)
#   BATCH                 per-pass generation batch size (default: 12)
#   FIT_ABSTENTION        1 to fit + apply conformal abstention (default: 1)
#   TARGET_COVERAGE       target marginal coverage (default: 0.90)
#   TARGET_SELECTIVE_ACC  target selective accuracy (default: 0.92)
#   SKIP_PREDICT          1 to reuse existing predictions_val.jsonl
#   SKIP_EVAL             1 to stop before run_full_eval
#
# Usage:
#   bash scripts/cluster_examples/run_phase_c_post_sft.sh
#
#   # Reuse predictions already produced (e.g. re-running eval with
#   # different abstention thresholds):
#   SKIP_PREDICT=1 bash scripts/cluster_examples/run_phase_c_post_sft.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ADAPTER="${ADAPTER:-${DDI_CKPT:-$PWD/ddi_checkpoints}/student/ddi_v4_sft_reasoning_safe}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
INPUT_JSONL="${INPUT_JSONL:-outputs/phase_c/teacher_clean.reasoning_safe.val.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/phase_c}"
RUN_NAME="${RUN_NAME:-ddi_v4_sft_reasoning_safe}"
LABELS="${LABELS:-data_processed/labels_hierarchical.parquet}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
BATCH="${BATCH:-12}"
FIT_ABSTENTION="${FIT_ABSTENTION:-1}"
TARGET_COVERAGE="${TARGET_COVERAGE:-0.90}"
TARGET_SELECTIVE_ACC="${TARGET_SELECTIVE_ACC:-0.92}"
SKIP_PREDICT="${SKIP_PREDICT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
RESUME_PREDICT="${RESUME_PREDICT:-0}"

AB_JSONL="$OUT_DIR/predictions_val.ab.jsonl"
BA_JSONL="$OUT_DIR/predictions_val.ba.jsonl"
COMBINED="$OUT_DIR/predictions_val.jsonl"
WITH_GOLD="$OUT_DIR/predictions_val.with_gold.jsonl"
THRESHOLDS="$OUT_DIR/abstention_thresholds.json"
FINAL="$OUT_DIR/predictions_val.final.jsonl"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo "[post_sft] adapter=$ADAPTER"
echo "[post_sft] input=$INPUT_JSONL"
echo "[post_sft] out=$OUT_DIR"
echo "[post_sft] run_name=$RUN_NAME"
echo "============================================================"

# --- Step 1: predict AB + BA (parallel across 2+2 GPUs) ----------------------
if [ "$SKIP_PREDICT" != "1" ]; then
    echo ""
    echo "[post_sft] Step 1: parallel predict (AB on GPU 0-1, BA on GPU 2-3)"

    RESUME_FLAG=""
    if [ "$RESUME_PREDICT" = "1" ] || [ "$RESUME_PREDICT" = "true" ]; then
        RESUME_FLAG="--resume"
        echo "[post_sft]   RESUME_PREDICT=1: pair_ids already written in "
        echo "               $AB_JSONL and $BA_JSONL will be skipped and appended to."
    fi

    # Visible-GPU assignment; device_map=auto shards the 7B model over the 2 GPUs.
    CUDA_VISIBLE_DEVICES=0,1 python -m src.inference.predict \
        --adapter   "$ADAPTER" \
        --base_model "$BASE_MODEL" \
        --input     "$INPUT_JSONL" \
        --output    "$AB_JSONL" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch "$BATCH" \
        $RESUME_FLAG \
        --device_map auto > "$OUT_DIR/predict_ab.log" 2>&1 &
    AB_PID=$!

    CUDA_VISIBLE_DEVICES=2,3 python -m src.inference.predict \
        --adapter   "$ADAPTER" \
        --base_model "$BASE_MODEL" \
        --input     "$INPUT_JSONL" \
        --output    "$BA_JSONL" \
        --mirror \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --batch "$BATCH" \
        $RESUME_FLAG \
        --device_map auto > "$OUT_DIR/predict_ba.log" 2>&1 &
    BA_PID=$!

    echo "[post_sft]   AB pid=$AB_PID  BA pid=$BA_PID  (see predict_{ab,ba}.log)"
    wait $AB_PID
    wait $BA_PID
    echo "[post_sft]   both predict passes complete."

    cat "$AB_JSONL" "$BA_JSONL" > "$COMBINED"
    n_ab=$(wc -l < "$AB_JSONL")
    n_ba=$(wc -l < "$BA_JSONL")
    n_all=$(wc -l < "$COMBINED")
    echo "[post_sft]   AB=$n_ab  BA=$n_ba  combined=$n_all -> $COMBINED"

    # Parse-rate sanity
    python - <<PY
import json
p = "$COMBINED"
n = 0
ok = 0
for line in open(p):
    n += 1
    ok += 1 if json.loads(line)["parse_ok"] else 0
print(f"[post_sft]   parse_ok={ok}/{n} = {ok/n:.3f}")
PY
else
    echo "[post_sft] SKIP_PREDICT=1: reusing $COMBINED"
fi

# --- Step 2: augment with gold labels (for abstention fit) --------------------
echo ""
echo "[post_sft] Step 2: augment with gold labels"
python -m src.inference.augment_predictions \
    --predictions "$COMBINED" \
    --labels      "$LABELS" \
    --output      "$WITH_GOLD"

# --- Step 3 (optional): abstention fit + apply --------------------------------
if [ "$FIT_ABSTENTION" = "1" ]; then
    echo ""
    echo "[post_sft] Step 3: fit abstention thresholds on val"
    python -m src.inference.abstention fit \
        --val_predictions "$WITH_GOLD" \
        --out_thresholds  "$THRESHOLDS" \
        --target_coverage "$TARGET_COVERAGE" \
        --target_selective_acc "$TARGET_SELECTIVE_ACC"

    echo ""
    echo "[post_sft] Step 3b: apply thresholds to val predictions"
    python -m src.inference.abstention apply \
        --thresholds      "$THRESHOLDS" \
        --in_predictions  "$WITH_GOLD" \
        --out_predictions "$FINAL"
else
    echo "[post_sft] Step 3: FIT_ABSTENTION=0, skipping abstention fit"
    cp "$WITH_GOLD" "$FINAL"
fi

# --- Step 4: full evaluation --------------------------------------------------
if [ "$SKIP_EVAL" != "1" ]; then
    echo ""
    echo "[post_sft] Step 4: run_full_eval"
    python -m src.evaluation.run_full_eval \
        --predictions "$FINAL" \
        --labels      "$LABELS" \
        --run_name    "$RUN_NAME"
    echo ""
    echo "[post_sft] done. Results in outputs/results/$RUN_NAME/"
    echo "[post_sft]   headline metrics in outputs/results/$RUN_NAME/metrics.md"
else
    echo "[post_sft] SKIP_EVAL=1: stopping before run_full_eval"
fi
