#!/usr/bin/env bash
# Trace-Alignment SFT pipeline (cloud, 1xH100, ~3-4 hours total).
#
# What it does
# ------------
# 1. Builds a 5000-pair stratified TRAIN-side manifest (split_section=val,
#    NOT test) so we never train on test examples.
# 2. Builds the with_neighbors prompts file for that subset.
# 3. Generates greedy + rerank=4 student predictions on those 5k pairs.
# 4. Builds a trace-alignment SFT JSONL from those predictions:
#       - rescues: trace majority == gold, final != gold
#       - demos:   trace == gold == final  (sampled, balanced)
# 5. Runs a SHORT LoRA SFT pass (2 epochs, low LR) starting from the
#    current Phase-4 student adapter.
# 6. Re-runs evaluation on all three test splits and prints macro-F1
#    deltas vs the pre-SFT model.
#
# Time budget on 1xH100
# ---------------------
#   pred generation (5k, rerank4)   ~35-45 min
#   SFT (2 epochs, 4k records)      ~25-40 min
#   re-eval on 3 splits             ~45-60 min
#   total                           ~2-3 hours
#
# Expected gain
# -------------
#   +1-2 macro-F1 on every split (worst case: 0; best case: +3 on cold).
#   Validate by inspecting outputs/student/trace_align/eval_after/*.json
#   vs the pre-SFT baselines in outputs/eval_prompts/.
#
# Pre-reqs (assumed already exported in your slurm shell):
#   $ADAPTER     -> current best student adapter
#   $PRM_BASE    -> reward base
#   $PRM_ADAPTER -> reward adapter
#   $THRESHOLDS  -> conformal thresholds JSON
#
set -euo pipefail

# ---- Required env vars (export these in your slurm shell before invoking) ----
: "${ADAPTER:=${DDI_CKPT:-/scratch/$USER/ddi_checkpoints}/student/ddi_v4_best_phase4_prm_dpo_macro0797}"
: "${PRM_BASE:=dmis-lab/llama-3.1-medprm-reward-v1.0}"
: "${PRM_ADAPTER:=${DDI_CKPT:-/scratch/$USER/ddi_checkpoints}/ddi_prm_v1}"
: "${THRESHOLDS:=outputs/student/ddi_v4_phase4/conformal_thresholds.json}"

echo "[trace-align] ADAPTER     = $ADAPTER"
echo "[trace-align] PRM_BASE    = $PRM_BASE"
echo "[trace-align] PRM_ADAPTER = $PRM_ADAPTER"
echo "[trace-align] THRESHOLDS  = $THRESHOLDS"

for p in "$ADAPTER" "$PRM_ADAPTER" "$THRESHOLDS"; do
  if [ ! -e "$p" ]; then
    echo "[trace-align] FATAL: required path not found: $p" >&2
    echo "  export the correct ADAPTER / PRM_ADAPTER / THRESHOLDS and re-run." >&2
    exit 2
  fi
done

OUT_ROOT="outputs/student/trace_align"
RESCUE_DIR="${OUT_ROOT}/rescue_data"
SFT_OUT="${OUT_ROOT}/adapter_v1"
EVAL_OUT="${OUT_ROOT}/eval_after"
mkdir -p "$RESCUE_DIR" "$EVAL_OUT" logs

# Defaults; can be overridden by env vars before invoking the script.
#   RESCUE_MODE=greedy   -> 4x faster, fits in one 8h salloc
#   RESCUE_MODE=rerank4  -> richer signal, needs ~24 GPU-hours
RESCUE_MODE="${RESCUE_MODE:-greedy}"
N_PROMPTS="${N_PROMPTS:-5000}"

# -----------------------------------------------------------------------------
# 1. Build a 5k VAL-side manifest (held-out from test).  We use the standard
#    random_full validation slice; this never overlaps test.
# -----------------------------------------------------------------------------
TRAIN_MANIFEST="outputs/eval_prompts/random_full_val_5000_stratified.manifest.jsonl"
TRAIN_PROMPTS="outputs/eval_prompts/random_full_val_5000_stratified.with_neighbors.prompts.jsonl"
NEIGHBOR_INDEX="data_processed/neighbor_index_random_full_val5k.parquet"

if [ ! -s "$TRAIN_MANIFEST" ]; then
  python -m src.data.build_stratified_manifest \
    --split random_full --section val --n 5000 \
    --output "$TRAIN_MANIFEST"
fi

# CRITICAL: rebuild a neighbor index over the 5k val pair_ids so the prompts
# have non-empty `SIMILAR LABELED PAIRS` blocks (matches training distribution).
if [ ! -s "$NEIGHBOR_INDEX" ]; then
  python -m src.teacher.context_builder \
    --precompute_neighbors random_full \
    --query_manifest_file "$TRAIN_MANIFEST" \
    --output_path "$NEIGHBOR_INDEX" \
    --top_k 6
fi

if [ ! -s "$TRAIN_PROMPTS" ]; then
  python -m src.data.build_student_eval_prompts \
    --split random_full --split_section val \
    --manifest_jsonl "$TRAIN_MANIFEST" \
    --neighbor_index_path "$NEIGHBOR_INDEX" \
    --output "$TRAIN_PROMPTS"
fi

# Sanity-check: every prompt must have a neighbor block AND most must be populated.
HAS_HDR=$(grep -c "mechanistic-neighbor" "$TRAIN_PROMPTS" || true)
EMPTY_NB=$(grep -c "no mechanistic-neighbor pairs surfaced" "$TRAIN_PROMPTS" || true)
POPULATED=$((HAS_HDR - EMPTY_NB))
echo "[trace-align] prompts: total_w_header=$HAS_HDR  empty=$EMPTY_NB  populated=$POPULATED"
if [ "$HAS_HDR" -lt 4500 ] || [ "$POPULATED" -lt 4500 ]; then
  echo "[trace-align] FATAL: too few populated neighbor blocks. Refusing to continue."
  exit 1
fi
echo "[trace-align] OK: $POPULATED / 5000 prompts have populated neighbor blocks."

# -----------------------------------------------------------------------------
# 2. Generate student predictions with rerank=4 (so we get diverse traces
#    we can mine for "right-trace, wrong-answer" rescues).
# -----------------------------------------------------------------------------
# Skip-condition: only consider the prediction file "done" when it has
# at least N unique pair_ids matching the expected count. Anything less
# means we resume.
if [ "$RESCUE_MODE" = "rerank4" ]; then
  RESCUE_PREDS="${RESCUE_DIR}/preds_val5k_rerank4.jsonl"
  PRED_ARGS=(--n_samples 4 --temperature 0.7 --top_p 0.95 --batch 4)
else
  RESCUE_PREDS="${RESCUE_DIR}/preds_val5k_greedy.jsonl"
  PRED_ARGS=(--n_samples 1 --temperature 0.0 --batch 8)
fi

UNIQUE_DONE=0
if [ -s "$RESCUE_PREDS" ]; then
  UNIQUE_DONE=$(python -c "
import json, sys
seen=set()
for line in open(sys.argv[1]):
    try:
        seen.add(json.loads(line)['pair_id'])
    except Exception: pass
print(len(seen))
" "$RESCUE_PREDS")
fi
echo "[trace-align] rescue preds: ${RESCUE_PREDS}  unique pair_ids done = $UNIQUE_DONE / $N_PROMPTS  mode=$RESCUE_MODE"

if [ "$UNIQUE_DONE" -lt "$N_PROMPTS" ]; then
  python -u -m src.inference.predict_with_rerank \
    --adapter "$ADAPTER" \
    --prm_base "$PRM_BASE" --prm_adapter "$PRM_ADAPTER" --prm_device cuda:0 \
    --input  "$TRAIN_PROMPTS" \
    --output "$RESCUE_PREDS" \
    "${PRED_ARGS[@]}" --max_new_tokens 768 \
    --torch_dtype bfloat16 --device_map cuda:0 \
    --resume 2>&1 | tee -a logs/trace_align_predict_${RESCUE_MODE}.log
fi

# -----------------------------------------------------------------------------
# 3. Build trace-alignment SFT data
# -----------------------------------------------------------------------------
SFT_DATA="${RESCUE_DIR}/trace_align_train_${RESCUE_MODE}.jsonl"
SFT_VAL="${RESCUE_DIR}/trace_align_val_${RESCUE_MODE}.jsonl"
python -m src.training.build_trace_alignment_sft \
  --prompt_jsonl "$TRAIN_PROMPTS" \
  --pred_jsonl   "$RESCUE_PREDS" \
  --output       "$SFT_DATA" \
  --mode trace_correct \
  --include_correct_demos 1400 \
  --rescue_weight 1.5 \
  --demo_weight 0.5

# Tiny held-out val: take first 200 records for SftJsonlDataset eval
head -n 200 "$SFT_DATA" > "$SFT_VAL"

# -----------------------------------------------------------------------------
# 4. Short LoRA SFT pass on the rescue set
# -----------------------------------------------------------------------------
python -u -m src.training.sft_train \
  --train_file "$SFT_DATA" \
  --val_file   "$SFT_VAL" \
  --output_dir "$SFT_OUT" \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_init "$ADAPTER" \
  --epochs 2 \
  --lr 5.0e-6 \
  --warmup_ratio 0.05 \
  --batch_per_gpu 1 \
  --grad_accum 16 \
  --gradient_checkpointing \
  --max_length 3072 \
  --faithfulness_weight 0.3 \
  --symmetry_weight 0.0 \
  --disable_symmetry_loss \
  --class_balance_mode sqrt_inverse \
  --save_steps 200 --eval_steps 200 --logging_steps 25 \
  --load_best_at_end \
  --metric_for_best_model eval_assistant_token_acc \
  --torch_dtype bfloat16 \
  2>&1 | tee logs/trace_align_sft.log

# --load_best_at_end leaves the in-memory best as the saved final;
# fall back to the most recent checkpoint dir.
NEW_ADAPTER=$(ls -td "$SFT_OUT"/checkpoint-* 2>/dev/null | head -1)
[ -z "$NEW_ADAPTER" ] && NEW_ADAPTER="$SFT_OUT"
echo "[trace-align] new adapter = $NEW_ADAPTER"

# -----------------------------------------------------------------------------
# 5. Re-evaluate on all three test splits (greedy is enough for a quick read;
#    rerun rerank4 separately if you want the full headline numbers)
# -----------------------------------------------------------------------------
for SPLIT in random_full drug_cold pair_cold; do
  PROMPTS="outputs/eval_prompts/${SPLIT}_test_5000_stratified.with_neighbors.prompts.jsonl"
  OUT="${EVAL_OUT}/pred_traceAlign_${SPLIT}_greedy.jsonl"
  python -u -m src.inference.predict \
    --adapter "$NEW_ADAPTER" \
    --input "$PROMPTS" \
    --output "$OUT" \
    --temperature 0.0 --max_new_tokens 768 \
    --batch 24 --torch_dtype bfloat16 --device_map cuda:0 \
    --resume 2>&1 | tee -a logs/trace_align_eval.log
done

# -----------------------------------------------------------------------------
# 6. Print macro-F1 deltas vs the pre-SFT baseline
# -----------------------------------------------------------------------------
python -m src.evaluation.cpu_ablation 2>&1 | tee -a logs/trace_align_eval.log
echo "[trace-align] DONE.  Adapter: $NEW_ADAPTER"
