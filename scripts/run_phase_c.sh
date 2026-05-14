#!/usr/bin/env bash
# =============================================================================
#  Phase C — Student training (Qwen-2.5-7B + LoRA)
#
#  Stage C0 — prepare tier-weighted SFT and preference corpora.
#  Stage C1 — SFT with tier-weighted CE + faithfulness loss + position-restricted
#             symmetry-KL on the direction-tag token.
#  Stage C2 — PRM-weighted DPO / IPO with four programmatic hard-negative
#             families.  Two backends, dispatched at runtime: exact per-loss
#             hook (preferred) or deterministic importance-sampling fallback.
#  Stage C3 — (optional) classifier head over the frozen reasoner.
#
#  Reads:   outputs/teacher/teacher_clean.reasoning_safe.jsonl
#           outputs/teacher/*.preference.jsonl
#  Writes:  outputs/student/{sft,dpo,head}/<run-name>/
#
#  Usage:
#      bash scripts/run_phase_c.sh
#
#  Optional env vars:
#      ACCELERATE_CONFIG   path to FSDP / DDP launch config
#                          (default: configs/accelerate_fsdp_qwen.yaml)
#      STAGES              subset of {c0 c1 c2 c3}  (default: all)
#      OUT_DIR             override the output root (default: outputs/student)
# =============================================================================
set -euo pipefail

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
have()   { [[ " ${STAGES} " == *" $1 "* ]]; }

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_fsdp_qwen.yaml}"
STAGES="${STAGES:-c0 c1 c2 c3}"
OUT_DIR="${OUT_DIR:-outputs/student}"

mkdir -p "$OUT_DIR/sft" "$OUT_DIR/dpo" "$OUT_DIR/head"

if have c0; then
    banner "C0 · Prepare tier-weighted SFT + mirror preference corpora"
    python -m src.data.prepare_phase_c
    python -m src.data.build_mirror_sft_corpus
fi

if have c1; then
    banner "C1 · SFT  (tier-weighted CE + faithfulness + symmetry-KL)"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
        -m src.training.sft_train
fi

if have c2; then
    banner "C2 · PRM-weighted DPO / IPO  (+ four hard-negative families)"
    accelerate launch --config_file "$ACCELERATE_CONFIG" \
        -m src.training.dpo_mirror
fi

if have c3; then
    banner "C3 · (optional) classifier head over the frozen reasoner"
    python -m src.training.train_classifier_head
fi

banner "Phase C complete."
