#!/usr/bin/env bash
# =============================================================================
#  Phase B — Multi-teacher consensus distillation
#
#  Trains the DDI-PRM critic, then generates candidate traces from each
#  teacher, scores them with the 10 rule gates + PRM, merges across teachers,
#  filters for reasoning safety, and builds the preference corpora for DPO.
#
#  Reads:   data_processed/*, outputs/splits/*, the trained PRM checkpoint
#  Writes:  outputs/teacher/*.jsonl  +  outputs/teacher/*.preference.jsonl
#
#  Usage:
#      # Default — run every teacher on the balanced subset.
#      bash scripts/run_phase_b.sh
#
#      # Run only a specific teacher and split:
#      TEACHERS="llama-3.3-70b"  SPLIT="subset"  bash scripts/run_phase_b.sh
#
#  Optional env vars:
#      TEACHERS   space-separated subset of {llama-3.3-70b, qwen-2.5-72b,
#                                            deepseek-r1-70b}
#      SPLIT      generation split id (default: subset)
#      SKIP_PRM   1  → reuse an existing PRM checkpoint, skip B0
#      SKIP_OOF   1  → skip the optional GPT/Claude/Gemini judge probe
#
#  Provider env (read by src/teacher/provider.py):
#      OPENAI_API_BASE    e.g. http://localhost:8000/v1   (vLLM server)
#      OPENAI_API_KEY     EMPTY for a local vLLM server
# =============================================================================
set -euo pipefail

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

TEACHERS="${TEACHERS:-llama-3.3-70b qwen-2.5-72b deepseek-r1-70b}"
SPLIT="${SPLIT:-subset}"

# -----------------------------------------------------------------------------
# B0 — DDI-PRM critic
# -----------------------------------------------------------------------------
if [[ "${SKIP_PRM:-0}" == "1" ]]; then
    banner "B0 · PRM training skipped (SKIP_PRM=1)"
else
    banner "B0 · Train the DDI-PRM critic (rubric: configs/prm_rubric.yaml)"
    python -m src.teacher.prm_data
    python -m src.teacher.prm_train
    python -m src.teacher.prm_verify
fi

# -----------------------------------------------------------------------------
# B1 — Teacher candidate generation (vLLM server expected at OPENAI_API_BASE)
# -----------------------------------------------------------------------------
for teacher in $TEACHERS; do
    banner "B1 · Generate candidates for $teacher (split=$SPLIT)"
    python -m src.teacher.generate --split "$SPLIT" --teacher "$teacher"
done

# -----------------------------------------------------------------------------
# B2–B4 — Rule QC → PRM critic → consensus merge → reasoning-safety filter
# -----------------------------------------------------------------------------
banner "B2 · Rule QC gates G1 … G10"
python -m src.teacher.qc

banner "B3 · PRM step-level critic + best-of rerank within each teacher"
python -m src.teacher.critic
python -m src.teacher.critic_rerank

banner "B4 · Cross-LLM consensus merge + reasoning-safety filter"
python -m src.teacher.merge_consensus
python -m src.teacher.apply_reasoning_safety
python -m src.teacher.audit_teacher_clean

# -----------------------------------------------------------------------------
# B5 — Preference corpora for Phase C
# -----------------------------------------------------------------------------
banner "B5 · Build preference corpora"
python -m src.teacher.build_preference_pairs
python -m src.teacher.build_direction_mirror_preferences
python -m src.teacher.build_phase4_hard_negative_preferences

# -----------------------------------------------------------------------------
# Optional · frontier LLM-as-judge OOF probe
# -----------------------------------------------------------------------------
if [[ "${SKIP_OOF:-0}" == "1" ]]; then
    banner "OOF probe skipped (SKIP_OOF=1)"
else
    if [[ -n "${OPENAI_API_KEY:-}${ANTHROPIC_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
        banner "Optional · LLM-as-judge OOF probe"
        python -m src.teacher.sample_for_judge
        python -m src.teacher.llm_judge
    else
        banner "OOF probe skipped (no OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)"
    fi
fi

banner "Phase B complete."
