#!/usr/bin/env bash
# =============================================================================
#  Phase D — Evaluation
#
#  Builds the evaluation prompts with the top-K mechanism-aware neighbour
#  block, runs JSON-constrained student inference, applies conformal +
#  entropy abstention, runs the XGBoost reference and the full 8-metric
#  suite on all three split protocols, and optionally runs the stress sets
#  (adversarial / counterfactual / polypharmacy).
#
#  Reads:   the trained LoRA checkpoint  (env: CHECKPOINT)
#  Writes:  outputs/predictions/<split>.jsonl
#           outputs/results/<run-name>/{metrics.json, per_family.json, ...}
#
#  Usage:
#      CHECKPOINT=outputs/student/dpo/<run-name>  bash scripts/run_phase_d.sh
#
#  Optional env vars:
#      CHECKPOINT   path to the LoRA adapter to evaluate (required for D2)
#      SPLITS       splits to predict on
#                   (default: "random_full drug_cold pair_cold")
#      SKIP_STRESS  1 → skip D5 stress sets
# =============================================================================
set -euo pipefail

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

SPLITS="${SPLITS:-random_full drug_cold pair_cold}"

banner "D1 · Build evaluation prompts with top-K neighbour block"
python -m src.data.build_student_eval_prompts

if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "[run_phase_d] WARNING: CHECKPOINT not set — skipping D2 inference + abstention."
    echo "              Re-run with:  CHECKPOINT=<path> bash scripts/run_phase_d.sh"
else
    for split in $SPLITS; do
        banner "D2 · Student inference + JSON parse — split=$split"
        python -m src.inference.predict --split "$split" --checkpoint "$CHECKPOINT"
    done

    banner "D2 · Conformal + entropy abstention"
    python -m src.inference.abstention
fi

banner "D3 · XGBoost reference (same 4-component feature pool)"
python -m src.evaluation.baseline_xgboost

banner "D4 · Full 8-metric suite × 3 splits"
python -m src.evaluation.run_full_eval

if [[ "${SKIP_STRESS:-0}" == "1" ]]; then
    banner "D5 · stress sets skipped (SKIP_STRESS=1)"
else
    banner "D5 · Stress sets (adversarial / counterfactual / polypharmacy)"
    python -m src.data.build_adversarial
    python -m src.data.build_counterfactual
    python -m src.data.build_polypharmacy
fi

banner "Phase D complete."
