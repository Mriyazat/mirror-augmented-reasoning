#!/usr/bin/env bash
# =============================================================================
#  Run the entire CoT-DDI pipeline end-to-end: A → B → C → D.
#
#  Long-running (multi-day on a 4× H100 node). For a partial run, call the
#  individual phase scripts directly or set the SKIP_* / STAGES env vars.
#
#  Usage:
#      bash scripts/run_all.sh
#      CHECKPOINT=outputs/student/dpo/<run>  bash scripts/run_all.sh
# =============================================================================
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

bash scripts/run_phase_a.sh
bash scripts/run_phase_b.sh
bash scripts/run_phase_c.sh
bash scripts/run_phase_d.sh

printf "\n\033[1;32m==> CoT-DDI pipeline complete.\033[0m\n"
