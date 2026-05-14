#!/usr/bin/env bash
# =============================================================================
#  Phase A — Corpus, taxonomy, splits, audits
#
#  Reads:   data_raw/drugbank_2026-04.xml  (+ optional KEGG / SMPDB / DDInter)
#  Writes:  data_processed/*.parquet, outputs/audit/*, outputs/splits/*
#
#  Usage:
#      bash scripts/run_phase_a.sh
#
#  Optional env vars:
#      SKIP_AUDIT=1     # skip the optional Phase A audit steps
# =============================================================================
set -euo pipefail

banner() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

banner "A1 · Parse DrugBank XML → parquet"
python -m src.data.parse_drugbank

banner "A2 · Pathway / target enrichment (KEGG + SMPDB) and per-drug PK flags"
python -m src.data.fetch_pathways
python -m src.data.build_pk_table
python -m src.data.build_signatures

banner "A3 · Hierarchical mechanism taxonomy"
python -m src.data.build_taxonomy

banner "A4 · Splits (random_full / drug_cold / pair_cold + balanced teacher subset)"
python -m src.data.build_splits

if [[ "${SKIP_AUDIT:-0}" == "1" ]]; then
    banner "A5 · audits skipped (SKIP_AUDIT=1)"
else
    banner "A5 · Audits + GO/NO-GO freeze"
    python -m src.audit.a06_label_cooccurrence
    python -m src.audit.a07_ddinter_severity
    python -m src.audit.drug_completeness
    python -m src.audit.freeze_phase_a
fi

banner "Phase A complete."
