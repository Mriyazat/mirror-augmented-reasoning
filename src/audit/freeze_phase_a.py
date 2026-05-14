"""Freeze Phase A — SHA256 every artifact the paper will cite.

Produces:
    outputs/audit/phase_a_artifacts_sha256.txt   (human-readable, sorted)
    outputs/audit/phase_a_artifacts_sha256.json  (machine-readable)

Run at the end of Phase A before Phase B starts.  Any later edit to a
Phase-A artifact that breaks these hashes should be a deliberate event
(i.e. a new freeze + a dated audit note) — not accidental regeneration.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_TXT = ROOT / "outputs" / "audit" / "phase_a_artifacts_sha256.txt"
OUT_JSON = ROOT / "outputs" / "audit" / "phase_a_artifacts_sha256.json"

# Artifacts the paper will cite, grouped.
# (group, relative_path, short_purpose)
ARTIFACTS: list[tuple[str, str, str]] = [
    # ── A2 — DrugBank extraction ─────────────────────────────────────────────
    ("A2", "data_processed/drugs.parquet",            "19,853 drugs × 56 fields"),
    ("A2", "data_processed/pairs.parquet",            "1,456,772 canonical pairs"),
    ("A2", "data_processed/drug_pathways.parquet",    "SMPDB drug↔pathway edges"),
    ("A2", "data_processed/drug_proteins.parquet",    "targets/enzymes/transporters/carriers"),
    ("A2", "data_processed/drug_xref.parquet",        "KEGG/PubChem/ChEBI/ChEMBL/… cross-refs"),
    ("A2", "data_processed/drug_reactions.parquet",   "metabolic reactions"),
    ("A2", "data_processed/drug_snps.parquet",        "SNP effects + SNP ADRs"),
    ("A2", "data_processed/drug_brands.parquet",      "product + international brand aliases"),
    # ── A3 — PK features ──────────────────────────────────────────────────────
    ("A3", "data_processed/pk_features.parquet",      "CYP/P-gp/BCRP/OATP flags + numeric PK"),
    # ── A4 — KEGG + SMPDB pathway unification ─────────────────────────────────
    ("A4", "data_processed/kegg_pathways.parquet",          "KEGG pathway definitions"),
    ("A4", "data_processed/kegg_drug_pathways.parquet",     "KEGG drug↔pathway via DrugBank xref"),
    ("A4", "data_processed/kegg_compound_pathways.parquet", "KEGG compound↔pathway via DrugBank xref"),
    ("A4", "data_processed/pathways_unified.parquet",       "SMPDB + KEGG unified pathway edges"),
    # ── A5 — Hierarchical taxonomy ────────────────────────────────────────────
    ("A5", "data_processed/labels_hierarchical.parquet", "1,453,987 pairs × (family,subtype,direction,polarity)"),
    ("A5", "data_processed/taxonomy_schema.json",        "family/subtype vocab + collapse rules"),
    ("A5", "outputs/audit/a5_dropped_pairs.parquet",     "2,785 unmatched/ambiguous pairs (transparency)"),
    # ── A6 — Pair signatures (retrieval keys) ─────────────────────────────────
    ("A6", "data_processed/pair_signatures.parquet",  "per-pair pathway/protein/SMILES/ATC similarities"),
    # ── A8 — Splits + baseline ────────────────────────────────────────────────
    ("A8", "data_processed/splits/manifest_random_full.parquet", "transductive split manifest"),
    ("A8", "data_processed/splits/manifest_drug_cold.parquet",   "inductive (one-sided unseen)"),
    ("A8", "data_processed/splits/manifest_pair_cold.parquet",   "strict inductive (both drugs unseen)"),
    ("A8", "data_processed/splits/manifest_subset25k.parquet",   "25k train-only dev subset"),
    ("A8", "data_processed/splits/splits_sha256.json",            "manifest-file hashes (self-referential)"),
    ("A8", "outputs/audit/a8_xgb_random_full.json", "XGBoost metrics — random_full (reported in paper)"),
    ("A8", "outputs/audit/a8_xgb_drug_cold.json",   "XGBoost metrics — drug_cold (reported in paper)"),
    ("A8", "outputs/audit/a8_xgb_pair_cold.json",   "XGBoost metrics — pair_cold (reported in paper)"),
    # ── Audit manifests ───────────────────────────────────────────────────────
    ("Audit", "data_processed/drug_completeness.parquet", "per-drug per-field availability bool matrix"),
    # ── Phase-A summary reports (markdown) ────────────────────────────────────
    ("Report", "outputs/audit/phase_a_summary.md",              "Phase A umbrella summary"),
    ("Report", "outputs/audit/pre_flight_report.md",             "A0 pre-flight"),
    ("Report", "outputs/audit/a2_extraction_report.md",          "A2 extraction audit"),
    ("Report", "outputs/audit/a3_pk_report.md",                  "A3 PK audit"),
    ("Report", "outputs/audit/a4_kegg_report.md",                "A4 KEGG audit"),
    ("Report", "outputs/audit/a5_taxonomy_report.md",            "A5 taxonomy audit"),
    ("Report", "outputs/audit/a5_integrity_report.md",           "A5 integrity gates (8/8 PASS)"),
    ("Report", "outputs/audit/a6_signatures_report.md",          "A6 signatures audit"),
    ("Report", "outputs/audit/a7_mor_report.md",                 "A7 MOR retrieval validation"),
    ("Report", "outputs/audit/a8_splits_report.md",              "A8 splits + leakage audit"),
    ("Report", "outputs/audit/a8_xgboost_report.md",             "A8.2 XGBoost baseline"),
    ("Report", "outputs/audit/drug_completeness_report.md",      "drug-level completeness audit"),
]


def sha256_file(p: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 20):  # 1 MB
            h.update(chunk)
    return h.hexdigest(), p.stat().st_size


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    frozen_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    missing = []

    for group, rel, purpose in ARTIFACTS:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        sha, size = sha256_file(p)
        records.append({
            "group": group,
            "path": rel,
            "sha256": sha,
            "bytes": size,
            "purpose": purpose,
        })

    records.sort(key=lambda r: (r["group"], r["path"]))

    # ── JSON output ──────────────────────────────────────────────────────────
    OUT_JSON.write_text(json.dumps({
        "frozen_at_utc": frozen_at,
        "phase": "A",
        "n_artifacts": len(records),
        "artifacts": records,
        "missing": missing,
    }, indent=2))

    # ── Human-readable output ────────────────────────────────────────────────
    lines = [
        f"# Phase A artifact SHA256 freeze",
        f"# frozen_at_utc: {frozen_at}",
        f"# n_artifacts:   {len(records)}",
        f"# missing:       {len(missing)}",
        f"#",
        f"# Columns: sha256  size  group  relative_path  # purpose",
        f"#",
    ]
    if missing:
        lines.append("# MISSING (expected but not found):")
        for m in missing:
            lines.append(f"#   - {m}")
        lines.append("#")
    for r in records:
        lines.append(
            f"{r['sha256']}  "
            f"{human_bytes(r['bytes']):>10}  "
            f"[{r['group']:<6}] "
            f"{r['path']:<60} "
            f"# {r['purpose']}"
        )

    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"[freeze] wrote {OUT_TXT.relative_to(ROOT)} ({len(records)} artifacts, {len(missing)} missing)")
    print(f"[freeze] wrote {OUT_JSON.relative_to(ROOT)}")
    if missing:
        print(f"[freeze] WARNING — {len(missing)} expected artifact(s) missing:")
        for m in missing:
            print(f"         - {m}")


if __name__ == "__main__":
    main()
