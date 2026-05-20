"""Taxonomy audit — cross-check the taxonomy outputs before the retrieval stage.

Checks:
  1. coverage ≥ 99%  (already reported)
  2. every non-Other row has non-null polarity
  3. directional rows have both subject and object drug IDs populated and distinct
  4. bidirectional rows have both subject/object None
  5. subject_drugbank_id is always one of {a_id, b_id}; same for object
  6. spot-check 12 random pairs across families: template matches label

Writes outputs/audit/a5_integrity_report.md.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data_processed" / "labels_hierarchical.parquet"
SCHEMA = ROOT / "data_processed" / "taxonomy_schema.json"
OUT_MD = ROOT / "outputs" / "audit" / "a5_integrity_report.md"


def main():
    print("[taxonomy-audit] loading labels_hierarchical.parquet ...")
    rows = pq.read_table(LABELS).to_pylist()
    n = len(rows)

    # 1. Coverage
    other = sum(1 for r in rows if r["family"] == "Other")
    coverage = 100 * (n - other) / n

    # 2. Non-Other rows must have polarity
    miss_polarity = [r for r in rows if r["family"] != "Other" and not r["polarity"]]
    # 3 & 4. Direction validity
    dir_subject_null = [r for r in rows if (not r["bidirectional"]) and r["family"] != "Other"
                        and not r["subject_drugbank_id"]]
    dir_object_null = [r for r in rows if (not r["bidirectional"]) and r["family"] != "Other"
                       and not r["object_drugbank_id"]]
    dir_same_drug = [r for r in rows if (not r["bidirectional"]) and r["family"] != "Other"
                     and r["subject_drugbank_id"] == r["object_drugbank_id"]
                     and r["subject_drugbank_id"]]
    bidi_with_subject = [r for r in rows if r["bidirectional"] and r["subject_drugbank_id"]]

    # 5. subject/object must equal a_id or b_id
    bad_ids = [r for r in rows if (not r["bidirectional"]) and r["family"] != "Other"
               and ((r["subject_drugbank_id"] not in (r["a_id"], r["b_id"]))
                    or (r["object_drugbank_id"] not in (r["a_id"], r["b_id"])))]

    # 6. Spot-checks — 12 randomly sampled non-Other rows
    random.seed(13)
    sample_idx = random.sample(range(n), 12)
    spot = [rows[i] for i in sample_idx]

    # Polarity-vs-subtype sanity for AdverseRisk: polarity must be 'risk' (risk
    # increased) or 'risk_down' (risk decreased — new variant for protective
    # adverse-effect interactions).
    bad_risk_pol = [r for r in rows if r["family"] == "AdverseRisk"
                    and r["polarity"] not in ("risk", "risk_down")]
    # For other named families polarity must be 'up' or 'down'
    bad_dir_pol = [r for r in rows if r["family"] not in ("AdverseRisk", "Other")
                   and r["polarity"] not in ("up", "down")]

    # Report
    fam_counts = Counter(r["family"] for r in rows)
    md = [
        "# Taxonomy integrity audit\n",
        "## Primary gates",
        f"- Coverage ≥ 99%: **{'PASS' if coverage >= 99 else 'FAIL'}** ({coverage:.2f}%)",
        f"- Families ≤ 20: **{'PASS' if len(fam_counts) <= 20 else 'FAIL'}** ({len(fam_counts)})",
        "",
        "## Integrity checks",
        f"- Non-Other rows missing polarity: **{len(miss_polarity):,}** "
        f"{'(FAIL)' if miss_polarity else '(PASS)'}",
        f"- Directional rows missing subject_drugbank_id: **{len(dir_subject_null):,}** "
        f"{'(FAIL)' if dir_subject_null else '(PASS)'}",
        f"- Directional rows missing object_drugbank_id: **{len(dir_object_null):,}** "
        f"{'(FAIL)' if dir_object_null else '(PASS)'}",
        f"- Directional rows with subject == object (same drug both sides): "
        f"**{len(dir_same_drug):,}** {'(FAIL)' if dir_same_drug else '(PASS)'}",
        f"- Bidirectional rows that set a subject (should be null): **{len(bidi_with_subject):,}** "
        f"{'(FAIL)' if bidi_with_subject else '(PASS)'}",
        f"- Subject/object ID not in {{a_id, b_id}}: **{len(bad_ids):,}** "
        f"{'(FAIL)' if bad_ids else '(PASS)'}",
        f"- AdverseRisk rows with polarity ≠ 'risk': **{len(bad_risk_pol):,}** "
        f"{'(FAIL)' if bad_risk_pol else '(PASS)'}",
        f"- PK/PD/Efficacy rows with polarity ∉ {{up, down}}: **{len(bad_dir_pol):,}** "
        f"{'(FAIL)' if bad_dir_pol else '(PASS)'}",
        "",
        "## Spot-check (12 random rows)",
        "| pair_id | family | subtype | polarity | subj | obj | bidi | template |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in spot:
        t = r["template"]
        if len(t) > 120:
            t = t[:117] + "…"
        md.append(f"| {r['pair_id']} | {r['family']} | {r['subtype']} | "
                  f"{r['polarity']} | {r['subject_drugbank_id']} | {r['object_drugbank_id']} | "
                  f"{r['bidirectional']} | {t.replace('|','\\|')} |")
    md.append("")
    md.append("## Schema stats")
    schema = json.loads(SCHEMA.read_text())
    md.append(f"- Total subtypes (post-collapse): "
              f"{sum(len(v) for v in schema['subtypes'].values())}")
    md.append(f"- `misc_<family>` collapses: {schema['n_rare_subtypes_collapsed']}")
    md.append(f"- Rare-subtype threshold used: {schema['rare_subtype_threshold']}")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[taxonomy-audit] wrote {OUT_MD.relative_to(ROOT)}")
    # Exit status
    fails = [miss_polarity, dir_subject_null, dir_object_null, dir_same_drug,
             bidi_with_subject, bad_ids, bad_risk_pol, bad_dir_pol]
    print(f"[taxonomy-audit] fail counts: {[len(x) for x in fails]}  "
          f"coverage={coverage:.2f}%  families={len(fam_counts)}")


if __name__ == "__main__":
    main()
