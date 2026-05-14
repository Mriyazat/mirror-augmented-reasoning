"""A0.7 — DDInter 2.0 severity extraction + agreement with DrugBank pairs.

Pipeline:
  1. Load all 8 DDInter 2.0 ATC CSVs -> unified table with (Drug_A, Drug_B, Level).
  2. Build drug-name <-> DrugBank-ID mapping from DrugBank XML (first pass, quick —
     reuses name lookup by iterating just <drug>/<name> + synonyms).
  3. Match DDInter pairs to DrugBank pairs via exact lowercase name (+ synonym) lookup.
  4. Compute severity distribution + coverage + agreement stats.

Outputs (DDI/outputs/audit/):
  a07_ddinter_unified.csv         # cleaned merged CSV (all 8 files)
  a07_severity_coverage.json      # coverage + confusion-style stats
  a07_report.md                   # human readable

Severity is **metadata-only** for V4 — used for stratified evaluation, never a
prediction target.
"""
from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
DDINTER_DIR = ROOT / "data_raw" / "ddinter_2"
DRUGBANK_XML = ROOT / "data_raw" / "drugbank_2026-04.xml"
A06_PAIRS = ROOT / "outputs" / "audit" / "a06_pair_label_counts.jsonl"
OUT = ROOT / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

NS = "{http://www.drugbank.ca}"
DRUG = f"{NS}drug"
DB_ID_TAG = f"{NS}drugbank-id"
NAME_TAG = f"{NS}name"
SYN_TAG = f"{NS}synonym"
SYNS_TAG = f"{NS}synonyms"


def load_ddinter() -> tuple[list[dict], Counter]:
    """Load all DDInter 2.0 CSVs into a deduped list of rows + severity counter."""
    rows: dict[tuple[str, str], str] = {}  # canonical (drug_a_lc, drug_b_lc) -> severity
    for f in sorted(DDINTER_DIR.glob("ddinter_downloads_code_*.csv")):
        with f.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                a = (r.get("Drug_A") or "").strip().lower()
                b = (r.get("Drug_B") or "").strip().lower()
                level = (r.get("Level") or "").strip()
                if not a or not b or not level:
                    continue
                key = tuple(sorted((a, b)))
                if key in rows and rows[key] != level:
                    # conflict — keep the more severe
                    order = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
                    if order.get(level, 0) > order.get(rows[key], 0):
                        rows[key] = level
                else:
                    rows[key] = level
    severity_counts = Counter(rows.values())
    out = [{"drug_a_lc": a, "drug_b_lc": b, "level": lvl} for (a, b), lvl in rows.items()]
    return out, severity_counts


def build_name_to_dbid() -> dict[str, str]:
    """First-pass quick parse: DrugBank name + all synonyms -> primary DrugBank ID (lowercased)."""
    t0 = time.time()
    name_to_id: dict[str, str] = {}
    ctx = ET.iterparse(str(DRUGBANK_XML), events=("start", "end"))
    depth = 0
    current_id = None
    current_name = None
    current_synonyms: list[str] = []
    n_drugs = 0
    for event, elem in ctx:
        if event == "start":
            if elem.tag == DRUG:
                depth += 1
                if depth == 1:
                    current_id = None
                    current_name = None
                    current_synonyms = []
        else:
            if elem.tag == DRUG:
                if depth == 1:
                    n_drugs += 1
                    if current_id and current_name:
                        name_to_id.setdefault(current_name.strip().lower(), current_id)
                        for syn in current_synonyms:
                            if syn:
                                name_to_id.setdefault(syn.strip().lower(), current_id)
                elem.clear()
                depth -= 1
                continue
            if depth != 1:
                continue
            if elem.tag == DB_ID_TAG and current_id is None:
                current_id = elem.text
            elif elem.tag == NAME_TAG and current_name is None:
                current_name = elem.text
            elif elem.tag == SYN_TAG:
                if elem.text:
                    current_synonyms.append(elem.text)
    print(f"[audit] built name->id map: {n_drugs} drugs, {len(name_to_id)} name/synonym keys, {time.time()-t0:.0f}s", flush=True)
    return name_to_id


def load_a06_pairs() -> set[tuple[str, str]]:
    """Load A0.6 canonical pair set (DrugBank IDs)."""
    pairs: set[tuple[str, str]] = set()
    with A06_PAIRS.open() as fh:
        for line in fh:
            rec = json.loads(line)
            a, b = rec["pair"].split("|")
            pairs.add(tuple(sorted((a, b))))
    return pairs


def main() -> None:
    assert DRUGBANK_XML.exists(), f"missing {DRUGBANK_XML}"
    assert DDINTER_DIR.exists(), f"missing {DDINTER_DIR}"
    assert A06_PAIRS.exists(), f"A0.6 outputs missing at {A06_PAIRS}"

    # --- Step 1: unify DDInter ---
    print("[audit] loading DDInter 2.0 CSVs ...", flush=True)
    ddinter_rows, severity_counts = load_ddinter()
    print(f"[audit] {len(ddinter_rows)} unique DDInter pairs  severity={dict(severity_counts)}", flush=True)

    unified_csv = OUT / "a07_ddinter_unified.csv"
    with unified_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["drug_a_lc", "drug_b_lc", "level"])
        w.writeheader()
        for r in ddinter_rows:
            w.writerow(r)

    # --- Step 2: build drug-name -> DrugBank ID map ---
    print("[audit] building DrugBank name->ID map ...", flush=True)
    name_to_id = build_name_to_dbid()

    # --- Step 3: match ---
    drugbank_pairs = load_a06_pairs()
    print(f"[audit] {len(drugbank_pairs)} DrugBank canonical pairs loaded from A0.6", flush=True)

    matched = 0
    unmatched_a = 0  # drug_a not in DrugBank name map
    unmatched_b = 0
    both_unmatched = 0
    in_ddinter_in_drugbank = 0  # DDInter pair also exists in DrugBank
    in_ddinter_not_drugbank = 0  # DDInter pair NOT in DrugBank (DDInter-only)
    severity_by_match: dict[str, Counter] = {"matched": Counter(), "unmatched_drugbank": Counter()}

    for row in ddinter_rows:
        a_id = name_to_id.get(row["drug_a_lc"])
        b_id = name_to_id.get(row["drug_b_lc"])
        if not a_id and not b_id:
            both_unmatched += 1
        elif not a_id:
            unmatched_a += 1
        elif not b_id:
            unmatched_b += 1
        else:
            matched += 1
            pair = tuple(sorted((a_id, b_id)))
            if pair in drugbank_pairs:
                in_ddinter_in_drugbank += 1
                severity_by_match["matched"][row["level"]] += 1
            else:
                in_ddinter_not_drugbank += 1
                severity_by_match["unmatched_drugbank"][row["level"]] += 1

    # Reverse: how many DrugBank pairs have DDInter severity?
    ddinter_pair_set: set[tuple[str, str]] = set()
    for row in ddinter_rows:
        a_id = name_to_id.get(row["drug_a_lc"])
        b_id = name_to_id.get(row["drug_b_lc"])
        if a_id and b_id:
            ddinter_pair_set.add(tuple(sorted((a_id, b_id))))
    drugbank_with_severity = len(drugbank_pairs & ddinter_pair_set)

    summary = {
        "n_ddinter_unique_pairs": len(ddinter_rows),
        "n_drugbank_canonical_pairs": len(drugbank_pairs),
        "ddinter_severity_distribution": dict(severity_counts),
        "match_stats": {
            "matched_both_drugs": matched,
            "in_ddinter_in_drugbank": in_ddinter_in_drugbank,
            "in_ddinter_not_drugbank": in_ddinter_not_drugbank,
            "unmatched_drug_a_name": unmatched_a,
            "unmatched_drug_b_name": unmatched_b,
            "both_drugs_unmatched": both_unmatched,
        },
        "drugbank_pairs_with_ddinter_severity": drugbank_with_severity,
        "pct_drugbank_pairs_with_severity": round(
            100.0 * drugbank_with_severity / max(len(drugbank_pairs), 1), 3
        ),
        "severity_breakdown_by_match": {
            k: dict(v) for k, v in severity_by_match.items()
        },
    }
    (OUT / "a07_severity_coverage.json").write_text(json.dumps(summary, indent=2))

    # Report
    md = [
        "# A0.7 — DDInter 2.0 Severity Coverage vs DrugBank 2026-04\n",
        f"- DDInter unified pairs: **{len(ddinter_rows):,}**",
        f"- Severity distribution (DDInter): **{dict(severity_counts)}**",
        f"- DrugBank canonical pairs: **{len(drugbank_pairs):,}**",
        f"- DrugBank pairs covered by DDInter severity: **{drugbank_with_severity:,}** "
        f"({summary['pct_drugbank_pairs_with_severity']}%)",
        f"- DDInter pairs mappable to DrugBank IDs (both drugs): **{matched:,}**",
        f"- Of those, **{in_ddinter_in_drugbank:,}** exist as DrugBank interactions, "
        f"**{in_ddinter_not_drugbank:,}** are DDInter-only",
        f"- Unmatched — drug_a name not in DrugBank: {unmatched_a:,}",
        f"- Unmatched — drug_b name not in DrugBank: {unmatched_b:,}",
        f"- Both unmatched: {both_unmatched:,}",
        "\n## Severity breakdown of matched DrugBank pairs\n",
        "| Level | matched (both in DrugBank) | mapped-but-no-DrugBank-edge |",
        "|---|---:|---:|",
    ]
    for lvl in ["Major", "Moderate", "Minor", "Unknown"]:
        m = severity_by_match["matched"].get(lvl, 0)
        u = severity_by_match["unmatched_drugbank"].get(lvl, 0)
        md.append(f"| {lvl} | {m:,} | {u:,} |")
    md.append("\n## Decision\n")
    md.append("- **DDInter severity is metadata only** for V4. Attached to each DrugBank pair "
              "when available. Used at **evaluation time** to compute severity-stratified metrics "
              "(Major pairs are high-stakes — abstention utility should be higher there).")
    md.append("- **We do not predict severity.** Our prediction target is the mechanism/label.")
    (OUT / "a07_report.md").write_text("\n".join(md) + "\n")

    print(f"[audit] wrote {unified_csv}")
    print(f"[audit] wrote {OUT/'a07_severity_coverage.json'}")
    print(f"[audit] wrote {OUT/'a07_report.md'}")


if __name__ == "__main__":
    main()
