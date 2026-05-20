"""A3 — Pharmacokinetic feature table.

Per-drug PK feature vector combining two sources:

  (1) Structured: drug_proteins.parquet — role in {enzymes, transporters, carriers},
      uniprot-matched to canonical CYPs and drug transporters. Use the `actions`
      field (inhibitor / inducer / substrate / ...) to set Boolean flags.

  (2) Backup from free text: regex over drugs.parquet columns
      {metabolism, pharmacodynamics, mechanism_of_action} for phrases like
      "inhibits CYP3A4" / "substrate of P-gp". This catches drugs that DrugBank
      documents in prose without a structured <enzymes> entry.

Numeric parsing from free text:
  - half_life_hours      — take the first numeric (with unit normalization)
  - protein_binding_pct  — first "%" value or "approximately 95%"

Output: data_processed/pk_features.parquet  (19,853 rows x ~70 cols)
        outputs/audit/a3_pk_report.md

Gate: every drug in drugs.parquet produces one PK row (even if all flags 0).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data_processed"
OUT_PARQUET = IN_DIR / "pk_features.parquet"
AUDIT_MD = ROOT / "outputs" / "audit" / "a3_pk_report.md"

# ── Canonical drug-metabolizing proteins (UniProt → display name) ─────────────
CYPS: dict[str, str] = {
    "P05177": "CYP1A2",
    "P11509": "CYP2A6",
    "P20813": "CYP2B6",
    "P10632": "CYP2C8",
    "P11712": "CYP2C9",
    "P33261": "CYP2C19",
    "P10635": "CYP2D6",
    "P05181": "CYP2E1",
    "P08684": "CYP3A4",
    "P20815": "CYP3A5",
}
TRANSPORTERS: dict[str, str] = {
    "P08183": "P-gp",       # ABCB1 / MDR1
    "Q9UNQ0": "BCRP",       # ABCG2
    "Q9Y6L6": "OATP1B1",    # SLCO1B1
    "Q9NPD5": "OATP1B3",    # SLCO1B3
    "Q4U2R8": "OAT1",       # SLC22A6
    "Q8TCC7": "OAT3",       # SLC22A8
    "O15245": "OCT1",       # SLC22A1
    "O15244": "OCT2",       # SLC22A2
}
ALL_PROTEINS = {**CYPS, **TRANSPORTERS}

ACTION_TO_FLAGS = {
    "inhibitor":     "inh",
    "inducer":       "ind",
    "substrate":     "sub",
    "activator":     "act",
    "weak inhibitor": "inh",
    "strong inhibitor": "inh",
    "moderate inhibitor": "inh",
}

# ── Regex patterns for text-based augmentation ────────────────────────────────
# Each key -> display name used for column naming.
PROTEIN_ALIASES = {
    "CYP1A2":  [r"CYP1A2", r"CYP ?1A2"],
    "CYP2A6":  [r"CYP2A6", r"CYP ?2A6"],
    "CYP2B6":  [r"CYP2B6", r"CYP ?2B6"],
    "CYP2C8":  [r"CYP2C8", r"CYP ?2C8"],
    "CYP2C9":  [r"CYP2C9", r"CYP ?2C9"],
    "CYP2C19": [r"CYP2C19", r"CYP ?2C19"],
    "CYP2D6":  [r"CYP2D6", r"CYP ?2D6"],
    "CYP2E1":  [r"CYP2E1", r"CYP ?2E1"],
    "CYP3A4":  [r"CYP3A4", r"CYP ?3A4"],
    "CYP3A5":  [r"CYP3A5", r"CYP ?3A5"],
    "P-gp":    [r"P-?gp", r"P-?glycoprotein", r"MDR1", r"ABCB1"],
    "BCRP":    [r"BCRP", r"ABCG2"],
    "OATP1B1": [r"OATP1B1", r"SLCO1B1"],
    "OATP1B3": [r"OATP1B3", r"SLCO1B3"],
    "OAT1":    [r"OAT1", r"SLC22A6"],
    "OAT3":    [r"OAT3", r"SLC22A8"],
    "OCT1":    [r"OCT1", r"SLC22A1"],
    "OCT2":    [r"OCT2", r"SLC22A2"],
}

INHIBIT_VERBS = r"(?:inhibit(?:s|ed|ion|or|ing|s of)?|block(?:s|ed|er|ing)?|suppress(?:es|ed|ing)?|reduc(?:es|ed) activity of)"
INDUCE_VERBS  = r"(?:induc(?:es|ed|tion|er|ing)|increas(?:es|ed) activity of)"
SUBSTRATE_NOUNS = r"(?:substrate(?: of|s of)?|metaboliz(?:ed|es|ation) by|metabolism by)"

def compile_patterns() -> dict:
    """Build compiled regexes per protein × action."""
    patterns: dict = {}
    for display, aliases in PROTEIN_ALIASES.items():
        alt = "|".join(aliases)
        patterns[display] = {
            "inh": re.compile(
                rf"\b(?:{INHIBIT_VERBS})\s+(?:\w+\s+){{0,3}}(?:{alt})\b"
                rf"|\b(?:{alt})\s+(?:\w+\s+){{0,3}}(?:is|are)?\s*(?:inhibited|blocked)",
                re.IGNORECASE,
            ),
            "ind": re.compile(
                rf"\b(?:{INDUCE_VERBS})\s+(?:\w+\s+){{0,3}}(?:{alt})\b"
                rf"|\b(?:{alt})\s+(?:\w+\s+){{0,3}}(?:is|are)?\s*induced",
                re.IGNORECASE,
            ),
            "sub": re.compile(
                rf"\b(?:{SUBSTRATE_NOUNS})\s+(?:the\s+)?(?:\w+\s+){{0,3}}(?:{alt})\b"
                rf"|\b(?:{alt})\s+substrate"
                rf"|\bmetaboliz(?:ed|es) by\s+(?:\w+\s+){{0,3}}(?:{alt})\b",
                re.IGNORECASE,
            ),
        }
    return patterns

TEXT_PATTERNS = compile_patterns()

# ── Numeric parsers ───────────────────────────────────────────────────────────
HALF_LIFE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?\s*)?\s*(hour|hours|hr|hrs|h|minute|minutes|min|day|days|d)\b",
    re.IGNORECASE,
)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)")


def parse_half_life_hours(text: str | None) -> float | None:
    if not text:
        return None
    m = HALF_LIFE_RE.search(text)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if unit.startswith("min"):
        return val / 60.0
    if unit.startswith("day") or unit == "d":
        return val * 24.0
    return val  # hours


def parse_protein_binding_pct(text: str | None) -> float | None:
    if not text:
        return None
    m = PCT_RE.search(text)
    if not m:
        return None
    return float(m.group(1))


def elimination_route(text: str | None) -> str | None:
    if not text:
        return None
    tl = text.lower()
    renal = any(k in tl for k in ["urine", "renal", "kidney"])
    hepatic = any(k in tl for k in ["feces", "faeces", "bile", "bilia", "hepatic", "biliary"])
    if renal and hepatic:
        return "mixed"
    if renal:
        return "renal"
    if hepatic:
        return "hepatic"
    return None


def main() -> None:
    t0 = time.time()
    drugs_tbl = pq.read_table(IN_DIR / "drugs.parquet")
    proteins_tbl = pq.read_table(IN_DIR / "drug_proteins.parquet")
    print(f"[A3] loaded {drugs_tbl.num_rows} drugs, {proteins_tbl.num_rows} protein edges")

    # Build drug_id -> {protein_display: set(action_flags)} from structured
    structured: dict[str, dict[str, set[str]]] = {}
    text_source_counts = Counter()  # how often each flag was set from text only

    # Pre-index proteins by drug for quick lookup
    proteins_list = proteins_tbl.to_pylist()
    for p in proteins_list:
        uid = p.get("uniprot")
        if uid not in ALL_PROTEINS:
            continue
        display = ALL_PROTEINS[uid]
        actions = p.get("actions") or []
        drug_id = p["drugbank_id"]
        per_drug = structured.setdefault(drug_id, {})
        flags = per_drug.setdefault(display, set())
        for a in actions:
            a_norm = (a or "").strip().lower()
            if not a_norm:
                continue
            # map variants
            if "inhibit" in a_norm:
                flags.add("inh")
            elif "induc" in a_norm:
                flags.add("ind")
            elif "substrate" in a_norm or "metabol" in a_norm:
                flags.add("sub")
            elif a_norm in ACTION_TO_FLAGS:
                flags.add(ACTION_TO_FLAGS[a_norm])

    print(f"[A3] structured PK annotations for {len(structured)} drugs")

    # Iterate drugs and compose feature rows
    feature_rows: list[dict] = []
    drugs_list = drugs_tbl.to_pylist()
    for d in drugs_list:
        drug_id = d["drugbank_id"]
        per_drug = structured.get(drug_id, {})
        row = {"drugbank_id": drug_id, "name": d.get("name")}

        # Protein flags (structured + text backup)
        text_bag = " ".join(filter(None, [
            d.get("metabolism") or "",
            d.get("pharmacodynamics") or "",
            d.get("mechanism_of_action") or "",
            d.get("description") or "",
        ]))
        for display in PROTEIN_ALIASES:
            structured_flags = per_drug.get(display, set())
            for action in ("inh", "ind", "sub"):
                col = f"{display.replace('-','_')}_{action}".lower()
                flag = action in structured_flags
                source = "structured" if flag else None
                if not flag:
                    pat = TEXT_PATTERNS[display][action]
                    if pat.search(text_bag):
                        flag = True
                        source = "text_regex"
                row[col] = flag
                if source == "text_regex":
                    text_source_counts[f"{display}_{action}"] += 1

        # Numeric PK
        row["half_life_hours"] = parse_half_life_hours(d.get("half_life"))
        row["protein_binding_pct"] = parse_protein_binding_pct(d.get("protein_binding"))
        row["elimination_route"] = elimination_route(d.get("route_of_elimination"))

        # Elimination: renal_clearance hints
        row["has_half_life_value"] = row["half_life_hours"] is not None
        row["has_protein_binding_value"] = row["protein_binding_pct"] is not None

        feature_rows.append(row)

    assert len(feature_rows) == drugs_tbl.num_rows, "row count gate failed"
    print(f"[A3] built {len(feature_rows)} PK rows in {time.time()-t0:.1f}s")

    # Write parquet
    tbl = pa.Table.from_pylist(feature_rows)
    pq.write_table(tbl, OUT_PARQUET, compression="snappy")
    print(f"[A3] wrote {OUT_PARQUET.relative_to(ROOT)} ({OUT_PARQUET.stat().st_size/1e6:.2f} MB)")

    # Audit summary
    def count_true(col: str) -> int:
        return sum(1 for r in feature_rows if r.get(col))

    cyp_cov = {}
    for cyp in CYPS.values():
        for action in ("inh", "ind", "sub"):
            col = f"{cyp}_{action}".lower()
            cyp_cov[col] = count_true(col)
    trans_cov = {}
    for tr in TRANSPORTERS.values():
        for action in ("inh", "ind", "sub"):
            col = f"{tr.replace('-','_')}_{action}".lower()
            trans_cov[col] = count_true(col)

    hl_cov = count_true("has_half_life_value")
    pb_cov = count_true("has_protein_binding_value")
    elim_counter = Counter(r.get("elimination_route") for r in feature_rows)

    md = [
        "# A3 — Pharmacokinetic feature table report\n",
        f"- Input: `data_processed/drugs.parquet` ({drugs_tbl.num_rows} drugs) + "
        f"`data_processed/drug_proteins.parquet` ({proteins_tbl.num_rows} edges)",
        f"- Output: `data_processed/pk_features.parquet` "
        f"({len(feature_rows)} rows × {len(feature_rows[0])} cols, "
        f"{OUT_PARQUET.stat().st_size/1e6:.2f} MB)",
        f"- Build time: {time.time()-t0:.1f}s",
        "",
        "## CYP flag coverage (drug count with flag=True)",
        "| Flag | n_drugs |",
        "|---|---:|",
    ]
    for col, n in sorted(cyp_cov.items()):
        md.append(f"| `{col}` | {n:,} |")
    md += ["", "## Transporter flag coverage", "| Flag | n_drugs |", "|---|---:|"]
    for col, n in sorted(trans_cov.items()):
        md.append(f"| `{col}` | {n:,} |")
    md += [
        "",
        "## Numeric feature coverage",
        f"- Drugs with parsed half_life_hours: **{hl_cov:,}** ({100*hl_cov/len(feature_rows):.2f}%)",
        f"- Drugs with parsed protein_binding_pct: **{pb_cov:,}** ({100*pb_cov/len(feature_rows):.2f}%)",
        f"- Elimination route distribution: `{dict(elim_counter)}`",
        "",
        "## Text-regex augmentation hits (flag set via text only, not structured)",
        "These are drugs that DrugBank did NOT annotate structurally but whose "
        "metabolism/pharmacodynamics prose explicitly mentions the CYP/transporter action. "
        "Useful for recall; will be validated during the retrieval audit (MOR) against retrieval mechanism overlap.",
        "",
        "| Protein_action | n_drugs recovered from text |",
        "|---|---:|",
    ]
    for k, v in sorted(text_source_counts.items(), key=lambda x: -x[1]):
        md.append(f"| `{k}` | {v:,} |")
    AUDIT_MD.write_text("\n".join(md) + "\n")
    print(f"[A3] wrote {AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
