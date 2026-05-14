"""Drug-level data completeness manifest.

For every drug in drugs.parquet we record which data sources are present
(SMILES, InChI, description, MoA, metabolism, half-life, ATC, pathways,
proteins, KEGG/xref IDs) and whether the drug participates in any pair.

Outputs:
    data_processed/drug_completeness.parquet   # one row per drug, per-field bool
    outputs/audit/drug_completeness_report.md  # human summary
    outputs/audit/drugs_missing_smiles.txt     # drugbank_ids missing SMILES
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
OUT_PQ = DATA / "drug_completeness.parquet"
OUT_MD = ROOT / "outputs" / "audit" / "drug_completeness_report.md"
OUT_MISS_SMILES = ROOT / "outputs" / "audit" / "drugs_missing_smiles.txt"


def _has(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    try:
        return len(v) > 0
    except TypeError:
        return True


def main() -> None:
    print("[DC] loading drugs.parquet ...")
    drugs = pq.read_table(DATA / "drugs.parquet").to_pylist()
    n_drugs = len(drugs)
    print(f"[DC] {n_drugs:,} drugs")

    print("[DC] loading auxiliary tables ...")
    pathways = pq.read_table(DATA / "pathways_unified.parquet",
                             columns=["drugbank_id"]).to_pylist()
    proteins = pq.read_table(DATA / "drug_proteins.parquet",
                             columns=["drugbank_id"]).to_pylist()
    xref = pq.read_table(DATA / "drug_xref.parquet",
                         columns=["drugbank_id", "resource"]).to_pylist()
    pairs = pq.read_table(DATA / "pairs.parquet",
                          columns=["a_id", "b_id"]).to_pylist()
    pk = pq.read_table(DATA / "pk_features.parquet",
                       columns=["drugbank_id"]).to_pylist()

    has_pathway = {r["drugbank_id"] for r in pathways}
    has_protein = {r["drugbank_id"] for r in proteins}
    has_pk = {r["drugbank_id"] for r in pk}
    has_xref = defaultdict(set)
    for r in xref:
        has_xref[r["drugbank_id"]].add(r["resource"])

    participates_in_pair = set()
    for p in pairs:
        participates_in_pair.add(p["a_id"])
        participates_in_pair.add(p["b_id"])

    rows = []
    for d in drugs:
        did = d["drugbank_id"]
        x = has_xref.get(did, set())
        rows.append({
            "drugbank_id": did,
            "name": d.get("name"),
            "in_any_pair": did in participates_in_pair,
            # Structural
            "has_smiles": _has(d.get("smiles")),
            "has_inchi": _has(d.get("inchi")),
            "has_mw": _has(d.get("mw")),
            "has_logp": _has(d.get("logp")),
            # Text
            "has_description": _has(d.get("description")),
            "has_mechanism_of_action": _has(d.get("mechanism_of_action")),
            "has_pharmacodynamics": _has(d.get("pharmacodynamics")),
            "has_metabolism_text": _has(d.get("metabolism")),
            "has_absorption_text": _has(d.get("absorption")),
            "has_half_life": _has(d.get("half_life")),
            "has_protein_binding": _has(d.get("protein_binding")),
            # IDs / classification
            "has_atc": _has(d.get("atc_codes")),
            "has_synonyms": _has(d.get("synonyms")),
            "has_categories": _has(d.get("categories")),
            # Auxiliary tables
            "has_pathway_edges": did in has_pathway,
            "has_protein_edges": did in has_protein,
            "has_pk_features": did in has_pk,
            "has_kegg_xref": "KEGG Drug" in x or "KEGG Compound" in x,
            "has_pubchem_xref": ("PubChem Compound" in x) or ("PubChem Substance" in x),
            "has_chebi_xref": "ChEBI" in x,
            "has_chembl_xref": "ChEMBL" in x,
            "has_uniprot_xref": "UniProtKB" in x,
        })

    pq.write_table(pa.Table.from_pylist(rows), OUT_PQ, compression="snappy")
    print(f"[DC] wrote {OUT_PQ.relative_to(ROOT)} ({len(rows):,} rows)")

    # Also dump plain-text list of DBids missing SMILES (for manual inspection / PubChem backfill)
    no_smiles = [r["drugbank_id"] for r in rows if not r["has_smiles"]]
    OUT_MISS_SMILES.write_text("\n".join(no_smiles) + "\n")
    print(f"[DC] wrote {OUT_MISS_SMILES.relative_to(ROOT)} ({len(no_smiles):,} drugs w/o SMILES)")

    # Markdown report
    n_in_pairs = sum(1 for r in rows if r["in_any_pair"])
    bool_fields = [c for c in rows[0].keys()
                   if c not in ("drugbank_id", "name", "in_any_pair")]

    def pct(count, total):
        return f"{count:,}/{total:,}  ({100*count/total:5.1f}%)" if total else "—"

    md = [
        "# Drug-level data completeness manifest\n",
        f"- Total drugs in DrugBank 2026-04: **{n_drugs:,}**",
        f"- Drugs appearing in ≥1 DDI pair: **{n_in_pairs:,}** "
        f"({100*n_in_pairs/n_drugs:.1f}%)  ← the population that matters",
        f"- Drugs *not* appearing in any pair (lonely): **{n_drugs - n_in_pairs:,}** "
        f"(mostly brand-new / biotech / OTC drugs without documented DDIs — irrelevant for V4)",
        "",
        "## Coverage on the pair-participating population",
        "",
        "| Field | All drugs | Pair drugs only |",
        "|---|---|---|",
    ]
    for c in bool_fields:
        n_all = sum(1 for r in rows if r[c])
        n_pair = sum(1 for r in rows if r[c] and r["in_any_pair"])
        md.append(f"| `{c}` | {pct(n_all, n_drugs)} | {pct(n_pair, n_in_pairs)} |")

    # Critical-field drug lists
    md.append("")
    md.append("## Downstream implications")
    pair_no_smiles = [r for r in rows if r["in_any_pair"] and not r["has_smiles"]]
    pair_no_path = [r for r in rows if r["in_any_pair"] and not r["has_pathway_edges"]]
    pair_no_prot = [r for r in rows if r["in_any_pair"] and not r["has_protein_edges"]]
    pair_no_moa = [r for r in rows if r["in_any_pair"] and not r["has_mechanism_of_action"]]
    md += [
        f"- Pair-drugs missing SMILES: **{len(pair_no_smiles):,}** "
        f"→ SMILES Tanimoto signal unavailable for their pairs (XGBoost uses sentinel 0-row; teacher RAG will omit structural context).",
        f"- Pair-drugs missing pathway edges: **{len(pair_no_path):,}** "
        f"→ pathway Jaccard = null; relies on protein/SMILES fallback tiers.",
        f"- Pair-drugs missing protein edges (targets/enzymes/transporters/carriers): **{len(pair_no_prot):,}** "
        f"→ protein Jaccard = null; relies on SMILES/ATC fallback.",
        f"- Pair-drugs missing mechanism-of-action text: **{len(pair_no_moa):,}** "
        f"→ teacher prompt will lack explicit MoA (fallback: description + pharmacology text).",
        "",
        f"Full drug-ID list of the {len(pair_no_smiles):,} pair-drugs without SMILES: "
        f"`outputs/audit/drugs_missing_smiles.txt`",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[DC] wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
