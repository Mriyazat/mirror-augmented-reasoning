"""A4 — KEGG pathway pull + unify with SMPDB.

Pulls four small dumps from the KEGG REST API (one HTTP request each,
no per-drug looping, so it's complete in ~15 s):

  1. link/pathway/drug       → 3k drug→pathway edges
  2. link/pathway/compound   → 20k compound→pathway edges  (we have 12.5%
                                coverage via KEGG Compound IDs, better than
                                KEGG Drug's 10.9%)
  3. list/pathway/map        → 585 pathway names (map-id → display name)
  4. list/drug               → 12k KEGG drug names (for sanity)

Then:
  - Load data_processed/drug_xref.parquet
  - Join DrugBank drugs to KEGG Drug IDs and KEGG Compound IDs
  - Expand to drug→pathway edges (de-dup)
  - Merge with SMPDB edges from data_processed/drug_pathways.parquet into
    pathways_unified.parquet (drugbank_id, source ∈ {smpdb, kegg_drug, kegg_compound},
    pathway_id, pathway_name, category)
  - Plus per-pathway metadata table pathways_metadata.parquet

Outputs (data_processed/):
  kegg_drug_pathways.parquet       # raw KEGG drug→pathway edges (drug IDs)
  kegg_compound_pathways.parquet   # raw KEGG compound→pathway edges
  kegg_pathways.parquet            # pathway_id → name (map.....)
  pathways_unified.parquet         # DrugBank drug_id → pathway (SMPDB+KEGG merged)
  outputs/audit/a4_kegg_report.md
"""
from __future__ import annotations

import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data_processed"
OUT_DIR = IN_DIR
AUDIT_MD = ROOT / "outputs" / "audit" / "a4_kegg_report.md"

KEGG_LINK_DRUG_PATHWAY = "https://rest.kegg.jp/link/pathway/drug"
KEGG_LINK_CPD_PATHWAY  = "https://rest.kegg.jp/link/pathway/compound"
KEGG_LIST_PATHWAY_MAP  = "https://rest.kegg.jp/list/pathway/map"
KEGG_LIST_DRUG         = "https://rest.kegg.jp/list/drug"


def fetch_lines(url: str, timeout: int = 60) -> list[str]:
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        txt = r.read().decode()
    lines = [ln for ln in txt.strip().split("\n") if ln.strip()]
    print(f"  GET {url} → {len(lines)} lines in {time.time()-t0:.1f}s", flush=True)
    return lines


def parse_link_tsv(lines: list[str]) -> list[tuple[str, str]]:
    """KEGG /link output: LHS<TAB>RHS per line (e.g. 'dr:D00001\tpath:map00100')."""
    out: list[tuple[str, str]] = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) == 2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def main() -> None:
    t0 = time.time()
    print("[A4] fetching KEGG dumps ...", flush=True)

    drug_links = parse_link_tsv(fetch_lines(KEGG_LINK_DRUG_PATHWAY))
    cpd_links = parse_link_tsv(fetch_lines(KEGG_LINK_CPD_PATHWAY))
    map_lines = fetch_lines(KEGG_LIST_PATHWAY_MAP)
    drug_list = fetch_lines(KEGG_LIST_DRUG)

    # Pathway name map (mapNNNNN → display name)
    pathway_name: dict[str, str] = {}
    for ln in map_lines:
        parts = ln.split("\t", 1)
        if len(parts) == 2:
            pid, pname = parts[0].strip(), parts[1].strip()
            # strip 'map' prefix variants; store canonical path:mapNNNNN
            if not pid.startswith("map"):
                pid = pid.replace("path:", "")
            pathway_name[f"path:{pid}"] = pname
    # Drug name map (Dxxxxx → display name)
    kegg_drug_name: dict[str, str] = {}
    for ln in drug_list:
        parts = ln.split("\t", 1)
        if len(parts) == 2:
            did, dname = parts[0].strip(), parts[1].strip()
            kegg_drug_name[did] = dname.split(";")[0]

    # ── Write raw KEGG tables ─────────────────────────────────────────────────
    kegg_drug_paths = [
        {"kegg_drug_id": a.replace("dr:", ""), "pathway_id": b.replace("path:", ""),
         "pathway_name": pathway_name.get(b)}
        for a, b in drug_links
    ]
    kegg_cpd_paths = [
        {"kegg_compound_id": a.replace("cpd:", ""), "pathway_id": b.replace("path:", ""),
         "pathway_name": pathway_name.get(b)}
        for a, b in cpd_links
    ]
    kegg_pathways_meta = [
        {"pathway_id": pid.replace("path:", ""), "pathway_name": name, "source": "kegg"}
        for pid, name in pathway_name.items()
    ]

    pq.write_table(pa.Table.from_pylist(kegg_drug_paths),
                   OUT_DIR / "kegg_drug_pathways.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(kegg_cpd_paths),
                   OUT_DIR / "kegg_compound_pathways.parquet", compression="snappy")
    pq.write_table(pa.Table.from_pylist(kegg_pathways_meta),
                   OUT_DIR / "kegg_pathways.parquet", compression="snappy")

    # ── Join DrugBank xrefs → KEGG Drug / KEGG Compound → pathways ────────────
    xref = pq.read_table(IN_DIR / "drug_xref.parquet").to_pylist()
    drugbank_to_kegg_drug: dict[str, str] = {}
    drugbank_to_kegg_cpd: dict[str, str] = {}
    for row in xref:
        if row["resource"] == "KEGG Drug":
            drugbank_to_kegg_drug[row["drugbank_id"]] = row["identifier"]
        elif row["resource"] == "KEGG Compound":
            drugbank_to_kegg_cpd[row["drugbank_id"]] = row["identifier"]
    print(f"[A4] xref join: {len(drugbank_to_kegg_drug)} DrugBank↔KEGG_Drug, "
          f"{len(drugbank_to_kegg_cpd)} DrugBank↔KEGG_Compound", flush=True)

    # Reverse KEGG drug-id → list of pathway ids
    kegg_drug_to_paths: dict[str, list[str]] = defaultdict(list)
    for r in kegg_drug_paths:
        kegg_drug_to_paths[r["kegg_drug_id"]].append(r["pathway_id"])
    kegg_cpd_to_paths: dict[str, list[str]] = defaultdict(list)
    for r in kegg_cpd_paths:
        kegg_cpd_to_paths[r["kegg_compound_id"]].append(r["pathway_id"])

    # ── Build pathways_unified.parquet ────────────────────────────────────────
    # Start with SMPDB edges from A2
    smpdb = pq.read_table(IN_DIR / "drug_pathways.parquet").to_pylist()
    unified: list[dict] = []
    for row in smpdb:
        unified.append({
            "drugbank_id": row["drugbank_id"],
            "source": "smpdb",
            "pathway_id": row["smpdb_id"],
            "pathway_name": row["pathway_name"],
            "category": row["category"],
        })
    # Add KEGG via KEGG Drug ID
    n_kegg_drug_edges = 0
    for db_id, kd_id in drugbank_to_kegg_drug.items():
        for pid in kegg_drug_to_paths.get(kd_id, []):
            unified.append({
                "drugbank_id": db_id,
                "source": "kegg_drug",
                "pathway_id": pid,
                "pathway_name": pathway_name.get(f"path:{pid}"),
                "category": None,
            })
            n_kegg_drug_edges += 1
    # Add KEGG via KEGG Compound ID
    n_kegg_cpd_edges = 0
    for db_id, kc_id in drugbank_to_kegg_cpd.items():
        for pid in kegg_cpd_to_paths.get(kc_id, []):
            unified.append({
                "drugbank_id": db_id,
                "source": "kegg_compound",
                "pathway_id": pid,
                "pathway_name": pathway_name.get(f"path:{pid}"),
                "category": None,
            })
            n_kegg_cpd_edges += 1

    # Deduplicate on (drugbank_id, source, pathway_id)
    before = len(unified)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict] = []
    for r in unified:
        key = (r["drugbank_id"], r["source"], r["pathway_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    after = len(deduped)
    pq.write_table(pa.Table.from_pylist(deduped),
                   OUT_DIR / "pathways_unified.parquet", compression="snappy")

    # ── Coverage stats ────────────────────────────────────────────────────────
    src_counts = Counter(r["source"] for r in deduped)
    drugs_by_src = {s: len({r["drugbank_id"] for r in deduped if r["source"] == s})
                    for s in src_counts}
    total_drugs_any = len({r["drugbank_id"] for r in deduped})
    total_drugs = 19853

    md = [
        "# A4 — KEGG pathway pull + SMPDB unification report\n",
        f"- Total runtime: **{time.time()-t0:.1f}s**",
        "",
        "## KEGG REST dumps",
        f"- `link/pathway/drug`: **{len(drug_links):,}** drug→pathway edges",
        f"- `link/pathway/compound`: **{len(cpd_links):,}** compound→pathway edges",
        f"- `list/pathway/map`: **{len(pathway_name):,}** pathways with names",
        f"- `list/drug`: **{len(kegg_drug_name):,}** KEGG drug entries",
        "",
        "## DrugBank ↔ KEGG joins (via external_identifiers)",
        f"- DrugBank drugs with KEGG Drug ID: **{len(drugbank_to_kegg_drug):,}** "
        f"({100*len(drugbank_to_kegg_drug)/total_drugs:.1f}%)",
        f"- DrugBank drugs with KEGG Compound ID: **{len(drugbank_to_kegg_cpd):,}** "
        f"({100*len(drugbank_to_kegg_cpd)/total_drugs:.1f}%)",
        "",
        "## Unified pathway table (data_processed/pathways_unified.parquet)",
        f"- Raw edges before dedupe: {before:,}",
        f"- After dedupe: **{after:,}**",
        f"- Edges by source: {dict(src_counts)}",
        f"- Unique drugs per source: {drugs_by_src}",
        f"- **Total unique drugs with ANY pathway link: {total_drugs_any:,} "
        f"({100*total_drugs_any/total_drugs:.2f}%)**",
        "",
        "## Implications for A6 (pathway-signature retrieval)",
        f"- SMPDB alone covered {drugs_by_src.get('smpdb',0):,} drugs "
        f"({100*drugs_by_src.get('smpdb',0)/total_drugs:.1f}%).",
        f"- After merging KEGG Drug + KEGG Compound we reach "
        f"{total_drugs_any:,} drugs ({100*total_drugs_any/total_drugs:.1f}%) — "
        f"a {100*total_drugs_any/max(drugs_by_src.get('smpdb',1),1):.1f}% relative boost.",
        "- A6 will compute Jaccard over the *union* of SMPDB + KEGG pathway IDs per drug. "
        "For the remaining drugs (no pathway match), A6 falls back to protein-overlap "
        "Jaccard (via drug_proteins.parquet, 52% coverage) and then SMILES-FP similarity "
        "(via drugs.parquet SMILES, 74% coverage).",
        "",
        "## Output files (data_processed/)",
        "| File | Rows | Notes |",
        "|---|---:|---|",
        f"| kegg_drug_pathways.parquet | {len(kegg_drug_paths):,} | raw KEGG `dr:` → `path:map` |",
        f"| kegg_compound_pathways.parquet | {len(kegg_cpd_paths):,} | raw KEGG `cpd:` → `path:map` |",
        f"| kegg_pathways.parquet | {len(kegg_pathways_meta):,} | pathway_id → name |",
        f"| pathways_unified.parquet | **{after:,}** | DrugBank drug_id × pathway (SMPDB+KEGG) |",
    ]
    AUDIT_MD.write_text("\n".join(md) + "\n")
    print(f"[A4] wrote {AUDIT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
