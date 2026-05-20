"""A2 (v2) — Complete streaming DrugBank 2026-04 extraction.

Writes (under data_processed/):

  drugs.parquet           # one row per drug (19,853)   — metadata + pharmacology text
                          #                             + calculated_properties (SMILES, InChI,
                          #                               InChIKey, IUPAC, logP, MW, PSA, pKa,
                          #                               H-bond donors/acceptors, rotB, RoF, …)
                          #                             + list cols: food_interactions,
                          #                               affected_organisms
  pairs.parquet           # one row per canonical pair (1,456,772)
  drug_pathways.parquet   # long: drug_id x smpdb pathway
  drug_proteins.parquet   # long: drug_id x protein (target|enzyme|transporter|carrier)
  drug_xref.parquet       # long: drug_id x (resource, identifier) — includes KEGG Drug/Compound,
                          #   ChEBI, PubChem, ChEMBL, UniProt, PharmGKB, RxCUI, …
  drug_reactions.parquet  # long: drug_id x reaction (sequence, left, right, enzyme_uniprots)
  drug_snps.parquet       # long: drug_id x SNP effect / SNP-ADR
  drug_brands.parquet     # long: drug_id x (brand_name, labeller, country, kind=product|intl)

Gates (vs pair construction):
  drugs == 19,853, interactions == 2,913,002, pairs == 1,456,772.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "data_raw" / "drugbank_2026-04.xml"
OUT = ROOT / "data_processed"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT_OUT = ROOT / "outputs" / "audit"
AUDIT_OUT.mkdir(parents=True, exist_ok=True)

NS = "http://www.drugbank.ca"

EXPECTED_DRUGS = 19853
EXPECTED_INTERACTIONS = 2913002
EXPECTED_PAIRS = 1456772


def tag(name: str) -> str:
    return f"{{{NS}}}{name}"


def text_of(elem, path: str) -> str | None:
    t = elem.find(f"{{{NS}}}{path}")
    if t is None:
        return None
    return (t.text or "").strip() or None


def texts_of_list(container_elem, singular: str) -> list[str]:
    if container_elem is None:
        return []
    out = []
    for c in container_elem.findall(tag(singular)):
        if c.text and c.text.strip():
            out.append(c.text.strip())
    return out


# ── calculated_properties flattening ──────────────────────────────────────────
# Canonical columns we want on drugs.parquet (normalised column names)
CALC_PROP_COLS = {
    "SMILES": "smiles",
    "InChI": "inchi",
    "InChIKey": "inchi_key",
    "IUPAC Name": "iupac_name",
    "Traditional IUPAC Name": "iupac_trad",
    "Molecular Formula": "molecular_formula",
    "Molecular Weight": "mw",
    "Monoisotopic Weight": "mw_monoisotopic",
    "logP": "logp",
    "logS": "logs",
    "Water Solubility": "water_solubility_calc",
    "Polar Surface Area (PSA)": "psa",
    "Polarizability": "polarizability",
    "Refractivity": "refractivity",
    "Rotatable Bond Count": "rotatable_bond_count",
    "H Bond Acceptor Count": "hba_count",
    "H Bond Donor Count": "hbd_count",
    "pKa (strongest acidic)": "pka_acidic",
    "pKa (strongest basic)": "pka_basic",
    "Physiological Charge": "physiological_charge",
    "Number of Rings": "n_rings",
    "Bioavailability": "bioavailability",
    "Rule of Five": "rule_of_five",
    "Ghose Filter": "ghose_filter",
    "MDDR-Like Rule": "mddr_like",
}

# Prefer ChemAxon values over ALOGPS when both exist (ChemAxon is curated more consistently).
SOURCE_PRIORITY = {"ChemAxon": 2, "ALOGPS": 1, None: 0}


def parse_calculated_properties(elem) -> dict:
    container = elem.find(tag("calculated-properties"))
    out: dict = {col: None for col in CALC_PROP_COLS.values()}
    if container is None:
        return out
    # Keep the value from the highest-priority source per kind
    sources: dict = {}
    for prop in container.findall(tag("property")):
        kind = (text_of(prop, "kind") or "").strip()
        value = text_of(prop, "value")
        source = text_of(prop, "source") or ""
        col = CALC_PROP_COLS.get(kind)
        if not col or value is None:
            continue
        prio = SOURCE_PRIORITY.get(source, 0)
        cur_prio = sources.get(col, -1)
        if prio >= cur_prio:
            out[col] = value
            sources[col] = prio
    return out


def parse_experimental_properties(elem) -> dict:
    """Capture measured properties where present; keyed by 'exp_' prefix."""
    container = elem.find(tag("experimental-properties"))
    out: dict = {
        "exp_logp": None,
        "exp_water_solubility": None,
        "exp_melting_point": None,
        "exp_pka": None,
        "exp_hydrophobicity": None,
    }
    if container is None:
        return out
    for prop in container.findall(tag("property")):
        kind = (text_of(prop, "kind") or "").strip().lower()
        value = text_of(prop, "value")
        if value is None:
            continue
        if kind in ("logp", "logp/hydrophobicity"):
            out["exp_logp"] = value
        elif kind == "water solubility":
            out["exp_water_solubility"] = value
        elif kind == "melting point":
            out["exp_melting_point"] = value
        elif kind == "pka":
            out["exp_pka"] = value
        elif kind == "hydrophobicity":
            out["exp_hydrophobicity"] = value
    return out


def parse_external_ids(elem, drug_id: str) -> list[dict]:
    container = elem.find(tag("external-identifiers"))
    rows: list[dict] = []
    if container is None:
        return rows
    for e in container.findall(tag("external-identifier")):
        r = text_of(e, "resource")
        i = text_of(e, "identifier")
        if r and i:
            rows.append({"drugbank_id": drug_id, "resource": r, "identifier": i})
    return rows


def parse_reactions(elem, drug_id: str) -> list[dict]:
    container = elem.find(tag("reactions"))
    rows: list[dict] = []
    if container is None:
        return rows
    for rx in container.findall(tag("reaction")):
        seq = text_of(rx, "sequence")
        le = rx.find(tag("left-element"))
        ri = rx.find(tag("right-element"))
        left_db = text_of(le, "drugbank-id") if le is not None else None
        right_db = text_of(ri, "drugbank-id") if ri is not None else None
        left_name = text_of(le, "name") if le is not None else None
        right_name = text_of(ri, "name") if ri is not None else None
        enz_container = rx.find(tag("enzymes"))
        enzyme_ids: list[str] = []
        if enz_container is not None:
            for en in enz_container.findall(tag("enzyme")):
                uid = text_of(en, "uniprot-id")
                if uid:
                    enzyme_ids.append(uid)
        rows.append({
            "drugbank_id": drug_id,
            "sequence": int(seq) if seq and seq.isdigit() else None,
            "left_drugbank_id": left_db,
            "left_name": left_name,
            "right_drugbank_id": right_db,
            "right_name": right_name,
            "enzyme_uniprot_ids": enzyme_ids,
        })
    return rows


def parse_snps(elem, drug_id: str) -> list[dict]:
    rows: list[dict] = []
    for container_name, kind in (("snp-effects", "snp_effect"),
                                  ("snp-adverse-drug-reactions", "snp_adr")):
        container = elem.find(tag(container_name))
        if container is None:
            continue
        for eff in container:
            rows.append({
                "drugbank_id": drug_id,
                "kind": kind,
                "protein_name": text_of(eff, "protein-name"),
                "gene_symbol": text_of(eff, "gene-symbol"),
                "uniprot_id": text_of(eff, "uniprot-id"),
                "rs_id": text_of(eff, "rs-id"),
                "allele": text_of(eff, "allele"),
                "defining_change": text_of(eff, "defining-change"),
                "description": text_of(eff, "description"),
                "pubmed_id": text_of(eff, "pubmed-id"),
                "adverse_reaction": text_of(eff, "adverse-reaction"),
                "severity": text_of(eff, "severity"),
            })
    return rows


def parse_brands(elem, drug_id: str) -> list[dict]:
    rows: list[dict] = []
    prods = elem.find(tag("products"))
    if prods is not None:
        for p in prods.findall(tag("product")):
            rows.append({
                "drugbank_id": drug_id,
                "brand_kind": "product",
                "name": text_of(p, "name"),
                "labeller": text_of(p, "labeller"),
                "country": text_of(p, "country"),
                "dosage_form": text_of(p, "dosage-form"),
                "strength": text_of(p, "strength"),
                "route": text_of(p, "route"),
                "started_marketing_on": text_of(p, "started-marketing-on"),
                "ended_marketing_on": text_of(p, "ended-marketing-on"),
                "approved": text_of(p, "approved"),
                "generic": text_of(p, "generic"),
                "over_the_counter": text_of(p, "over-the-counter"),
                "fda_application_number": text_of(p, "fda-application-number"),
                "source": text_of(p, "source"),
            })
    intls = elem.find(tag("international-brands"))
    if intls is not None:
        for ib in intls.findall(tag("international-brand")):
            rows.append({
                "drugbank_id": drug_id,
                "brand_kind": "international",
                "name": text_of(ib, "name"),
                "labeller": None,
                "country": None,
                "dosage_form": None,
                "strength": None,
                "route": None,
                "started_marketing_on": None,
                "ended_marketing_on": None,
                "approved": None,
                "generic": None,
                "over_the_counter": None,
                "fda_application_number": None,
                "source": None,
            })
    return rows


def parse_drug(drug_elem) -> tuple[dict, list[dict], list[dict], list[dict],
                                    list[dict], list[dict], list[dict], list[dict], int]:
    # primary id
    drug_id = None
    for dbid in drug_elem.findall(tag("drugbank-id")):
        if dbid.get("primary") == "true":
            drug_id = dbid.text
            break
    if drug_id is None:
        first = drug_elem.find(tag("drugbank-id"))
        drug_id = first.text if first is not None else None
    if drug_id is None:
        return None, [], [], [], [], [], [], [], 0

    name = text_of(drug_elem, "name")
    dtype = drug_elem.get("type")
    groups = texts_of_list(drug_elem.find(tag("groups")), "group")
    synonyms = []
    syn = drug_elem.find(tag("synonyms"))
    if syn is not None:
        synonyms = [s.text.strip() for s in syn.findall(tag("synonym")) if s.text]

    atc_codes: list[str] = []
    atc = drug_elem.find(tag("atc-codes"))
    if atc is not None:
        for a in atc.findall(tag("atc-code")):
            code = a.get("code")
            if code:
                atc_codes.append(code)

    categories = []
    cat = drug_elem.find(tag("categories"))
    if cat is not None:
        for c in cat.findall(tag("category")):
            nm = c.find(tag("category"))
            if nm is not None and nm.text:
                categories.append(nm.text.strip())

    classif = drug_elem.find(tag("classification"))
    classification = {}
    if classif is not None:
        for k in ("direct-parent", "kingdom", "superclass", "class", "subclass"):
            v = text_of(classif, k)
            if v:
                classification[k] = v

    food_interactions: list[str] = []
    fi = drug_elem.find(tag("food-interactions"))
    if fi is not None:
        food_interactions = [x.text.strip() for x in fi.findall(tag("food-interaction")) if x.text]

    affected_organisms: list[str] = []
    ao = drug_elem.find(tag("affected-organisms"))
    if ao is not None:
        affected_organisms = [x.text.strip() for x in ao.findall(tag("affected-organism")) if x.text]

    # FASTA sequences (biotech drugs)
    seq_texts: list[str] = []
    seqs = drug_elem.find(tag("sequences"))
    if seqs is not None:
        for s in seqs.findall(tag("sequence")):
            if s.text and s.text.strip():
                seq_texts.append(s.text.strip())

    row = {
        "drugbank_id": drug_id,
        "name": name,
        "type": dtype,
        "groups": groups,
        "atc_codes": atc_codes,
        "cas_number": text_of(drug_elem, "cas-number"),
        "unii": text_of(drug_elem, "unii"),
        "state": text_of(drug_elem, "state"),
        "synonyms": synonyms,
        "categories": categories,
        "classification": json.dumps(classification, ensure_ascii=False) if classification else None,
        "description": text_of(drug_elem, "description"),
        "indication": text_of(drug_elem, "indication"),
        "pharmacodynamics": text_of(drug_elem, "pharmacodynamics"),
        "mechanism_of_action": text_of(drug_elem, "mechanism-of-action"),
        "metabolism": text_of(drug_elem, "metabolism"),
        "absorption": text_of(drug_elem, "absorption"),
        "half_life": text_of(drug_elem, "half-life"),
        "protein_binding": text_of(drug_elem, "protein-binding"),
        "route_of_elimination": text_of(drug_elem, "route-of-elimination"),
        "volume_of_distribution": text_of(drug_elem, "volume-of-distribution"),
        "clearance": text_of(drug_elem, "clearance"),
        "toxicity": text_of(drug_elem, "toxicity"),
        "food_interactions": food_interactions,
        "affected_organisms": affected_organisms,
        "sequences": seq_texts,
    }
    # Merge calculated + experimental properties
    row.update(parse_calculated_properties(drug_elem))
    row.update(parse_experimental_properties(drug_elem))

    # drug-interactions
    interactions: list[dict] = []
    inter_container = drug_elem.find(tag("drug-interactions"))
    n_int = 0
    if inter_container is not None:
        for inter in inter_container.findall(tag("drug-interaction")):
            b_id = text_of(inter, "drugbank-id")
            b_name = text_of(inter, "name")
            desc = text_of(inter, "description")
            if b_id and desc:
                n_int += 1
                interactions.append({
                    "a_id": drug_id, "b_id": b_id,
                    "a_name": name, "b_name": b_name, "description": desc,
                })

    # pathways (SMPDB)
    pathways: list[dict] = []
    pc = drug_elem.find(tag("pathways"))
    if pc is not None:
        for p in pc.findall(tag("pathway")):
            pid = text_of(p, "smpdb-id")
            pname = text_of(p, "name")
            pcat = text_of(p, "category")
            enzymes = p.find(tag("enzymes"))
            enzyme_ids = []
            if enzymes is not None:
                enzyme_ids = [(u.text or "").strip() for u in enzymes if u.text]
            pathways.append({
                "drugbank_id": drug_id, "smpdb_id": pid, "pathway_name": pname,
                "category": pcat, "n_enzymes": len(enzyme_ids),
                "enzyme_uniprot_ids": enzyme_ids,
            })

    # proteins
    proteins: list[dict] = []
    for role in ("targets", "enzymes", "transporters", "carriers"):
        container = drug_elem.find(tag(role))
        if container is None:
            continue
        singular = {"targets": "target", "enzymes": "enzyme",
                    "transporters": "transporter", "carriers": "carrier"}[role]
        for p in container.findall(tag(singular)):
            actions = []
            acts = p.find(tag("actions"))
            if acts is not None:
                actions = [a.text.strip() for a in acts.findall(tag("action")) if a.text]
            poly = p.find(tag("polypeptide"))
            proteins.append({
                "drugbank_id": drug_id,
                "role": role,
                "protein_id": text_of(p, "id"),
                "protein_name": text_of(p, "name"),
                "uniprot": poly.get("id") if poly is not None else None,
                "organism": text_of(p, "organism"),
                "actions": actions,
                "known_action": text_of(p, "known-action"),
                "inhibition_strength": text_of(p, "inhibition-strength"),
                "induction_strength": text_of(p, "induction-strength"),
            })

    xrefs = parse_external_ids(drug_elem, drug_id)
    reactions = parse_reactions(drug_elem, drug_id)
    snps = parse_snps(drug_elem, drug_id)
    brands = parse_brands(drug_elem, drug_id)

    return row, interactions, pathways, proteins, xrefs, reactions, snps, brands, n_int


def write_parquet(rows: list[dict], path: Path, chunk_size: int = 200_000) -> None:
    if not rows:
        pa.table({}).to_parquet(path)
        return
    if len(rows) <= chunk_size:
        tbl = pa.Table.from_pylist(rows)
        pq.write_table(tbl, path, compression="snappy")
        return
    writer = None
    for i in range(0, len(rows), chunk_size):
        tbl = pa.Table.from_pylist(rows[i:i+chunk_size])
        if writer is None:
            writer = pq.ParquetWriter(path, tbl.schema, compression="snappy")
        writer.write_table(tbl)
    if writer:
        writer.close()


def main():
    assert XML.exists(), f"missing {XML}"
    t0 = time.time()
    print(f"[A2v2] parsing {XML} ({XML.stat().st_size/1e9:.2f} GB)", flush=True)

    drugs: list[dict] = []
    pathways: list[dict] = []
    proteins: list[dict] = []
    xrefs: list[dict] = []
    reactions: list[dict] = []
    snps: list[dict] = []
    brands: list[dict] = []
    pair_map: dict = {}
    n_int_total = 0
    n_drops = 0

    context = etree.iterparse(str(XML), events=("end",), tag=tag("drug"))
    n_top = 0
    for _, elem in context:
        parent = elem.getparent()
        if parent is None or not parent.tag.endswith("}drugbank"):
            continue
        row, interactions, dps, dprot, dxref, drxn, dsnp, dbrand, n_int = parse_drug(elem)
        if row is None:
            elem.clear(keep_tail=True)
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue
        drugs.append(row)
        pathways.extend(dps)
        proteins.extend(dprot)
        xrefs.extend(dxref)
        reactions.extend(drxn)
        snps.extend(dsnp)
        brands.extend(dbrand)
        n_int_total += n_int
        for inter in interactions:
            a, b, d = inter["a_id"], inter["b_id"], inter["description"]
            key = (a, b) if a < b else (b, a)
            if key in pair_map:
                if pair_map[key]["description"] != d:
                    n_drops += 1
                pair_map[key]["bidirectional"] = True
            else:
                pair_map[key] = {
                    "a_id": key[0], "b_id": key[1],
                    "a_name": inter["a_name"] if key[0] == a else inter["b_name"],
                    "b_name": inter["b_name"] if key[0] == a else inter["a_name"],
                    "raw_subject_id": a, "description": d, "bidirectional": False,
                }
        n_top += 1
        if n_top % 2000 == 0:
            print(f"  {n_top} drugs | {len(pair_map):,} pairs | xrefs={len(xrefs):,} "
                  f"| reactions={len(reactions):,} | snps={len(snps):,} | brands={len(brands):,} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        elem.clear(keep_tail=True)
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    print(f"[A2v2] parse complete: drugs={n_top} pairs={len(pair_map):,} "
          f"interactions={n_int_total:,} xrefs={len(xrefs):,} reactions={len(reactions):,} "
          f"snps={len(snps):,} brands={len(brands):,} ({time.time()-t0:.0f}s)", flush=True)

    # Gates
    assert n_top == EXPECTED_DRUGS
    assert n_int_total == EXPECTED_INTERACTIONS
    assert len(pair_map) == EXPECTED_PAIRS
    print("[A2v2] gates passed", flush=True)

    tw = time.time()
    print("[A2v2] writing parquet files...", flush=True)

    write_parquet(drugs, OUT / "drugs.parquet")
    write_parquet([
        {"pair_id": f"{a}|{b}", **rec} for (a, b), rec in pair_map.items()
    ], OUT / "pairs.parquet")
    write_parquet(pathways, OUT / "drug_pathways.parquet")
    write_parquet(proteins, OUT / "drug_proteins.parquet")
    write_parquet(xrefs, OUT / "drug_xref.parquet")
    write_parquet(reactions, OUT / "drug_reactions.parquet")
    write_parquet(snps, OUT / "drug_snps.parquet")
    write_parquet(brands, OUT / "drug_brands.parquet")

    elapsed = time.time() - t0
    print(f"[A2v2] write complete in {time.time()-tw:.0f}s. Total {elapsed:.0f}s.")

    # Audit
    from collections import Counter as C
    resource_counts = C(x["resource"] for x in xrefs)
    kegg_drug = sum(1 for x in xrefs if x["resource"] == "KEGG Drug")
    kegg_cmp = sum(1 for x in xrefs if x["resource"] == "KEGG Compound")
    pubchem = sum(1 for x in xrefs if x["resource"] == "PubChem Compound")
    chebi = sum(1 for x in xrefs if x["resource"] == "ChEBI")
    chembl = sum(1 for x in xrefs if x["resource"] == "ChEMBL")
    uniprot_x = sum(1 for x in xrefs if x["resource"].startswith("UniProt") if True)
    smiles_cov = sum(1 for d in drugs if d.get("smiles"))
    inchi_cov = sum(1 for d in drugs if d.get("inchi"))
    inchikey_cov = sum(1 for d in drugs if d.get("inchi_key"))
    mw_cov = sum(1 for d in drugs if d.get("mw"))
    logp_cov = sum(1 for d in drugs if d.get("logp"))
    food_cov = sum(1 for d in drugs if d.get("food_interactions"))
    ao_cov = sum(1 for d in drugs if d.get("affected_organisms"))
    seq_cov = sum(1 for d in drugs if d.get("sequences"))
    drugs_with_rxn = len({r["drugbank_id"] for r in reactions})
    drugs_with_snp = len({s["drugbank_id"] for s in snps})
    drugs_with_brand = len({b["drugbank_id"] for b in brands})

    md = [
        "# A2 (v2) — Complete DrugBank 2026-04 extraction report\n",
        f"- Source: `{XML}` (SHA256 verified in A0)",
        f"- Parse time: **{elapsed:.0f}s**",
        "",
        "## Counts (gates passed against pair construction)",
        f"- Drugs: **{n_top:,}**",
        f"- Interactions: **{n_int_total:,}**",
        f"- Canonical pairs: **{len(pair_map):,}**",
        "",
        "## Chemistry (from calculated_properties)",
        f"- SMILES coverage: **{smiles_cov:,} / {n_top:,}** ({100*smiles_cov/n_top:.1f}%)",
        f"- InChI coverage: **{inchi_cov:,}** ({100*inchi_cov/n_top:.1f}%)",
        f"- InChIKey coverage: **{inchikey_cov:,}** ({100*inchikey_cov/n_top:.1f}%)",
        f"- Molecular Weight coverage: **{mw_cov:,}** ({100*mw_cov/n_top:.1f}%)",
        f"- logP coverage: **{logp_cov:,}** ({100*logp_cov/n_top:.1f}%)",
        "",
        "## Cross-references (from external_identifiers)",
        f"- Total xref rows: **{len(xrefs):,}**",
        f"- KEGG Drug coverage: **{kegg_drug:,} drugs** ({100*kegg_drug/n_top:.1f}%)   ← **used by A4**",
        f"- KEGG Compound coverage: **{kegg_cmp:,}** ({100*kegg_cmp/n_top:.1f}%)",
        f"- PubChem Compound: {pubchem:,}",
        f"- ChEBI: {chebi:,}",
        f"- ChEMBL: {chembl:,}",
        "",
        "## Other content",
        f"- Food-interaction text coverage: {food_cov:,} drugs ({100*food_cov/n_top:.1f}%)",
        f"- Affected-organisms coverage: {ao_cov:,} drugs ({100*ao_cov/n_top:.1f}%)",
        f"- FASTA sequences (biotech): {seq_cov:,} drugs ({100*seq_cov/n_top:.1f}%)",
        f"- Drug reactions table: **{len(reactions):,}** rows · {drugs_with_rxn:,} drugs",
        f"- SNPs + SNP-ADRs table: **{len(snps):,}** rows · {drugs_with_snp:,} drugs",
        f"- Products + international brands: **{len(brands):,}** rows · {drugs_with_brand:,} drugs",
        "",
        "## Output files (`data_processed/`)",
        "| File | Rows | Notes |",
        "|---|---:|---|",
        f"| drugs.parquet | {n_top:,} | + chemistry cols (SMILES/InChI/…) + food/organisms/sequences |",
        f"| pairs.parquet | {len(pair_map):,} | canonical |",
        f"| drug_pathways.parquet | {len(pathways):,} | SMPDB |",
        f"| drug_proteins.parquet | {len(proteins):,} | targets/enzymes/transporters/carriers |",
        f"| drug_xref.parquet | {len(xrefs):,} | KEGG/PubChem/ChEBI/ChEMBL/UniProt/… |",
        f"| drug_reactions.parquet | {len(reactions):,} | metabolic reactions |",
        f"| drug_snps.parquet | {len(snps):,} | pharmacogenomics |",
        f"| drug_brands.parquet | {len(brands):,} | name aliases |",
    ]
    (AUDIT_OUT / "a2_extraction_report.md").write_text("\n".join(md) + "\n")
    print(f"[A2v2] wrote {AUDIT_OUT/'a2_extraction_report.md'}")


if __name__ == "__main__":
    main()
