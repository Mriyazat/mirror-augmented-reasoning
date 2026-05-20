"""A6 — Pair signatures (retrieval keys) via a cascade of four similarity views.

For every pair in pairs.parquet we compute four similarities between drug_A
and drug_B and record which view was informative (= the retrieval tier that
will supply neighbours in B1 RAG).

  tier-1  pathway_jaccard     — Jaccard over SMPDB + KEGG pathway IDs
  tier-2  protein_jaccard     — Jaccard over targets/enzymes/transporters/carriers
  tier-3  smiles_tanimoto     — Morgan fingerprint (radius 2, 2048 bits)
  tier-4  atc_prefix_depth    — max ATC-code prefix length across A×B

Output:
  data_processed/pair_signatures.parquet
Audit:
  outputs/audit/a6_signatures_report.md
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")  # silence rdkit parse warnings

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"

PAIRS = DATA / "pairs.parquet"
DRUGS = DATA / "drugs.parquet"
PATHS = DATA / "pathways_unified.parquet"
PROTS = DATA / "drug_proteins.parquet"
OUT = DATA / "pair_signatures.parquet"
AUDIT = ROOT / "outputs" / "audit" / "a6_signatures_report.md"


# ── precompute per-drug feature sets ─────────────────────────────────────────
def build_pathway_sets() -> dict[str, set[str]]:
    t = pq.read_table(PATHS, columns=["drugbank_id", "pathway_id"]).to_pylist()
    out: dict[str, set[str]] = defaultdict(set)
    for r in t:
        if r["pathway_id"]:
            out[r["drugbank_id"]].add(r["pathway_id"])
    return dict(out)


def build_protein_sets() -> dict[str, set[str]]:
    t = pq.read_table(PROTS, columns=["drugbank_id", "uniprot"]).to_pylist()
    out: dict[str, set[str]] = defaultdict(set)
    for r in t:
        if r["uniprot"]:
            out[r["drugbank_id"]].add(r["uniprot"])
    return dict(out)


def build_smiles_fps(drugs_rows) -> tuple[dict[str, object], int]:
    """Returns (drugbank_id → Morgan fp) + count parsed."""
    n_ok = 0
    out: dict[str, object] = {}
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    for d in drugs_rows:
        smi = d.get("smiles")
        if not smi:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        out[d["drugbank_id"]] = fp
        n_ok += 1
    return out, n_ok


def build_atc_sets(drugs_rows) -> dict[str, list[str]]:
    return {d["drugbank_id"]: [a for a in (d.get("atc_codes") or []) if a]
            for d in drugs_rows}


# ── signature computation ────────────────────────────────────────────────────
def jaccard(a: set, b: set) -> tuple[float | None, int]:
    if not a or not b:
        return None, 0
    inter = len(a & b)
    union = len(a | b)
    return (inter / union if union else None, inter)


def atc_prefix_depth(atc_a: list[str], atc_b: list[str]) -> int:
    """Max length of common prefix across any pair (max 7 chars — full ATC)."""
    best = 0
    if not atc_a or not atc_b:
        return 0
    for x in atc_a:
        for y in atc_b:
            k = 0
            ml = min(len(x), len(y))
            while k < ml and x[k] == y[k]:
                k += 1
            if k > best:
                best = k
    return best


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("[A6] loading inputs ...", flush=True)
    drugs = pq.read_table(DRUGS, columns=["drugbank_id", "smiles", "atc_codes"]).to_pylist()

    pathway_sets = build_pathway_sets()
    protein_sets = build_protein_sets()
    print(f"[A6] pathway sets: {len(pathway_sets):,} drugs | "
          f"protein sets: {len(protein_sets):,} drugs", flush=True)

    print("[A6] computing Morgan fingerprints ...", flush=True)
    fps, n_fp = build_smiles_fps(drugs)
    print(f"[A6] fps: {n_fp:,}/{len(drugs):,} drugs ({100*n_fp/len(drugs):.1f}%)", flush=True)

    atc_sets = build_atc_sets(drugs)

    pairs = pq.read_table(PAIRS, columns=["pair_id", "a_id", "b_id"]).to_pylist()
    n_pairs = len(pairs)
    print(f"[A6] pairs to score: {n_pairs:,}", flush=True)

    rows: list[dict] = []
    pathway_cov = 0
    protein_cov = 0
    tanimoto_cov = 0
    atc_cov = 0

    for i, p in enumerate(pairs):
        a, b = p["a_id"], p["b_id"]
        p_set_a = pathway_sets.get(a, set())
        p_set_b = pathway_sets.get(b, set())
        pr_set_a = protein_sets.get(a, set())
        pr_set_b = protein_sets.get(b, set())

        pw_jac, pw_int = jaccard(p_set_a, p_set_b)
        pr_jac, pr_int = jaccard(pr_set_a, pr_set_b)

        tani = None
        fp_a = fps.get(a)
        fp_b = fps.get(b)
        if fp_a is not None and fp_b is not None:
            tani = float(DataStructs.TanimotoSimilarity(fp_a, fp_b))

        atc_depth = atc_prefix_depth(atc_sets.get(a, []), atc_sets.get(b, []))

        rows.append({
            "pair_id": p["pair_id"],
            "a_id": a, "b_id": b,
            "pathway_jaccard": pw_jac,
            "pathway_shared": pw_int,
            "n_pathways_a": len(p_set_a),
            "n_pathways_b": len(p_set_b),
            "protein_jaccard": pr_jac,
            "protein_shared": pr_int,
            "n_proteins_a": len(pr_set_a),
            "n_proteins_b": len(pr_set_b),
            "smiles_tanimoto": tani,
            "atc_prefix_depth": atc_depth,
        })
        if pw_jac is not None:
            pathway_cov += 1
        if pr_jac is not None:
            protein_cov += 1
        if tani is not None:
            tanimoto_cov += 1
        if atc_depth > 0:
            atc_cov += 1

        if (i + 1) % 200_000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1:,}/{n_pairs:,}  ({elapsed:.0f}s, "
                  f"pathway_cov={100*pathway_cov/(i+1):.1f}%, "
                  f"protein_cov={100*protein_cov/(i+1):.1f}%)", flush=True)

    pq.write_table(pa.Table.from_pylist(rows), OUT, compression="snappy")
    elapsed = time.time() - t0
    print(f"[A6] wrote {OUT.relative_to(ROOT)} ({len(rows):,} rows) in {elapsed:.0f}s", flush=True)

    # ── audit ─────────────────────────────────────────────────────────────────
    def pct(x): return 100 * x / n_pairs
    any_signal = sum(1 for r in rows if (r["pathway_jaccard"] is not None
                                          or r["protein_jaccard"] is not None
                                          or r["smiles_tanimoto"] is not None
                                          or r["atc_prefix_depth"] > 0))
    # Tier distribution: which is the BEST tier (pathway > protein > smiles > atc)
    tier_hist = Counter()
    for r in rows:
        if r["pathway_jaccard"] is not None:
            tier_hist["pathway"] += 1
        elif r["protein_jaccard"] is not None:
            tier_hist["protein"] += 1
        elif r["smiles_tanimoto"] is not None:
            tier_hist["smiles"] += 1
        elif r["atc_prefix_depth"] > 0:
            tier_hist["atc"] += 1
        else:
            tier_hist["none"] += 1

    # Signal-strength bins
    def bin_counts(vals: list[float | None], thresholds=(0.0, 0.1, 0.3, 0.5, 0.7)):
        bins = [0] * (len(thresholds) + 1)  # one trailing for 'None'
        for v in vals:
            if v is None:
                bins[-1] += 1; continue
            placed = False
            for i, th in enumerate(thresholds):
                if v <= th:
                    bins[i] += 1
                    placed = True
                    break
            if not placed:
                bins[-2] += 1
        return bins

    pw_bins = bin_counts([r["pathway_jaccard"] for r in rows])
    pr_bins = bin_counts([r["protein_jaccard"] for r in rows])
    tn_bins = bin_counts([r["smiles_tanimoto"] for r in rows])

    md = [
        "# A6 — Pair signatures report",
        "",
        f"- Pairs scored: **{n_pairs:,}**",
        f"- Runtime: {elapsed:.0f}s",
        "",
        "## Coverage (fraction of pairs with a non-null similarity)",
        "",
        "| Tier | Pairs with signal | % |",
        "|---|---:|---:|",
        f"| pathway (Jaccard over SMPDB+KEGG) | {pathway_cov:,} | {pct(pathway_cov):.2f}% |",
        f"| protein (Jaccard over targets/enzymes/transporters/carriers) | "
        f"{protein_cov:,} | {pct(protein_cov):.2f}% |",
        f"| smiles (Morgan FP Tanimoto) | {tanimoto_cov:,} | {pct(tanimoto_cov):.2f}% |",
        f"| atc (non-zero shared prefix depth) | {atc_cov:,} | {pct(atc_cov):.2f}% |",
        f"| **any tier (retrieval coverage)** | **{any_signal:,}** | **{pct(any_signal):.2f}%** |",
        "",
        "## First-available tier per pair (cascade ordering)",
        "",
        "| First available tier | Pairs | % |",
        "|---|---:|---:|",
    ]
    for k in ("pathway", "protein", "smiles", "atc", "none"):
        md.append(f"| {k} | {tier_hist.get(k, 0):,} | {100*tier_hist.get(k, 0)/n_pairs:.2f}% |")

    md += [
        "",
        "## Jaccard distributions",
        "",
        "Bins: `(=0, ≤0.1, ≤0.3, ≤0.5, ≤0.7, >0.7, null)`",
        "",
        f"- pathway_jaccard: {pw_bins}",
        f"- protein_jaccard: {pr_bins}",
        f"- smiles_tanimoto: {tn_bins}",
        "",
        "## Output file",
        f"- `data_processed/pair_signatures.parquet` ({n_pairs:,} rows, 12 cols)",
    ]
    AUDIT.write_text("\n".join(md) + "\n")
    print(f"[A6] wrote {AUDIT.relative_to(ROOT)}")
    print(f"[A6] tier hist: {dict(tier_hist)}")
    print(f"[A6] any-signal coverage: {pct(any_signal):.2f}%")


if __name__ == "__main__":
    main()
