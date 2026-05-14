"""A7 — Mechanistic Overlap Rate (MOR) validation.

MOR is the P2 sanity gate for pathway/protein/FP retrieval.

For each query pair q=(A,B) sampled from the data:
  1. Retrieve top-k neighbour pairs n=(C,D) using pair-level similarity
     sim(q, n) = max( sim_drug(A,C)+sim_drug(B,D),  sim_drug(A,D)+sim_drug(B,C) )
     where sim_drug = union over tiers:
       α · protein_jaccard(x,y) + β · smiles_tanimoto(x,y) + γ · atc_depth(x,y)/7
  2. MOR@k = fraction of top-k neighbours whose family label
              matches the query pair's family.

A high MOR means the retrieval index surfaces mechanistically-similar pairs,
which is the whole point of P2.

Design choices:
  - We drop the pathway term from sim_drug for MOR because only 13.4% of drugs
    have pathway sets — that would shrink the candidate pool massively. Protein
    + SMILES + ATC are all near-universal.
  - We exclude any candidate pair that *shares a drug* with the query (trivial
    match) and the query itself.
  - We stratify the 2,000 query sample by family so minority families (~0.3%
    Other, 0.9% Absorption) still get measured.

Output:
  outputs/audit/a7_mor_report.md
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"
OUT_MD = ROOT / "outputs" / "audit" / "a7_mor_report.md"

PAIRS_PER_FAMILY = 250     # stratified sample
TOP_K_LIST = [1, 5, 10, 20]
ALPHA_PROTEIN = 1.0
BETA_SMILES = 1.0
GAMMA_ATC = 0.5
RANDOM_SEED = 17


def build_drug_features():
    drugs = pq.read_table(DATA / "drugs.parquet",
                          columns=["drugbank_id", "smiles", "atc_codes"]).to_pylist()
    drug_id_to_idx = {d["drugbank_id"]: i for i, d in enumerate(drugs)}
    # ATC codes (list of str) per drug
    atc_list = [d.get("atc_codes") or [] for d in drugs]
    # Morgan fp per drug
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps: list = [None] * len(drugs)
    for i, d in enumerate(drugs):
        smi = d.get("smiles")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps[i] = gen.GetFingerprint(mol)
    # Protein sets per drug
    prots = pq.read_table(DATA / "drug_proteins.parquet",
                          columns=["drugbank_id", "uniprot"]).to_pylist()
    protein_sets: list[set[str]] = [set() for _ in drugs]
    for r in prots:
        if r["uniprot"] and r["drugbank_id"] in drug_id_to_idx:
            protein_sets[drug_id_to_idx[r["drugbank_id"]]].add(r["uniprot"])
    return drug_id_to_idx, fps, protein_sets, atc_list


def atc_prefix_depth_pair(a_codes: list[str], b_codes: list[str]) -> int:
    best = 0
    for x in a_codes:
        for y in b_codes:
            k = 0
            ml = min(len(x), len(y))
            while k < ml and x[k] == y[k]:
                k += 1
            if k > best:
                best = k
    return best


def main():
    t0 = time.time()
    print("[A7] building drug features ...", flush=True)
    drug_id_to_idx, fps, protein_sets, atc_list = build_drug_features()
    n_drugs = len(drug_id_to_idx)
    fp_list = fps  # same order as drug_id_to_idx enumeration
    print(f"[A7] {n_drugs:,} drugs  | fps: {sum(1 for f in fps if f is not None):,}"
          f" | proteins: {sum(1 for s in protein_sets if s):,}"
          f" | atc: {sum(1 for a in atc_list if a):,}", flush=True)

    # Load pairs + labels
    pairs_tbl = pq.read_table(DATA / "pairs.parquet",
                              columns=["pair_id", "a_id", "b_id"]).to_pylist()
    labels_tbl = pq.read_table(DATA / "labels_hierarchical.parquet",
                               columns=["pair_id", "family"]).to_pylist()
    family_of = {r["pair_id"]: r["family"] for r in labels_tbl}
    # Index pairs: a_idx, b_idx via drug_id_to_idx
    pair_arr_a = np.empty(len(pairs_tbl), dtype=np.int32)
    pair_arr_b = np.empty(len(pairs_tbl), dtype=np.int32)
    pair_fam: list[str] = [None] * len(pairs_tbl)
    pair_id_list: list[str] = [None] * len(pairs_tbl)
    ok = 0
    family_to_pairs: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pairs_tbl):
        a_idx = drug_id_to_idx.get(p["a_id"], -1)
        b_idx = drug_id_to_idx.get(p["b_id"], -1)
        fam = family_of.get(p["pair_id"])
        pair_arr_a[i] = a_idx
        pair_arr_b[i] = b_idx
        pair_fam[i] = fam
        pair_id_list[i] = p["pair_id"]
        if a_idx >= 0 and b_idx >= 0 and fam:
            family_to_pairs[fam].append(i)
            ok += 1
    print(f"[A7] usable pairs (both drugs + family known): {ok:,} / {len(pairs_tbl):,}")

    # Sample queries stratified by family
    random.seed(RANDOM_SEED)
    query_indices: list[int] = []
    for fam, idx_list in family_to_pairs.items():
        k = min(PAIRS_PER_FAMILY, len(idx_list))
        query_indices.extend(random.sample(idx_list, k))
    print(f"[A7] {len(query_indices):,} query pairs sampled "
          f"(stratified, ≤{PAIRS_PER_FAMILY}/family over {len(family_to_pairs)} families)")

    # Per-drug protein-count vector (for Jaccard denominators)
    # Jaccard(A,B) = |A∩B| / |A∪B| = inter / (|A|+|B|-inter)
    # We'll compute inter at query time.

    # Compute MOR via sampled queries. For each query:
    #  - Build sim_drug(x) vec for x∈{query.a, query.b}:
    #      sim_drug(x, y) = α · proteinJ(x,y) + β · tanimoto(x,y) + γ · atc_depth(x,y)/7
    #  - Then pair-sim(q,n) = max(sim(a,c)+sim(b,d), sim(a,d)+sim(b,c))
    #  - Exclude candidate pairs that share a drug with query.

    mor_hits = {k: 0 for k in TOP_K_LIST}
    mor_total = 0
    family_mor: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    family_count: dict[str, int] = defaultdict(int)

    # Precompute ATC-depth matrix is too big (O(n²)); compute on the fly per query.
    # We'll short-list candidates via numpy bulk Tanimoto over all fps.
    fp_none_mask = np.array([f is None for f in fp_list], dtype=bool)
    fp_present = [f for f in fp_list if f is not None]
    fp_present_idx = np.array([i for i, f in enumerate(fp_list) if f is not None], dtype=np.int32)

    t_loop = time.time()
    for qn, qi in enumerate(query_indices):
        a_idx = int(pair_arr_a[qi]); b_idx = int(pair_arr_b[qi])
        q_family = pair_fam[qi]
        # Vectorized Tanimoto using rdkit C bulk
        fp_a = fp_list[a_idx]; fp_b = fp_list[b_idx]
        tani_a = np.zeros(n_drugs, dtype=np.float32)
        tani_b = np.zeros(n_drugs, dtype=np.float32)
        if fp_a is not None:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp_a, fp_present),
                              dtype=np.float32)
            tani_a[fp_present_idx] = sims
        if fp_b is not None:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp_b, fp_present),
                              dtype=np.float32)
            tani_b[fp_present_idx] = sims

        # Protein Jaccard per drug (vs query A and B) — computed on the fly but cheap.
        # |A| = |prot_sets[a_idx]|; |B| similarly. For each drug x:
        #   inter_a = |A ∩ P_x|
        #   jac_a = inter_a / (|A|+|P_x|-inter_a)
        pA = protein_sets[a_idx]; pB = protein_sets[b_idx]
        aA, aB = len(pA), len(pB)
        jac_a = np.zeros(n_drugs, dtype=np.float32)
        jac_b = np.zeros(n_drugs, dtype=np.float32)
        if aA:
            for idx, s in enumerate(protein_sets):
                ls = len(s)
                if not ls:
                    continue
                inter = len(pA & s) if ls < 50 else len(pA.intersection(s))
                denom = aA + ls - inter
                if denom:
                    jac_a[idx] = inter / denom
        if aB:
            for idx, s in enumerate(protein_sets):
                ls = len(s)
                if not ls:
                    continue
                inter = len(pB & s) if ls < 50 else len(pB.intersection(s))
                denom = aB + ls - inter
                if denom:
                    jac_b[idx] = inter / denom

        # ATC depth per drug (vs query A and B)
        atc_q_a = atc_list[a_idx]; atc_q_b = atc_list[b_idx]
        atc_a = np.zeros(n_drugs, dtype=np.float32)
        atc_b = np.zeros(n_drugs, dtype=np.float32)
        if atc_q_a:
            for idx, ac in enumerate(atc_list):
                if ac:
                    atc_a[idx] = atc_prefix_depth_pair(atc_q_a, ac) / 7.0
        if atc_q_b:
            for idx, ac in enumerate(atc_list):
                if ac:
                    atc_b[idx] = atc_prefix_depth_pair(atc_q_b, ac) / 7.0

        sim_to_a = ALPHA_PROTEIN * jac_a + BETA_SMILES * tani_a + GAMMA_ATC * atc_a
        sim_to_b = ALPHA_PROTEIN * jac_b + BETA_SMILES * tani_b + GAMMA_ATC * atc_b

        # Build pair-level scores vectorized
        # cand_sim[i] = max( sim_to_a[pair_a[i]] + sim_to_b[pair_b[i]],
        #                    sim_to_a[pair_b[i]] + sim_to_b[pair_a[i]] )
        s_ac = sim_to_a[pair_arr_a]
        s_bd = sim_to_b[pair_arr_b]
        s_ad = sim_to_a[pair_arr_b]
        s_bc = sim_to_b[pair_arr_a]
        pair_sim = np.maximum(s_ac + s_bd, s_ad + s_bc)

        # Exclude: self-pair; pairs sharing a drug with query
        share_a = (pair_arr_a == a_idx) | (pair_arr_b == a_idx)
        share_b = (pair_arr_a == b_idx) | (pair_arr_b == b_idx)
        exclude = share_a | share_b
        pair_sim[exclude] = -np.inf

        # Top-K
        max_k = max(TOP_K_LIST)
        top_idx = np.argpartition(-pair_sim, max_k)[:max_k]
        top_scores = pair_sim[top_idx]
        order = np.argsort(-top_scores)
        top_idx = top_idx[order]
        top_fams = [pair_fam[i] for i in top_idx]
        # Record hits
        mor_total += 1
        for k in TOP_K_LIST:
            h = sum(1 for f in top_fams[:k] if f == q_family)
            mor_hits[k] += h
        for j, k in enumerate(TOP_K_LIST):
            h = sum(1 for f in top_fams[:k] if f == q_family)
            family_mor[q_family][j] += h
        family_count[q_family] += 1

        if (qn + 1) % 100 == 0:
            elapsed = time.time() - t_loop
            rate = (qn + 1) / elapsed
            eta = (len(query_indices) - qn - 1) / rate
            print(f"  query {qn+1:,}/{len(query_indices):,}   "
                  f"{rate:.1f} q/s   eta {eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[A7] done in {elapsed:.0f}s")

    # Report
    md = ["# A7 — Mechanistic Overlap Rate (MOR) report\n",
          f"- Queries: **{mor_total:,}** (stratified, ≤{PAIRS_PER_FAMILY}/family)",
          f"- Similarity weights: α·protein + β·smiles + γ·atc/7 "
          f"(α={ALPHA_PROTEIN}, β={BETA_SMILES}, γ={GAMMA_ATC})",
          f"- Runtime: {elapsed:.0f}s", "",
          "## Overall MOR@k (macro across queries)", "",
          "| k | Neighbours retrieved | Same-family hits | MOR@k |",
          "|---:|---:|---:|---:|"]
    for k in TOP_K_LIST:
        total = mor_total * k
        md.append(f"| {k} | {total:,} | {mor_hits[k]:,} | "
                  f"**{100*mor_hits[k]/max(total,1):.2f}%** |")
    md.append("")
    md.append("## Per-family MOR@10")
    md.append("")
    md.append("| Family | Queries | MOR@1 | MOR@5 | MOR@10 | MOR@20 |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for fam in sorted(family_count):
        n = family_count[fam]
        if n == 0:
            continue
        cells = []
        for j, k in enumerate(TOP_K_LIST):
            denom = n * k
            cells.append(f"{100*family_mor[fam][j]/max(denom,1):.1f}%")
        md.append(f"| `{fam}` | {n} | " + " | ".join(cells) + " |")

    # Sanity gate: MOR@10 ≥ 50% (random would be ~1/8=12.5% with 8 families)
    overall_mor10 = mor_hits[10] / max(mor_total * 10, 1) * 100
    random_baseline = 100 / 8  # 8 families
    md.append("")
    md.append("## Gates")
    md.append(f"- Random baseline (8 equally-weighted families): **{random_baseline:.1f}%**")
    md.append(f"- MOR@10 ≥ 50%: **{'PASS' if overall_mor10 >= 50 else 'SOFT FAIL'}** "
              f"(actual {overall_mor10:.2f}%)")
    md.append(f"- MOR@10 ≥ random + 15pts: "
              f"**{'PASS' if overall_mor10 >= random_baseline + 15 else 'FAIL'}** "
              f"(need ≥ {random_baseline + 15:.1f}%)")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"[A7] wrote {OUT_MD.relative_to(ROOT)}")
    print(f"[A7] MOR@1={100*mor_hits[1]/max(mor_total,1):.2f}%  "
          f"MOR@10={overall_mor10:.2f}%")


if __name__ == "__main__":
    main()
