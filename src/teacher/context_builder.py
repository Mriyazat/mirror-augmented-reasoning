"""B1a — Retrieval context builder.

For each DDI pair (a, b), emits a `ContextBundle` containing:

  Direct context (needed for evidence grounding):
    - drug A / drug B:  name, atc_codes, mechanism_of_action (truncated),
                         pk_flags that are True (CYP/P-gp/OATP/BCRP induces
                         /inhibits/substrate)
    - shared_pathways:  list of (pathway_id, pathway_name, category, source)
    - shared_proteins:  list of (uniprot, protein_name, role, actions)
    - similarity scalars: pathway_jaccard, protein_jaccard, smiles_tanimoto,
                          atc_prefix_depth

  Neighbor context (for analogical reasoning — P2 retrieval novelty):
    - top_k mechanistic-neighbor pairs, each with pair_id, drug names, family,
      subtype, direction, polarity, and the composite similarity score

  Metadata:
    - context_ids:  union of every ID a grounded claim could legally cite
                    (pathway IDs, uniprot IDs, DrugBank IDs of A/B and
                    neighbors, ATC codes).  QC uses this as the gold set
                    for the `evidence_grounded` dimension.

Usage:
    from src.teacher.context_builder import ContextBuilder
    cb = ContextBuilder()          # loads everything once
    ctx = cb.build("DB00497|DB01234")

Offline precomputation:
    python -m src.teacher.context_builder --precompute_neighbors subset25k
    → writes data_processed/neighbor_index_subset25k.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_processed"

# How many top-K neighbors to surface in each context
TOP_K_NEIGHBORS = 5

# Text truncation (chars) — keeps prompt length predictable.
# p95 of DrugBank MoA = 2089 chars; 800 gives us most first-paragraph info
# (CYP targets, PK role) without blowing the 70B context budget.
MAX_MOA_CHARS = 800
MAX_DESC_CHARS = 400

# Limits on how much of the "shared" context to surface
MAX_SHARED_PATHWAYS = 8
MAX_SHARED_PROTEINS = 8
# Per-drug pathway / protein listings (99% of subset25k pairs have 0 *shared*
# pathways → we must give the teacher per-drug evidence so it has any
# pathway IDs to cite at all).
MAX_DRUG_PATHWAYS = 6
MAX_DRUG_PROTEINS = 6


# ──────────────────────────────────────────────────────────────────────────
# Context bundle
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class DrugFact:
    drugbank_id: str
    name: str
    atc_codes: list[str]
    moa: str | None
    active_pk_flags: list[str]       # e.g. ["cyp3a4_inh", "p_gp_sub"]
    has_smiles: bool
    smiles: str | None
    mw: float | None
    half_life_hours: float | None


@dataclass
class SharedPathway:
    pathway_id: str
    pathway_name: str
    category: str
    source: str                       # "SMPDB" | "KEGG"


@dataclass
class DrugPathway:
    """Single-drug pathway (for cases with no cross-drug overlap)."""
    pathway_id: str
    pathway_name: str
    source: str


@dataclass
class DrugProtein:
    """Single-drug protein (for cases with no cross-drug overlap)."""
    uniprot: str
    protein_name: str
    role: str                          # target | enzyme | transporter | carrier
    actions: list[str]


@dataclass
class SharedProtein:
    uniprot: str
    protein_name: str
    a_role: str                        # target | enzyme | transporter | carrier
    b_role: str
    a_actions: list[str]               # e.g. ["inhibitor"]
    b_actions: list[str]


@dataclass
class NeighborPair:
    pair_id: str
    a_name: str
    b_name: str
    family: str
    subtype: str
    direction_tag: str                 # a_to_b | b_to_a | bidirectional
    polarity: str | None
    similarity: float                  # composite score (higher = more similar)


@dataclass
class ContextBundle:
    pair_id: str
    a: DrugFact
    b: DrugFact
    shared_pathways: list[SharedPathway]
    shared_proteins: list[SharedProtein]
    a_pathways: list[DrugPathway]        # per-drug fallbacks (cover the 99%
    b_pathways: list[DrugPathway]        # of pairs with no shared pathways)
    a_proteins: list[DrugProtein]
    b_proteins: list[DrugProtein]
    pathway_jaccard: float
    protein_jaccard: float
    smiles_tanimoto: float
    atc_prefix_depth: int
    n_pathways_a: int
    n_pathways_b: int
    n_proteins_a: int
    n_proteins_b: int
    neighbors: list[NeighborPair] = field(default_factory=list)

    def context_ids(self) -> set[str]:
        """All IDs a teacher claim may legally cite.

        Includes everything rendered in the prompt's evidence pool: drugs,
        ATCs, pathways, proteins, pk-flags, neighbor pair_ids, pair-similarity
        metric names, and (crucially) the DrugBank literature/article refs
        embedded inline in each drug's `moa` text (patterns: A<digits>,
        L<digits>, F<digits>, T<digits>).  Without these, claims that cite
        valid MoA-sourced references are wrongly flagged as hallucinations.
        """
        ids: set[str] = {self.a.drugbank_id, self.b.drugbank_id}
        ids.update(self.a.atc_codes)
        ids.update(self.b.atc_codes)
        for p in self.shared_pathways:
            ids.add(p.pathway_id)
        for p in self.shared_proteins:
            ids.add(p.uniprot)
        for p in self.a_pathways:
            ids.add(p.pathway_id)
        for p in self.b_pathways:
            ids.add(p.pathway_id)
        for p in self.a_proteins:
            ids.add(p.uniprot)
        for p in self.b_proteins:
            ids.add(p.uniprot)
        for n in self.neighbors:
            ids.add(n.pair_id)
        for f in self.a.active_pk_flags:
            ids.add(f)
        for f in self.b.active_pk_flags:
            ids.add(f)
        # Similarity metric names — shown as `pathway_jaccard = 0.000` in the
        # prompt; the LLM reasonably cites the metric name as evidence.
        ids.update({"pathway_jaccard", "protein_jaccard",
                    "smiles_tanimoto", "atc_prefix_depth"})
        # DrugBank article / literature refs embedded in moa text.
        # These appear in the prompt verbatim (the moa narrative), so
        # claims that cite them are grounded, not hallucinated.
        import re as _re
        _REF_RX = _re.compile(r"\b([ALFTa][0-9]{2,6})\b")
        for text in (self.a.moa or "", self.b.moa or ""):
            for m in _REF_RX.finditer(text):
                ids.add(m.group(1))
        ids.discard("")
        ids.discard(None)
        return ids

    def evidence_density(self) -> dict[str, int]:
        """How many evidence pools are non-empty?  Used by the prompt to
        decide whether the teacher should abstain."""
        return {
            "moa_a": int(bool(self.a.moa)),
            "moa_b": int(bool(self.b.moa)),
            "pk_flags_a": int(bool(self.a.active_pk_flags)),
            "pk_flags_b": int(bool(self.b.active_pk_flags)),
            "shared_pathways": len(self.shared_pathways),
            "shared_proteins": len(self.shared_proteins),
            "a_pathways": len(self.a_pathways),
            "b_pathways": len(self.b_pathways),
            "a_proteins": len(self.a_proteins),
            "b_proteins": len(self.b_proteins),
            "neighbors": len(self.neighbors),
        }

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────
# Context builder  (one-time load; fast per-pair access)
# ──────────────────────────────────────────────────────────────────────────
class ContextBuilder:
    def __init__(self, neighbor_index_path: Path | None = None):
        print("[ctx] loading drugs ...", flush=True)
        self._load_drugs()
        print("[ctx] loading pathways ...", flush=True)
        self._load_pathways()
        print("[ctx] loading proteins ...", flush=True)
        self._load_proteins()
        print("[ctx] loading pk features ...", flush=True)
        self._load_pk()
        print("[ctx] loading pair signatures + labels ...", flush=True)
        self._load_pair_signatures()
        self._load_labels()

        self._neighbors: dict[str, list[NeighborPair]] | None = None
        if neighbor_index_path is not None and neighbor_index_path.exists():
            print(f"[ctx] loading neighbor index from {neighbor_index_path.name} ...", flush=True)
            self._load_neighbor_index(neighbor_index_path)
        else:
            print("[ctx] no precomputed neighbor index — neighbors will be [] "
                  "(run --precompute_neighbors to build one)", flush=True)
        print(f"[ctx] ready.  {len(self.drugs):,} drugs, "
              f"{len(self.pair_sig):,} pair signatures, "
              f"{len(self.labels):,} labeled pairs", flush=True)

    # -------- loaders --------
    def _load_drugs(self):
        cols = ["drugbank_id", "name", "atc_codes", "mechanism_of_action",
                "smiles", "mw", "synonyms"]
        tbl = pq.read_table(DATA / "drugs.parquet", columns=cols).to_pylist()
        self.drugs: dict[str, dict] = {d["drugbank_id"]: d for d in tbl}

    def _load_pathways(self):
        # drug → list of (pathway_id, pathway_name, category, source)
        tbl = pq.read_table(DATA / "pathways_unified.parquet").to_pylist()
        d2p: dict[str, list[tuple]] = defaultdict(list)
        pathway_meta: dict[str, tuple[str, str, str]] = {}  # pid → (name, cat, source)
        for r in tbl:
            pid = r["pathway_id"]
            d2p[r["drugbank_id"]].append(pid)
            pathway_meta[pid] = (r.get("pathway_name") or pid,
                                 r.get("category") or "unknown",
                                 r.get("source") or "?")
        self.drug_pathways: dict[str, set[str]] = {k: set(v) for k, v in d2p.items()}
        self.pathway_meta = pathway_meta

    def _load_proteins(self):
        tbl = pq.read_table(DATA / "drug_proteins.parquet").to_pylist()
        # drug → list of (uniprot, role, protein_name, actions)
        d2prot: dict[str, list[dict]] = defaultdict(list)
        for r in tbl:
            uni = r.get("uniprot")
            if not uni:
                continue
            actions = r.get("actions") or []
            if isinstance(actions, str):
                actions = [actions]
            d2prot[r["drugbank_id"]].append({
                "uniprot": uni,
                "role": r.get("role") or "unknown",
                "protein_name": r.get("protein_name") or uni,
                "actions": list(actions),
            })
        self.drug_proteins: dict[str, list[dict]] = dict(d2prot)

    def _load_pk(self):
        tbl = pq.read_table(DATA / "pk_features.parquet").to_pylist()
        self.drug_pk: dict[str, dict] = {r["drugbank_id"]: r for r in tbl}
        # all bool flag columns
        sample = tbl[0] if tbl else {}
        self.pk_flag_cols = [c for c in sample
                             if c.endswith(("_inh", "_ind", "_sub"))]

    def _load_pair_signatures(self):
        tbl = pq.read_table(DATA / "pair_signatures.parquet").to_pylist()
        self.pair_sig: dict[str, dict] = {r["pair_id"]: r for r in tbl}

    def _load_labels(self):
        tbl = pq.read_table(DATA / "labels_hierarchical.parquet").to_pylist()
        self.labels: dict[str, dict] = {r["pair_id"]: r for r in tbl}

    def _load_neighbor_index(self, path: Path):
        tbl = pq.read_table(path).to_pylist()
        by_pair: dict[str, list[NeighborPair]] = defaultdict(list)
        for r in tbl:
            by_pair[r["query_pair_id"]].append(NeighborPair(
                pair_id=r["neighbor_pair_id"],
                a_name=r["a_name"],
                b_name=r["b_name"],
                family=r["family"],
                subtype=r["subtype"],
                direction_tag=r["direction_tag"],
                polarity=r.get("polarity"),
                similarity=float(r["similarity"]),
            ))
        # Keep top-K only (already sorted by precompute step)
        self._neighbors = dict(by_pair)

    # -------- per-pair construction --------
    def _drug_fact(self, db_id: str) -> DrugFact:
        d = self.drugs.get(db_id, {})
        name = d.get("name") or db_id
        atc = d.get("atc_codes") or []
        moa = d.get("mechanism_of_action")
        if moa and len(moa) > MAX_MOA_CHARS:
            moa = moa[:MAX_MOA_CHARS].rstrip() + " …"
        # PK active flags
        pk = self.drug_pk.get(db_id, {})
        active = [col for col in self.pk_flag_cols if pk.get(col)]
        return DrugFact(
            drugbank_id=db_id,
            name=name,
            atc_codes=list(atc) if atc is not None else [],
            moa=moa,
            active_pk_flags=active,
            has_smiles=bool(d.get("smiles")),
            smiles=d.get("smiles"),
            mw=float(d["mw"]) if d.get("mw") else None,
            half_life_hours=float(pk["half_life_hours"])
                if pk.get("has_half_life_value") else None,
        )

    def _shared_pathways(self, a: str, b: str) -> list[SharedPathway]:
        pa_set = self.drug_pathways.get(a, set())
        pb_set = self.drug_pathways.get(b, set())
        shared = list(pa_set & pb_set)
        # Sort by (source, name) for determinism
        shared.sort(key=lambda pid: (self.pathway_meta.get(pid, ("", "", ""))[2],
                                     self.pathway_meta.get(pid, ("", "", ""))[0]))
        out = []
        for pid in shared[:MAX_SHARED_PATHWAYS]:
            name, cat, source = self.pathway_meta.get(pid, (pid, "unknown", "?"))
            out.append(SharedPathway(pathway_id=pid, pathway_name=name,
                                     category=cat, source=source))
        return out

    def _drug_pathways_top(self, db_id: str,
                           exclude: set[str]) -> list[DrugPathway]:
        """Top-N pathway IDs for a single drug (excluding any already in the
        shared list)."""
        pids = [p for p in self.drug_pathways.get(db_id, set())
                if p not in exclude]
        # Deterministic ordering: by (source, name) — favors SMPDB-first.
        pids.sort(key=lambda pid: (self.pathway_meta.get(pid, ("", "", ""))[2],
                                   self.pathway_meta.get(pid, ("", "", ""))[0]))
        out: list[DrugPathway] = []
        for pid in pids[:MAX_DRUG_PATHWAYS]:
            name, _cat, source = self.pathway_meta.get(pid, (pid, "unknown", "?"))
            out.append(DrugPathway(pathway_id=pid, pathway_name=name, source=source))
        return out

    def _drug_proteins_top(self, db_id: str,
                           exclude: set[str]) -> list[DrugProtein]:
        """Top-N proteins for a single drug (excluding any already in the
        shared list), prioritized target→enzyme→transporter→carrier."""
        role_priority = {"target": 0, "enzyme": 1, "transporter": 2, "carrier": 3}
        prots = [p for p in self.drug_proteins.get(db_id, [])
                 if p["uniprot"] not in exclude]
        prots.sort(key=lambda p: (role_priority.get(p["role"], 9),
                                  p["protein_name"] or p["uniprot"]))
        out: list[DrugProtein] = []
        for p in prots[:MAX_DRUG_PROTEINS]:
            out.append(DrugProtein(
                uniprot=p["uniprot"],
                protein_name=p["protein_name"] or p["uniprot"],
                role=p["role"],
                actions=list(p["actions"] or []),
            ))
        return out

    def _shared_proteins(self, a: str, b: str) -> list[SharedProtein]:
        a_prots = {p["uniprot"]: p for p in self.drug_proteins.get(a, [])}
        b_prots = {p["uniprot"]: p for p in self.drug_proteins.get(b, [])}
        shared_uni = list(set(a_prots) & set(b_prots))
        # Sort to prefer target/enzyme roles first (most mechanistically relevant)
        role_priority = {"target": 0, "enzyme": 1, "transporter": 2, "carrier": 3}
        shared_uni.sort(key=lambda u: (
            role_priority.get(a_prots[u]["role"], 9),
            role_priority.get(b_prots[u]["role"], 9),
            a_prots[u]["protein_name"],
        ))
        out = []
        for uni in shared_uni[:MAX_SHARED_PROTEINS]:
            ap, bp = a_prots[uni], b_prots[uni]
            out.append(SharedProtein(
                uniprot=uni,
                protein_name=ap["protein_name"] or bp["protein_name"],
                a_role=ap["role"],
                b_role=bp["role"],
                a_actions=ap["actions"],
                b_actions=bp["actions"],
            ))
        return out

    def build(self, pair_id: str) -> ContextBundle:
        sig = self.pair_sig.get(pair_id)
        if sig is None:
            # Signature missing — try to reconstruct a_id, b_id from pair_id
            parts = pair_id.split("|")
            if len(parts) != 2:
                raise KeyError(f"pair_id {pair_id!r} not in pair_signatures and unparseable")
            a_id, b_id = parts
            sig = {"a_id": a_id, "b_id": b_id,
                   "pathway_jaccard": 0, "protein_jaccard": 0,
                   "smiles_tanimoto": 0, "atc_prefix_depth": 0,
                   "n_pathways_a": 0, "n_pathways_b": 0,
                   "n_proteins_a": 0, "n_proteins_b": 0}

        a_id, b_id = sig["a_id"], sig["b_id"]
        shared_pw = self._shared_pathways(a_id, b_id)
        shared_pr = self._shared_proteins(a_id, b_id)
        shared_pw_ids = {p.pathway_id for p in shared_pw}
        shared_pr_ids = {p.uniprot for p in shared_pr}
        bundle = ContextBundle(
            pair_id=pair_id,
            a=self._drug_fact(a_id),
            b=self._drug_fact(b_id),
            shared_pathways=shared_pw,
            shared_proteins=shared_pr,
            a_pathways=self._drug_pathways_top(a_id, shared_pw_ids),
            b_pathways=self._drug_pathways_top(b_id, shared_pw_ids),
            a_proteins=self._drug_proteins_top(a_id, shared_pr_ids),
            b_proteins=self._drug_proteins_top(b_id, shared_pr_ids),
            pathway_jaccard=float(sig.get("pathway_jaccard") or 0),
            protein_jaccard=float(sig.get("protein_jaccard") or 0),
            smiles_tanimoto=float(sig.get("smiles_tanimoto") or 0),
            atc_prefix_depth=int(sig.get("atc_prefix_depth") or 0),
            n_pathways_a=int(sig.get("n_pathways_a") or 0),
            n_pathways_b=int(sig.get("n_pathways_b") or 0),
            n_proteins_a=int(sig.get("n_proteins_a") or 0),
            n_proteins_b=int(sig.get("n_proteins_b") or 0),
        )
        if self._neighbors is not None:
            bundle.neighbors = self._neighbors.get(pair_id, [])
        return bundle


# ──────────────────────────────────────────────────────────────────────────
# Neighbor precomputation  (offline job)
#
# Approach (avoids O(n²) full scoring):
#   1. Compute a drug × drug similarity matrix for the pair-participating
#      drug set  (~4.6k drugs, dense ~85 MB).
#   2. For each drug, keep a top-M most-similar-drug neighbor list.
#   3. For each query pair (qa, qb):
#        candidate drug-pairs = { (x, y) : x ∈ top-M(qa)∪top-M(qb),
#                                         y ∈ top-M(qa)∪top-M(qb) }
#        → look up each candidate (x,y) in our labeled-pair index, score it,
#          and keep top-K.
#   4. Writes data_processed/neighbor_index_<split>.parquet.
#
# Leakage: neighbor candidates are restricted to `allowed_train_pairs` (pairs
# that appear in the train split of `random_full`) so nothing leaks from
# cold-test drugs into the teacher context.
# ──────────────────────────────────────────────────────────────────────────
DRUG_TOPM = 50                         # per-drug top-M similar drugs
# Weights for the composite drug-drug similarity score (sum = 1 is not required)
W_PATHWAY = 1.0
W_PROTEIN = 1.0
W_ATC = 0.5
W_SMILES = 1.0


def _drug_sim_matrix(cb: ContextBuilder, drug_ids: list[str]) -> tuple[np.ndarray, dict]:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    n = len(drug_ids)
    id_to_idx = {d: i for i, d in enumerate(drug_ids)}

    ps = [cb.drug_pathways.get(d, set()) for d in drug_ids]
    pr = [{p["uniprot"] for p in cb.drug_proteins.get(d, [])} for d in drug_ids]
    atc = [(cb.drugs.get(d, {}).get("atc_codes") or []) for d in drug_ids]

    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for d in drug_ids:
        smi = cb.drugs.get(d, {}).get("smiles")
        if smi:
            mol = Chem.MolFromSmiles(smi)
            fps.append(gen.GetFingerprint(mol) if mol else None)
        else:
            fps.append(None)

    def _atc_depth(ac_a, ac_b) -> int:
        best = 0
        for ca in ac_a:
            for cb_code in ac_b:
                m = 0
                for k in range(min(len(ca), len(cb_code))):
                    if ca[k] == cb_code[k]:
                        m += 1
                    else:
                        break
                if m > best:
                    best = m
        return best

    sim = np.zeros((n, n), dtype=np.float32)

    fp_present_idx = np.array([i for i, f in enumerate(fps) if f is not None],
                              dtype=np.int32)
    fp_only = [fps[i] for i in fp_present_idx]

    print(f"[nb]   computing drug-drug similarity over {n:,} drugs ...", flush=True)
    for i in range(n):
        # Pathway Jaccard
        if ps[i]:
            size_i = len(ps[i])
            for j in range(n):
                if i == j or not ps[j]:
                    continue
                inter = len(ps[i] & ps[j])
                if inter == 0:
                    continue
                sim[i, j] += W_PATHWAY * inter / (size_i + len(ps[j]) - inter)
        # Protein Jaccard
        if pr[i]:
            size_i = len(pr[i])
            for j in range(n):
                if i == j or not pr[j]:
                    continue
                inter = len(pr[i] & pr[j])
                if inter == 0:
                    continue
                sim[i, j] += W_PROTEIN * inter / (size_i + len(pr[j]) - inter)
        # ATC depth
        if atc[i]:
            for j in range(n):
                if i == j or not atc[j]:
                    continue
                sim[i, j] += W_ATC * _atc_depth(atc[i], atc[j]) / 7.0
        # SMILES Tanimoto (bulk)
        if fps[i] is not None:
            tanis = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fps[i], fp_only),
                dtype=np.float32
            )
            sim[i, fp_present_idx] += W_SMILES * tanis
        if (i + 1) % 500 == 0:
            print(f"[nb]     ... {i+1:,}/{n:,}", flush=True)

    np.fill_diagonal(sim, 0.0)
    return sim, id_to_idx


def precompute_neighbors(split_name: str, top_k: int = TOP_K_NEIGHBORS,
                         query_pids: list[str] | None = None,
                         output_path: Path | None = None,
                         audit_path: Path | None = None) -> None:
    """For every pair in manifest_<split>.parquet (or in `query_pids`),
    write top-K neighbor pairs.

    The neighbor *universe* is always restricted to `random_full.train` pairs
    (no leakage from any held-out test drugs) and further restricted to pairs
    that have a known family label.

    Args:
      split_name: tag used in default output filenames; also fallback source
                  of query pairs if `query_pids` is None.
      top_k: how many neighbors per query.
      query_pids: optional explicit list of query pair_ids to use instead of
                  the full manifest_<split>.parquet (e.g. for a held-out
                  evaluation subset).
      output_path: optional override for where to write the neighbor index.
      audit_path: optional override for where to write the audit JSON.
    """
    print(f"\n[nb] precomputing neighbors for split '{split_name}', K={top_k}")
    cb = ContextBuilder(neighbor_index_path=None)

    # Leakage-safe neighbor universe: random_full.train pair_ids
    rf = pq.read_table(DATA / "splits" / "manifest_random_full.parquet").to_pylist()
    allowed_train_pids = {r["pair_id"] for r in rf if r["split"] == "train"}
    print(f"[nb] allowed neighbor universe (random_full.train): "
          f"{len(allowed_train_pids):,}")

    if query_pids is None:
        manifest = pq.read_table(DATA / "splits" / f"manifest_{split_name}.parquet").to_pylist()
        query_pids = [r["pair_id"] for r in manifest]
        print(f"[nb] {len(query_pids):,} query pairs in split '{split_name}'")
    else:
        print(f"[nb] {len(query_pids):,} query pairs from custom manifest")

    # Drug universe = all drugs touched by either the allowed-train pairs
    # or any query pair.  Bounds memory.
    drug_universe = set()
    for pid, p in cb.pair_sig.items():
        if pid in allowed_train_pids:
            drug_universe.add(p["a_id"])
            drug_universe.add(p["b_id"])
    for qpid in query_pids:
        p = cb.pair_sig.get(qpid)
        if p:
            drug_universe.add(p["a_id"])
            drug_universe.add(p["b_id"])
    drug_universe = sorted(drug_universe)
    print(f"[nb] drug universe: {len(drug_universe):,}")

    t0 = time.time()
    sim, id2idx = _drug_sim_matrix(cb, drug_universe)
    print(f"[nb] drug-drug similarity built in {time.time()-t0:.0f}s "
          f"(nnz={int((sim>0).sum()):,})")

    # Per-drug top-M similar drugs (indices)
    m = min(DRUG_TOPM, len(drug_universe) - 1)
    topm = np.argpartition(-sim, range(m), axis=1)[:, :m]
    # Reorder each row so it's sorted by similarity desc
    for i in range(topm.shape[0]):
        topm[i] = topm[i][np.argsort(-sim[i, topm[i]])]
    print(f"[nb] per-drug top-{m} neighbor list built")

    # Pair lookup: (drug_x_id, drug_y_id) sorted → pair_id (restricted to allowed set)
    allowed_pair_of: dict[tuple[str, str], str] = {}
    for pid in allowed_train_pids:
        p = cb.pair_sig.get(pid)
        if p is None:
            continue
        x, y = sorted((p["a_id"], p["b_id"]))
        allowed_pair_of[(x, y)] = pid
    print(f"[nb] allowed pair lookup: {len(allowed_pair_of):,}")

    rows = []
    t_loop = time.time()
    stats_hit = 0
    for n_done, q_pid in enumerate(query_pids):
        sig = cb.pair_sig.get(q_pid)
        if sig is None:
            continue
        qa_id, qb_id = sig["a_id"], sig["b_id"]
        qa = id2idx.get(qa_id, -1)
        qb = id2idx.get(qb_id, -1)
        if qa < 0 or qb < 0:
            continue

        cand_drugs = set(topm[qa].tolist()) | set(topm[qb].tolist())
        # Exclude query's own drugs
        cand_drugs.discard(qa)
        cand_drugs.discard(qb)
        cand_drug_ids = [drug_universe[i] for i in cand_drugs]

        scored = []
        for i, x_id in enumerate(cand_drug_ids):
            x_idx = id2idx[x_id]
            for j in range(i + 1, len(cand_drug_ids)):
                y_id = cand_drug_ids[j]
                y_idx = id2idx[y_id]
                # Does pair exist in allowed universe?
                key = (x_id, y_id) if x_id < y_id else (y_id, x_id)
                npid = allowed_pair_of.get(key)
                if npid is None or npid == q_pid:
                    continue
                # Score: parallel or cross alignment between (qa,qb) and (x,y)
                s_par = sim[qa, x_idx] * sim[qb, y_idx]
                s_crs = sim[qa, y_idx] * sim[qb, x_idx]
                s = float(max(s_par, s_crs))
                if s <= 0:
                    continue
                scored.append((s, npid))

        if not scored:
            continue
        scored.sort(reverse=True)
        for s, npid in scored[:top_k]:
            lab = cb.labels.get(npid)
            if lab is None:
                continue
            a_name = cb.drugs.get(lab["a_id"], {}).get("name") or lab["a_id"]
            b_name = cb.drugs.get(lab["b_id"], {}).get("name") or lab["b_id"]
            if lab.get("bidirectional"):
                dtag = "bidirectional"
            elif lab.get("subject_drugbank_id") == lab["a_id"]:
                dtag = "a_to_b"
            elif lab.get("subject_drugbank_id") == lab["b_id"]:
                dtag = "b_to_a"
            else:
                dtag = "n/a"
            rows.append({
                "query_pair_id": q_pid,
                "neighbor_pair_id": npid,
                "similarity": s,
                "family": lab["family"],
                "subtype": lab["subtype"],
                "direction_tag": dtag,
                "polarity": lab.get("polarity"),
                "a_name": a_name,
                "b_name": b_name,
            })
            stats_hit += 1

        if (n_done + 1) % 500 == 0:
            elapsed = time.time() - t_loop
            eta = elapsed / (n_done + 1) * (len(query_pids) - n_done - 1)
            print(f"[nb]   processed {n_done+1:,}/{len(query_pids):,} "
                  f"({stats_hit:,} edges)  elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

    out_path = output_path if output_path is not None else (DATA / f"neighbor_index_{split_name}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path, compression="snappy")
    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"\n[nb] wrote {rel} "
          f"({len(rows):,} edges, avg "
          f"{len(rows)/max(1,len(query_pids)):.2f} per query)")

    # Audit JSON
    fam_counts = {}
    for r in rows:
        fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
    audit = {
        "split": split_name,
        "top_k_requested": top_k,
        "query_pairs": len(query_pids),
        "total_edges": len(rows),
        "avg_edges_per_query": len(rows) / max(1, len(query_pids)),
        "queries_with_any_neighbor": len({r["query_pair_id"] for r in rows}),
        "neighbors_by_family": fam_counts,
        "neighbor_universe_size": len(allowed_pair_of),
        "drug_universe_size": len(drug_universe),
        "sim_weights": {"pathway": W_PATHWAY, "protein": W_PROTEIN,
                        "atc": W_ATC, "smiles": W_SMILES},
        "drug_topm": DRUG_TOPM,
    }
    if audit_path is None:
        audit_path = ROOT / "outputs" / "audit" / f"b1a_neighbor_index_{split_name}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    try:
        rel_audit = audit_path.relative_to(ROOT)
    except ValueError:
        rel_audit = audit_path
    print(f"[nb] audit → {rel_audit}")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--precompute_neighbors", type=str,
                   help="Precompute top-K neighbor index for a split "
                        "(e.g. 'subset25k')")
    p.add_argument("--top_k", type=int, default=TOP_K_NEIGHBORS)
    p.add_argument("--query_manifest_file", type=str, default=None,
                   help="Optional JSONL/CSV/Parquet of query pair_ids to use "
                        "instead of the full split manifest. JSONL must have "
                        "a `pair_id` field per line.")
    p.add_argument("--output_path", type=str, default=None,
                   help="Override output path for the neighbor index parquet.")
    p.add_argument("--audit_path", type=str, default=None,
                   help="Override output path for the audit JSON.")
    p.add_argument("--demo", type=str,
                   help="Build context for a single pair_id and print it")
    args = p.parse_args()

    if args.precompute_neighbors:
        custom_pids: list[str] | None = None
        if args.query_manifest_file:
            qm = Path(args.query_manifest_file)
            if qm.suffix == ".parquet":
                custom_pids = [r["pair_id"] for r in pq.read_table(qm).to_pylist()]
            else:
                custom_pids = []
                with qm.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("{"):
                            custom_pids.append(json.loads(line)["pair_id"])
                        else:
                            custom_pids.append(line.split(",")[0].strip())
        precompute_neighbors(
            args.precompute_neighbors,
            args.top_k,
            query_pids=custom_pids,
            output_path=Path(args.output_path) if args.output_path else None,
            audit_path=Path(args.audit_path) if args.audit_path else None,
        )
    elif args.demo:
        cb = ContextBuilder()
        ctx = cb.build(args.demo)
        print(json.dumps(ctx.to_dict(), indent=2, default=str)[:4000])
        print(f"\ncontext_ids ({len(ctx.context_ids())}): "
              f"{sorted(ctx.context_ids())[:30]}...")
    else:
        p.print_help()
