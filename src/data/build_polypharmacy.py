"""evaluation data -- Polypharmacy 3-drug evaluation set.

Motivation
----------
Real-world DDI prediction rarely stops at the 2-drug level: most
elderly patients in the US take 5+ medications simultaneously.  the earlier baseline's
evaluation was strictly 2-drug, which meant it never tested
*compositional* reasoning -- can the model reason about three drugs
interacting through a shared mechanism?

This builder finds all "triangles" (A, B, C) where all three pairwise
labels exist in our dataset, so we can evaluate whether the model's
three pairwise predictions are MUTUALLY CONSISTENT.

Consistency checks (computed at evaluation time, not here)
----------------------------------------------------------
    - **Family-triangle consistency**: if two of the three pairs are
      PK_Metabolism (shared-enzyme mediated) AND the third pair is also
      PK_Metabolism, the model's implied mechanism enzymes must agree.
    - **Direction-triangle consistency**: for three directional
      predictions forming a cycle (A->B, B->C, C->A), at least one
      must be wrong (no true 3-cycle of strict dominance exists in our
      annotation).  A consistent model abstains on at least one edge.
    - **Severity-triangle consistency**: if A<->B is severe and B<->C
      is severe and both route through the same enzyme, A<->C should
      also be flagged severe.

Output record format (JSONL)
----------------------------
    {
      "triangle_id":  "DB00001|DB06605|DB06695",   # sorted drugbank ids
      "drugs":        ["DB00001", "DB06605", "DB06695"],
      "pair_ids":     ["DB00001|DB06605", "DB00001|DB06695", "DB06605|DB06695"],
      "pair_labels": [
          {"pair_id": ..., "family": ..., "subtype": ..., "direction_tag": ..., "bidirectional": ...},
          ...
      ],
      "shared_mechanism_candidate": "cyp3a4" | null,  # if any two edges
                                                       # share an enzyme prefix
      "severity_max":                "Major" | "Moderate" | "Minor" | null,
      "split":                       "test"
    }

Generation strategy
-------------------
- Build a drug-interaction graph restricted to the split test set.
- For each drug A, iterate over pairs (B, C) of its neighbors; if B-C
  also has a labeled edge, emit a triangle.
- De-duplicate triangles by canonical-sorted drug id.
- Optionally cap total triangles with `--max_triangles` (sampled).

Complexity: O(sum_A deg(A) * deg(A)) which is tractable when we
restrict to the test split.

CLI
---
    python -m src.data.build_polypharmacy \
        --labels    data_processed/labels_hierarchical.parquet \
        --splits    data_processed/splits/manifest_pair_cold.parquet \
        --split_section test \
        --output    data_processed/polypharmacy_pair_cold_test.jsonl \
        --report    outputs/audit/polypharmacy_pair_cold_test.md \
        --max_triangles 5000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


_FAMILY_TO_PREFIX: dict[str, str | None] = {
    # Approximate "shared-mechanism" heuristic: PK families get tagged
    # with their most likely enzyme prefix from the subtype; others get None.
    "PK_Metabolism":   "enzyme",
    "PK_Excretion":    "transporter",
    "PK_Distribution": "protein_binding",
    "PK_Absorption":   "transporter",
}


def _load_split_pairs(path: Path | None, section: str) -> set[str] | None:
    if not path or not path.exists():
        return None
    tbl = pq.read_table(path, columns=["pair_id", "split"]).to_pylist()
    return {r["pair_id"] for r in tbl if r.get("split") == section}


def _canonical_triangle_id(drugs: list[str]) -> str:
    return "|".join(sorted(drugs))


def _shared_mechanism(labels: list[dict]) -> str | None:
    """Lightweight heuristic: if >= 2 of 3 edges share the same family,
    return that family's prefix tag.  Otherwise None."""
    fams = [l.get("family") for l in labels]
    c = Counter(fams)
    top_fam, top_n = c.most_common(1)[0]
    if top_n >= 2:
        return _FAMILY_TO_PREFIX.get(top_fam, top_fam)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",       required=True)
    ap.add_argument("--splits",       default=None)
    ap.add_argument("--split_section", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--output",       required=True)
    ap.add_argument("--report",       default=None)
    ap.add_argument("--max_triangles", type=int, default=5000)
    ap.add_argument("--min_shared_family", type=int, default=2,
                    help="Require at least N of the 3 edges share the same family."
                         "  Set to 0 to keep all triangles.  Default 2 (the subset"
                         " likely to expose compositional-mechanism reasoning).")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    split_pairs = _load_split_pairs(Path(args.splits), args.split_section) if args.splits else None
    print(f"[poly] split filter: {len(split_pairs) if split_pairs else 'ALL'} pair_ids")

    tbl = pq.read_table(args.labels, columns=[
        "pair_id", "a_id", "b_id", "family", "subtype",
        "bidirectional", "subject_drugbank_id",
    ]).to_pylist()

    # Restrict to split
    if split_pairs is not None:
        tbl = [r for r in tbl if r["pair_id"] in split_pairs]
    print(f"[poly] edges in scope: {len(tbl):,}")

    # Build adjacency: drug -> list of (neighbor, label_row)
    adj: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in tbl:
        a, b = r["a_id"], r["b_id"]
        adj[a][b] = r
        adj[b][a] = r

    # Enumerate triangles by iterating edges (A, B) and intersecting
    # adj[A] & adj[B] to find common neighbors C.
    seen: set[str] = set()
    triangles: list[dict] = []
    n_checked = 0

    edges = [(r["a_id"], r["b_id"], r) for r in tbl]
    rng.shuffle(edges)  # random sampling when we hit --max_triangles

    for a, b, _ in edges:
        common = set(adj[a].keys()) & set(adj[b].keys())
        for c in common:
            if c == a or c == b:
                continue
            drugs = [a, b, c]
            tri_id = _canonical_triangle_id(drugs)
            if tri_id in seen:
                continue
            seen.add(tri_id)
            n_checked += 1

            l_ab = adj[a][b]
            l_ac = adj[a][c]
            l_bc = adj[b][c]
            labels = [l_ab, l_ac, l_bc]

            fam_counter = Counter(l.get("family") for l in labels)
            top_n = fam_counter.most_common(1)[0][1] if fam_counter else 0
            if top_n < args.min_shared_family:
                continue

            pair_labels = [{
                "pair_id":             l["pair_id"],
                "family":              l.get("family"),
                "subtype":             l.get("subtype"),
                "bidirectional":       bool(l.get("bidirectional", False)),
                "subject_drugbank_id": l.get("subject_drugbank_id"),
            } for l in labels]

            rec = {
                "triangle_id":  tri_id,
                "drugs":        sorted(drugs),
                "pair_ids":     [l["pair_id"] for l in labels],
                "pair_labels":  pair_labels,
                "shared_mechanism_candidate": _shared_mechanism(labels),
                "split":        args.split_section,
            }
            triangles.append(rec)

            if len(triangles) >= args.max_triangles:
                break
        if len(triangles) >= args.max_triangles:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in triangles:
            f.write(json.dumps(rec) + "\n")

    print(f"[poly] triangles examined : {n_checked:,}")
    print(f"[poly] triangles emitted  : {len(triangles):,}")
    fam_mix = Counter(
        tuple(sorted(l.get("family") or "" for l in tri["pair_labels"]))
        for tri in triangles
    )
    print(f"[poly] top family-triples : {fam_mix.most_common(5)}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        with open(rep, "w") as f:
            f.write("# Polypharmacy 3-drug evaluation set\n\n")
            f.write(f"- Source labels: `{args.labels}`\n")
            f.write(f"- Split:         `{args.split_section}` ({args.splits or 'all'})\n")
            f.write(f"- Triangles examined: **{n_checked:,}**\n")
            f.write(f"- Triangles emitted : **{len(triangles):,}**\n\n")
            f.write(f"- Filter: at least {args.min_shared_family} of 3 edges "
                    f"share the same family.\n\n")
            f.write("## Top family triples\n\n| families | n |\n|---|---:|\n")
            for triple, n in fam_mix.most_common(20):
                f.write(f"| `{triple}` | {n:,} |\n")
        print(f"[poly] wrote report {rep}")


if __name__ == "__main__":
    main()
