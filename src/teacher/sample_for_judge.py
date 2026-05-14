"""B7.1 -- Stratified sampler for human / LLM-as-judge review.

Pulls a stratified-random sample from teacher_clean and joins it with:
  - The official DrugBank description text from labels_hierarchical
    (the canonical gold-text for each pair).
  - A compact summary of the ContextBundle (drug names, key flags,
    shared proteins, neighbor pairs).

Output shape (one record per line):
    {
      "pair_id":  "DB00567|DB13409",
      "drugs":    {"a": {"id":..., "name":...}, "b": {...}},
      "gold":     {"family": "...", "subtype": "...",
                   "direction_tag": "...", "polarity": "...",
                   "description": "<official DrugBank prose>",
                   "subject_drugbank_id": "..."},
      "context_summary": {
          "shared_proteins": ["P10635 / Cytochrome P450 2D6", ...],
          "shared_pathways": ["SMP00002 / Heparin metabolism"],
          "a_pk_flags":     ["cyp3a4_sub", ...],
          "b_pk_flags":     ["cyp3a4_inh", ...],
          "neighbor_pairs": ["DB01333|DB14678 (PK_Metabolism)"]
      },
      "trace": {
          "tier": "...",
          "steps": [{step_id, role, claim, evidence_ids, direction_tag}, ...],
          "final_answer": {...}
      }
    }

The output is intended to be read directly by a reviewer (human or LLM
judge) and graded.  Rubric (5 dimensions, 1-5 each):

  1. **Pharmacology correctness**: does the trace describe a real-world
     plausible mechanism for the gold family/subtype?
  2. **Mechanism specificity**: does it name specific enzymes, transporters,
     or proteins -- not just generic "drug A interacts with drug B"?
  3. **Evidence grounding**: do the cited evidence_ids actually support
     the claims they're attached to?
  4. **Calibration**: is the abstain/confidence appropriate to the
     reasoning's certainty?
  5. **Coherence vs DrugBank**: does the trace's final_answer match the
     direction + polarity expressed in the DrugBank description?

CLI
---
    python -m src.teacher.sample_for_judge \
        --teacher_clean   /path/to/teacher_clean.jsonl \
        --labels          data_processed/labels_hierarchical.parquet \
        --output          outputs/audit/judge_sample.jsonl \
        --n_per_cell      5 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.teacher.schema import extract_json_block
from src.teacher.context_builder import ContextBuilder, DATA


def _load_descriptions(labels_path: Path) -> dict[str, dict]:
    """pair_id -> {description, family, subtype, polarity, subject_id, ...}"""
    tbl = pq.read_table(labels_path, columns=[
        "pair_id", "description", "template", "family", "subtype",
        "polarity", "subject_drugbank_id", "object_drugbank_id",
        "bidirectional",
    ]).to_pylist()
    out = {}
    for r in tbl:
        out[r["pair_id"]] = r
    return out


def _ctx_summary(cb: ContextBuilder, pair_id: str) -> dict:
    try:
        ctx = cb.build(pair_id)
    except Exception as e:
        return {"error": f"ctx_fail:{type(e).__name__}"}
    sp = [f"{p.uniprot} / {p.protein_name} (a:{p.a_role}/{','.join(p.a_actions) or '-'} b:{p.b_role}/{','.join(p.b_actions) or '-'})"
          for p in ctx.shared_proteins[:6]]
    sm = [f"{p.pathway_id} / {p.pathway_name} ({p.source})" for p in ctx.shared_pathways[:5]]
    a_prot = [f"{p.uniprot} / {p.protein_name} ({p.role}/{','.join(p.actions) or '-'})"
              for p in ctx.a_proteins[:6]]
    b_prot = [f"{p.uniprot} / {p.protein_name} ({p.role}/{','.join(p.actions) or '-'})"
              for p in ctx.b_proteins[:6]]
    a_flags = list(ctx.a.active_pk_flags)[:10]
    b_flags = list(ctx.b.active_pk_flags)[:10]
    nb = []
    for n in ctx.neighbors[:5]:
        nb.append(f"{n.pair_id} ({n.family}/{n.subtype})")
    return {
        "shared_proteins":  sp,
        "shared_pathways":  sm,
        "a_proteins_top":   a_prot,
        "b_proteins_top":   b_prot,
        "a_pk_flags":       a_flags,
        "b_pk_flags":       b_flags,
        "neighbor_pairs":   nb,
        "a_atc":            list(ctx.a.atc_codes)[:5],
        "b_atc":            list(ctx.b.atc_codes)[:5],
        "n_context_ids":    len(ctx.context_ids()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_clean", required=True)
    ap.add_argument("--labels",        required=True)
    ap.add_argument("--output",        required=True)
    ap.add_argument("--split",         default="subset25k")
    ap.add_argument("--n_per_cell",    type=int, default=3,
                    help="Sample N per (tier x family) bucket.")
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"[judge] loading labels...")
    descs = _load_descriptions(Path(args.labels))
    print(f"[judge] loading teacher_clean...")
    teacher_records: list[dict] = []
    for line in open(args.teacher_clean):
        teacher_records.append(json.loads(line))
    print(f"[judge] {len(teacher_records):,} records loaded")

    # Stratify by (tier, family)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in teacher_records:
        cell = (r.get("tier", "?"), r.get("family", "?"))
        by_cell[cell].append(r)

    sample: list[dict] = []
    for cell, recs in sorted(by_cell.items()):
        rng.shuffle(recs)
        n_take = min(args.n_per_cell, len(recs))
        for r in recs[:n_take]:
            sample.append(r)
    print(f"[judge] sampled {len(sample):,} records across {len(by_cell):,} (tier,family) cells")

    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{args.split}.parquet")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_emitted = 0
    with open(args.output, "w") as fout:
        for r in sample:
            pid = r["pair_id"]
            parsed = extract_json_block(r["messages"][2]["content"])
            if parsed is None:
                continue
            d = descs.get(pid, {})
            ctx_sum = _ctx_summary(cb, pid)
            try:
                ctx = cb.build(pid)
                a_info = {"id": ctx.a.drugbank_id, "name": ctx.a.name}
                b_info = {"id": ctx.b.drugbank_id, "name": ctx.b.name}
            except Exception:
                a_info = {"id": pid.split("|")[0]}
                b_info = {"id": pid.split("|")[1]}

            out = {
                "pair_id":          pid,
                "drugs":            {"a": a_info, "b": b_info},
                "gold": {
                    "family":               d.get("family"),
                    "subtype":              d.get("subtype"),
                    "polarity":             d.get("polarity"),
                    "subject_drugbank_id":  d.get("subject_drugbank_id"),
                    "object_drugbank_id":   d.get("object_drugbank_id"),
                    "bidirectional":        bool(d.get("bidirectional", False)),
                    "description":          d.get("description"),
                    "template":             d.get("template"),
                },
                "context_summary":  ctx_sum,
                "trace": {
                    "tier":          r.get("tier"),
                    "critic_score":  r.get("critic_score"),
                    "qc_strict_passed": r.get("qc_strict_passed"),
                    "sample_weight": r.get("sample_weight"),
                    "steps":         parsed.get("steps", []),
                    "final_answer":  parsed.get("final_answer", {}),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_emitted += 1

    print(f"[judge] wrote {n_emitted:,} judge records -> {args.output}")


if __name__ == "__main__":
    main()
