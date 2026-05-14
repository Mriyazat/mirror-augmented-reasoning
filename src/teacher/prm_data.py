"""B3 pre-work — Build PRM training data from QC'd teacher traces.

Med-PRM format:  (question, solution_with_separator_tokens, expected +/- per step)

For DDI-PRM we convert each candidate trace into:

    [question]
      A = <drug A name>  (DB_ID)
      B = <drug B name>  (DB_ID)
      Context:
        shared pathways: <pathway_id list>
        shared proteins: <uniprot list>
        PK flags: <flag list>

    [solution]  (per step, ending with " ки")
      Step 1: <claim>  [evidence: ID1, ID2]  (direction_tag)\n ки
      Step 2: ...

    [gold labels per step]
      1 ↦ + / -   (auto-derived)

We derive the per-step gold label by running a miniature version of the QC
gates that act on INDIVIDUAL STEPS (not the whole trace):

    Step is GOOD (+) iff ALL of:
      - every evidence_id ∈ context_ids
      - direction_tag ∈ {n/a, bidirectional, or matches gold_direction}
      - family_hint (if present) == gold_family
      - no banned silencing phrase
      - claim is plausibly relevant  (mentions drug A name OR drug B name OR
        any ID from evidence_ids that also appears in context_ids)

This gives a noisy but dense supervisory signal.  Because we derive labels
from the SAME data the teacher must cite, the PRM learns to spot actual
evidence mismatches, direction flips, and off-family reasoning.

Usage:
    python -m src.teacher.prm_data \\
        --qc   outputs/teacher/qc_subset25k_ollama-llama3.1_8b.jsonl \\
        --split subset25k \\
        --out_train outputs/teacher/prm_train.jsonl \\
        --out_eval  outputs/teacher/prm_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.teacher.context_builder import ContextBuilder, DATA
from src.teacher.schema import (BANNED_ABSTAIN_PHRASES, load_rubric)

ROOT = Path(__file__).resolve().parents[2]


def derive_step_gold(step: dict, ctx_ids: set[str],
                     gold_family: str | None, gold_direction: str | None,
                     a_name: str, b_name: str) -> tuple[bool, list[str]]:
    """Returns (label, reasons_if_bad). label=True means '+' (good step)."""
    bad: list[str] = []
    ev = step.get("evidence_ids", [])
    for eid in ev:
        if eid not in ctx_ids:
            bad.append(f"evidence {eid!r} not in context")
            break
    if step.get("family_hint") and gold_family and step["family_hint"] != gold_family:
        bad.append(f"family_hint {step['family_hint']!r} != gold {gold_family!r}")
    dt = step.get("direction_tag", "n/a")
    if dt in {"a_to_b", "b_to_a"} and gold_direction in {"a_to_b", "b_to_a"}:
        if dt != gold_direction:
            bad.append(f"direction {dt!r} contradicts gold {gold_direction!r}")
    claim_lc = step.get("claim", "").lower()
    for phr in BANNED_ABSTAIN_PHRASES:
        if phr in claim_lc:
            bad.append(f"silencing phrase {phr!r}")
            break
    # Relevance: mention a drug name or an evidence id in ctx_ids
    mentions = (a_name.lower() in claim_lc or b_name.lower() in claim_lc
                or any(eid in ctx_ids for eid in ev))
    if not mentions:
        bad.append("no drug/evidence mention in claim (relevance)")
    return (len(bad) == 0, bad)


def trace_to_medprm_example(qc_rec: dict, ctx, a_name: str, b_name: str,
                            sep: str) -> dict:
    """Convert one QC record into Med-PRM training example.

    Returns a dict with:
        question:   str
        solution:   str (ends each step with sep; final conclusion too)
        step_labels: list[bool]  (one per sep marker, True=+)
        metadata:   dict  (pair_id, family, gates, etc.)
    """
    parsed = qc_rec["parsed"]
    if parsed is None:
        return None  # schema-failed; skip
    steps = parsed["steps"]
    ctx_ids = ctx.context_ids()

    # Build question
    qlines = [f"A = {a_name} ({ctx.a.drugbank_id})",
              f"B = {b_name} ({ctx.b.drugbank_id})",
              "Context:"]
    if ctx.shared_pathways:
        qlines.append("  shared pathways: " + ", ".join(p.pathway_id for p in ctx.shared_pathways))
    if ctx.shared_proteins:
        qlines.append("  shared proteins: " + ", ".join(p.uniprot for p in ctx.shared_proteins))
    if ctx.a.active_pk_flags or ctx.b.active_pk_flags:
        qlines.append("  active PK flags A: " + ", ".join(ctx.a.active_pk_flags[:10]))
        qlines.append("  active PK flags B: " + ", ".join(ctx.b.active_pk_flags[:10]))
    question = "\n".join(qlines)

    # Build solution with per-step separator
    sol_lines = []
    step_labels = []
    step_reasons = []
    for s in steps:
        ev = f" [evidence: {', '.join(s.get('evidence_ids', []))}]" if s.get("evidence_ids") else ""
        dt = s.get("direction_tag", "n/a")
        dt_str = f" ({dt})" if dt != "n/a" else ""
        sol_lines.append(f"Step {s['step_id']}: {s['claim']}{ev}{dt_str}{sep}")
        ok, bad = derive_step_gold(s, ctx_ids, qc_rec.get("gold_family"),
                                   qc_rec.get("gold_direction"),
                                   a_name, b_name)
        step_labels.append(ok)
        step_reasons.append(bad)

    # Final conclusion line (ORM signal) — pass/fail on whole-trace QC outcome
    ans = parsed["final_answer"]
    sol_lines.append(f"Final: family={ans['family']}, subtype={ans['subtype']}, "
                     f"direction={ans['direction_tag']}, polarity={ans['polarity']}, "
                     f"abstain={ans['abstain']}{sep}")
    # Final step label: overall QC pass
    step_labels.append(bool(qc_rec.get("critical_passed")))
    step_reasons.append([] if qc_rec.get("critical_passed") else qc_rec.get("errors", []))

    return {
        "question": question,
        "solution": "\n".join(sol_lines),
        "step_labels": step_labels,
        "step_reasons": step_reasons,
        "pair_id": qc_rec["pair_id"],
        "candidate_id": qc_rec["candidate_id"],
        "family": qc_rec.get("gold_family"),
        "direction": qc_rec.get("gold_direction"),
        "trace_strict_passed": bool(qc_rec.get("passed")),
    }


def run(qc_path: Path, split: str, out_train: Path, out_eval: Path,
        eval_frac: float = 0.05, seed: int = 13):
    rubric = load_rubric()
    sep = rubric["separator_token"]
    cb = ContextBuilder(neighbor_index_path=DATA / f"neighbor_index_{split}.parquet")

    # Bucket by pair_id so all candidates from a pair end up in the same split
    by_pair: dict[str, list[dict]] = defaultdict(list)
    with qc_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_pair[r["pair_id"]].append(r)
    pairs = list(by_pair)
    random.Random(seed).shuffle(pairs)
    n_eval = int(len(pairs) * eval_frac)
    eval_pairs = set(pairs[:n_eval])
    print(f"[prm] pairs: {len(pairs):,}  (eval={n_eval:,})")

    # Stats
    n_emit = 0
    n_skipped = 0
    step_plus = Counter()
    step_total = Counter()
    fam_counts = Counter()

    with out_train.open("w") as ftr, out_eval.open("w") as fev:
        for pid, recs in by_pair.items():
            try:
                ctx = cb.build(pid)
            except Exception as e:
                n_skipped += len(recs)
                continue
            a_name = ctx.a.name
            b_name = ctx.b.name
            for r in recs:
                ex = trace_to_medprm_example(r, ctx, a_name, b_name, sep)
                if ex is None:
                    n_skipped += 1
                    continue
                target = fev if pid in eval_pairs else ftr
                target.write(json.dumps(ex) + "\n")
                n_emit += 1
                fam_counts[r.get("gold_family") or "?"] += 1
                for lbl in ex["step_labels"]:
                    step_total["all"] += 1
                    if lbl:
                        step_plus["all"] += 1

    print(f"[prm] emitted {n_emit:,} examples (skipped {n_skipped})")
    pct = 100 * step_plus["all"] / max(1, step_total["all"])
    print(f"[prm] step-level '+': {step_plus['all']:,}/{step_total['all']:,} ({pct:.1f}%)")
    print(f"[prm] family dist: {dict(fam_counts.most_common())}")
    print(f"[prm] train → {out_train}")
    print(f"[prm] eval  → {out_eval}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qc", required=True)
    p.add_argument("--split", default="subset25k")
    p.add_argument("--out_train", required=True)
    p.add_argument("--out_eval", required=True)
    p.add_argument("--eval_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    run(Path(args.qc), args.split, Path(args.out_train), Path(args.out_eval),
        args.eval_frac, args.seed)
