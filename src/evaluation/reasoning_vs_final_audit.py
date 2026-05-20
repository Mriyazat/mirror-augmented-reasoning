"""Quantify how often the student's reasoning trace points to a different family
than its final_answer.family — and what fraction of those traces are *correct*
(i.e. trace majority matches gold).

Outputs per-split JSON + a printed table.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def trace_majority(steps):
    hints = [s.get("family_hint") for s in steps if s.get("role") != "conclusion"]
    hints = [h for h in hints if h in FAMS]
    if not hints:
        return None, 0
    c = Counter(hints)
    fam, k = c.most_common(1)[0]
    return fam, k / len(hints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    truth = {r["pair_id"]: r["family"] for r in pq.read_table(
        "data_processed/labels_hierarchical.parquet",
        columns=["pair_id", "family"]).to_pylist()}
    keep = set()
    with open(args.manifest) as f:
        for line in f:
            keep.add(json.loads(line)["pair_id"])

    n_total = 0
    n_final_correct = 0
    n_trace_correct = 0
    n_disagree = 0
    n_disagree_trace_correct = 0
    n_disagree_final_correct = 0
    n_both_wrong_but_different = 0

    rescuable_examples = []

    with open(args.predictions) as f:
        for line in f:
            r = json.loads(line)
            pid = r["pair_id"]
            if (r.get("input_order") or "ab") != "ab":
                continue
            if pid not in keep or pid not in truth:
                continue
            gold = truth[pid]
            fp = r.get("final_prediction") or {}
            final_fam = fp.get("family")
            try:
                conf = float(fp.get("confidence", 0.0) or 0.0)
            except Exception:
                conf = 0.0
            trace = r.get("trace") or {}
            steps = trace.get("steps") or []
            trace_fam, trace_strength = trace_majority(steps)

            n_total += 1
            if final_fam == gold:
                n_final_correct += 1
            if trace_fam == gold:
                n_trace_correct += 1
            if trace_fam and trace_fam != final_fam:
                n_disagree += 1
                if trace_fam == gold:
                    n_disagree_trace_correct += 1
                if final_fam == gold:
                    n_disagree_final_correct += 1
                if trace_fam != gold and final_fam != gold:
                    n_both_wrong_but_different += 1

                # Capture rescue candidates (trace right, final wrong)
                if (trace_fam == gold and final_fam != gold and
                    trace_strength >= 0.5 and len(rescuable_examples) < 25):
                    rescuable_examples.append({
                        "pair_id": pid,
                        "gold": gold,
                        "final_fam": final_fam,
                        "trace_fam": trace_fam,
                        "trace_strength": round(trace_strength, 2),
                        "conf": round(conf, 2),
                        "conclusion_claim": (steps[-1].get("claim") if steps else "")[:200],
                    })

    out = {
        "split": args.split,
        "n_total": n_total,
        "final_correct_frac": round(n_final_correct / max(1, n_total), 4),
        "trace_majority_correct_frac": round(n_trace_correct / max(1, n_total), 4),
        "n_disagree_final_vs_trace": n_disagree,
        "disagree_frac": round(n_disagree / max(1, n_total), 4),
        "of_disagree_trace_correct": n_disagree_trace_correct,
        "of_disagree_final_correct": n_disagree_final_correct,
        "of_disagree_both_wrong_different": n_both_wrong_but_different,
        "rescue_upper_bound": n_disagree_trace_correct,
        "miss_risk_if_rescued":  n_disagree_final_correct,
        "net_potential_gain": n_disagree_trace_correct - n_disagree_final_correct,
        "rescuable_examples": rescuable_examples,
    }
    print(f"\n=== {args.split} ===")
    for k, v in out.items():
        if k != "rescuable_examples":
            print(f"  {k}: {v}")
    print(f"  (top {len(rescuable_examples)} rescuable examples saved)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
