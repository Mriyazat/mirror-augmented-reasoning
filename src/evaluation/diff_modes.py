"""Diagnose where greedy and rerank4 disagree on the val5k rescue set.

Used to size the trace-align merge gain: tells us how many *additional*
rescue candidates greedy surfaces that rerank4 doesn't, and on which
families the two modes systematically diverge.

Outputs a small markdown summary to stdout + writes:
  outputs/student/trace_align/rescue_data/mode_diff.md
  outputs/student/trace_align/rescue_data/mode_diff.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FAMS = ["AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
        "PK_Distribution", "PK_Excretion", "PK_Metabolism"]


def _load_labels():
    rows = pq.read_table(ROOT / "data_processed/labels_hierarchical.parquet").to_pylist()
    return {r["pair_id"]: r["family"] for r in rows}


def _trace_majority(trace):
    if not isinstance(trace, dict):
        return None
    hints = [s.get("family_hint") for s in (trace.get("steps") or [])
             if s.get("family_hint") in FAMS]
    if not hints:
        return None
    return Counter(hints).most_common(1)[0][0]


def _load_preds(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("input_order") or "ab") != "ab":
                continue
            pid = r["pair_id"]
            fp = r.get("final_prediction") or {}
            out[pid] = {
                "final": fp.get("family"),
                "abstain": bool(fp.get("abstain", False)),
                "trace_maj": _trace_majority(r.get("trace")),
            }
    return out


def main():
    rerank4 = _load_preds(
        ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_rerank4.jsonl")
    greedy = _load_preds(
        ROOT / "outputs/student/trace_align/rescue_data/preds_val5k_greedy.jsonl")
    labels = _load_labels()

    common = sorted(set(rerank4) & set(greedy) & set(labels))
    n = len(common)

    agree = 0
    both_correct = 0
    rerank_only_correct = 0
    greedy_only_correct = 0
    both_wrong = 0

    rerank_rescues = set()
    greedy_rescues = set()
    both_rescues = set()

    fam_diff = defaultdict(lambda: Counter())

    for pid in common:
        gold = labels[pid]
        r = rerank4[pid]
        g = greedy[pid]
        if r["final"] == g["final"]:
            agree += 1
        if r["final"] == gold and g["final"] == gold:
            both_correct += 1
        elif r["final"] == gold:
            rerank_only_correct += 1
            fam_diff[gold]["rerank_only"] += 1
        elif g["final"] == gold:
            greedy_only_correct += 1
            fam_diff[gold]["greedy_only"] += 1
        else:
            both_wrong += 1
            fam_diff[gold]["both_wrong"] += 1

        # rescue = trace_maj == gold AND final != gold AND not abstain
        if (r["trace_maj"] == gold and r["final"] != gold
                and not r["abstain"]):
            rerank_rescues.add(pid)
        if (g["trace_maj"] == gold and g["final"] != gold
                and not g["abstain"]):
            greedy_rescues.add(pid)

    both_rescues = rerank_rescues & greedy_rescues
    greedy_unique = greedy_rescues - rerank_rescues
    rerank_unique = rerank_rescues - greedy_rescues

    md = []
    md.append("# rerank4 vs greedy on val5k (rescue diagnosis)\n")
    md.append(f"n_common = {n}\n")
    md.append("## Top-line agreement\n")
    md.append(f"- both correct          : {both_correct}  ({both_correct/n:.1%})")
    md.append(f"- rerank only correct   : {rerank_only_correct}  ({rerank_only_correct/n:.1%})")
    md.append(f"- greedy only correct   : {greedy_only_correct}  ({greedy_only_correct/n:.1%})")
    md.append(f"- both wrong            : {both_wrong}  ({both_wrong/n:.1%})")
    md.append(f"- exact-final-agreement : {agree}  ({agree/n:.1%})\n")

    md.append("## Rescue candidates (trace_maj == gold AND final != gold)\n")
    md.append(f"- rerank4-only rescues : {len(rerank_unique)}")
    md.append(f"- greedy-only rescues  : {len(greedy_unique)}")
    md.append(f"- both modes rescue    : {len(both_rescues)}")
    md.append(f"- union (merge gain)   : {len(rerank_rescues | greedy_rescues)}  "
              f"(rerank-alone = {len(rerank_rescues)}, "
              f"delta = +{len(rerank_rescues | greedy_rescues) - len(rerank_rescues)})\n")

    md.append("## Per-family disagreement (when one mode is right, the other wrong)\n")
    md.append("| family | rerank_only | greedy_only | both_wrong |")
    md.append("|---|---:|---:|---:|")
    for f in FAMS:
        row = fam_diff.get(f, Counter())
        md.append(f"| {f} | {row['rerank_only']} | {row['greedy_only']} | {row['both_wrong']} |")

    out_md = ROOT / "outputs/student/trace_align/rescue_data/mode_diff.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    print("\n".join(md))

    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps({
        "n_common": n,
        "both_correct": both_correct,
        "rerank_only_correct": rerank_only_correct,
        "greedy_only_correct": greedy_only_correct,
        "both_wrong": both_wrong,
        "exact_final_agreement": agree,
        "rerank_rescues": len(rerank_rescues),
        "greedy_rescues": len(greedy_rescues),
        "both_rescues": len(both_rescues),
        "greedy_unique_rescues": len(greedy_unique),
        "rerank_unique_rescues": len(rerank_unique),
        "union_rescues": len(rerank_rescues | greedy_rescues),
        "per_family": {f: dict(fam_diff.get(f, Counter())) for f in FAMS},
    }, indent=2))
    print(f"\nwrote {out_md}\nwrote {out_json}")


if __name__ == "__main__":
    main()
