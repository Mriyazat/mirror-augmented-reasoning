"""Audit low-scoring traces to spot-check whether judges were unfair.

Identifies (a) the N lowest-scoring traces for a given model (default: ours)
and (b) the M traces with the most judge disagreement. For each, prints:
  * The actual trace text (steps + final answer)
  * Each judge's per-dim scores + composite
  * Each judge's justifications (why they scored each dim that way)
  * Each judge's length-bias self-check
  * Gold-truth family for the pair (so you can verify correctness)

Output: a Markdown audit file you can read in the IDE.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.evaluation.run_trace_quality_xjudge import (
    extract_trace_for_pair,
    render_trace_for_judge,
)
from src.evaluation.trace_quality_rubric import RUBRIC_DIMENSIONS

DIM_IDS = [d["id"] for d in RUBRIC_DIMENSIONS]
DIM_NAMES = {d["id"]: d["name"] for d in RUBRIC_DIMENSIONS}


def composite(scores: dict) -> float:
    vals = [scores[d] for d in DIM_IDS if d in scores]
    return sum(vals) / max(len(vals), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", required=True)
    ap.add_argument("--traces_file", required=True, help="Trace file for the model being audited.")
    ap.add_argument("--model", default="ours", help="Model key to audit.")
    ap.add_argument("--n_low", type=int, default=10, help="Number of lowest-scoring cases.")
    ap.add_argument("--n_high_disagreement", type=int, default=5, help="Number of high-judge-disagreement cases.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth_file", default="outputs/diag2/trace_quality/sampled_pair_truth.json")
    args = ap.parse_args()

    truth = json.load(open(args.truth_file))

    # Load model traces by pair_id
    model_traces = {}
    with open(args.traces_file) as f:
        for L in f:
            r = json.loads(L)
            if r.get("pair_id"):
                model_traces[r["pair_id"]] = r

    # Load all judgments for this model: pid -> {judge: judgment}
    judgments_by_pair = defaultdict(dict)
    with open(args.judgments) as f:
        for L in f:
            r = json.loads(L)
            if r.get("model") != args.model:
                continue
            if not r.get("scores"):
                continue
            judgments_by_pair[r["pair_id"]][r["judge"]] = r

    # Compute per-pair stats
    pair_stats = []
    for pid, judges in judgments_by_pair.items():
        if len(judges) < 2:
            continue
        comps = [composite(j["scores"]) for j in judges.values()]
        mean_comp = sum(comps) / len(comps)
        spread = max(comps) - min(comps)
        pair_stats.append({
            "pid": pid,
            "mean_composite": mean_comp,
            "spread": spread,
            "n_judges": len(judges),
            "judges": judges,
        })

    # Sort and pick top
    lowest = sorted(pair_stats, key=lambda x: x["mean_composite"])[: args.n_low]
    highest_disagreement = sorted(pair_stats, key=lambda x: -x["spread"])[: args.n_high_disagreement]

    # Write the audit report
    lines = [f"# Audit: {args.model} — low-scoring & high-disagreement cases\n"]
    lines.append(
        f"**What this is:** Spot-check of judges' fairness on the worst-scoring traces.\n"
        f"For each case, we show the actual trace, each judge's per-dimension scores,\n"
        f"their justifications, and their length-bias self-check.\n"
    )
    lines.append(f"\n**Total {args.model} traces audited:** {len(pair_stats)}\n")
    lines.append(f"**Composite range:** {min(p['mean_composite'] for p in pair_stats):.2f} → "
                 f"{max(p['mean_composite'] for p in pair_stats):.2f}\n")
    lines.append(f"**Median composite:** {statistics.median(p['mean_composite'] for p in pair_stats):.2f}\n")

    lines.append("\n---\n## SECTION A — LOWEST {} TRACES (by mean composite)\n".format(args.n_low))
    for rank, ps in enumerate(lowest, 1):
        pid = ps["pid"]
        rec = model_traces.get(pid, {})
        trace_obj = extract_trace_for_pair(rec)
        gold_fam = truth.get(pid, "UNKNOWN")
        predicted_fam = (trace_obj or {}).get("final_answer", {}).get("family") if trace_obj else None
        is_correct = (predicted_fam == gold_fam)

        lines.append(f"\n### A.{rank}  pair={pid}  mean composite={ps['mean_composite']:.2f}  "
                     f"(spread={ps['spread']:.2f})")
        lines.append(f"- **Gold family:** `{gold_fam}`  |  **Predicted family:** `{predicted_fam}`  |  "
                     f"**Family-prediction correct:** {'✅' if is_correct else '❌'}")
        lines.append(f"- **Judges:** {sorted(ps['judges'].keys())}")
        lines.append("")
        lines.append("#### Trace as the judges saw it")
        lines.append("```")
        lines.append(render_trace_for_judge(trace_obj))
        lines.append("```")
        lines.append("")
        lines.append("#### Per-judge scores")
        lines.append("| Judge | " + " | ".join(DIM_NAMES[d][:14] for d in DIM_IDS) + " | Composite |")
        lines.append("|" + "---|" * (len(DIM_IDS) + 2))
        for judge_name, j in sorted(ps["judges"].items()):
            sc = j["scores"]
            row = "| " + judge_name + " | " + " | ".join(str(sc.get(d, "-")) for d in DIM_IDS) + f" | {composite(sc):.2f} |"
            lines.append(row)
        lines.append("")
        lines.append("#### Justifications (why each judge scored each dim that way)")
        for judge_name, j in sorted(ps["judges"].items()):
            lines.append(f"**Judge: {judge_name}**")
            justs = j.get("justifications") or {}
            for d in DIM_IDS:
                lines.append(f"- *{DIM_NAMES[d]} ({j['scores'].get(d, '-')}/8)*: {justs.get(d, '(no justification)')}")
            self_check = j.get("length_bias_self_check") or "(no self-check)"
            lines.append(f"- *Length-bias self-check:* {self_check}")
            lines.append("")

    lines.append("\n---\n## SECTION B — HIGHEST DISAGREEMENT TRACES (judges disagree most)\n")
    for rank, ps in enumerate(highest_disagreement, 1):
        pid = ps["pid"]
        rec = model_traces.get(pid, {})
        trace_obj = extract_trace_for_pair(rec)
        gold_fam = truth.get(pid, "UNKNOWN")
        predicted_fam = (trace_obj or {}).get("final_answer", {}).get("family") if trace_obj else None

        lines.append(f"\n### B.{rank}  pair={pid}  mean composite={ps['mean_composite']:.2f}  "
                     f"(spread={ps['spread']:.2f})")
        lines.append(f"- **Gold family:** `{gold_fam}`  |  **Predicted family:** `{predicted_fam}`")
        lines.append("")
        lines.append("#### Trace as the judges saw it")
        lines.append("```")
        lines.append(render_trace_for_judge(trace_obj))
        lines.append("```")
        lines.append("")
        lines.append("#### Per-judge scores (note the spread)")
        lines.append("| Judge | " + " | ".join(DIM_NAMES[d][:14] for d in DIM_IDS) + " | Composite |")
        lines.append("|" + "---|" * (len(DIM_IDS) + 2))
        for judge_name, j in sorted(ps["judges"].items()):
            sc = j["scores"]
            row = "| " + judge_name + " | " + " | ".join(str(sc.get(d, "-")) for d in DIM_IDS) + f" | {composite(sc):.2f} |"
            lines.append(row)
        lines.append("")
        lines.append("#### Justifications")
        for judge_name, j in sorted(ps["judges"].items()):
            lines.append(f"**Judge: {judge_name}**")
            justs = j.get("justifications") or {}
            for d in DIM_IDS:
                lines.append(f"- *{DIM_NAMES[d]} ({j['scores'].get(d, '-')}/8)*: {justs.get(d, '(no justification)')}")
            self_check = j.get("length_bias_self_check") or "(no self-check)"
            lines.append(f"- *Length-bias self-check:* {self_check}")
            lines.append("")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[audit] wrote {out_path}")
    print(f"[audit] lowest mean composite: {lowest[0]['mean_composite']:.2f} (pair {lowest[0]['pid']})")
    print(f"[audit] highest disagreement spread: {highest_disagreement[0]['spread']:.2f} (pair {highest_disagreement[0]['pid']})")


if __name__ == "__main__":
    main()
