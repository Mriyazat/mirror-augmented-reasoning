"""Inventory student prediction files and estimate achievable ceilings.

This is a research diagnostic, not a deployable method. It computes:
  * individual macro-F1 for every candidate prediction file found;
  * oracle macro-F1 if we could pick the right family from any candidate;
  * simple majority vote macro-F1 across candidates;
  * best pair/triple oracle ceilings.

The goal is to answer: "Can the student reach ~0.60+ if we use a richer
candidate pool, or do we need new training?"
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/diag2/student_ceiling"
OUT.mkdir(parents=True, exist_ok=True)

FAMS = [
    "AdverseRisk", "Efficacy", "PD_Activity", "PK_Absorption",
    "PK_Distribution", "PK_Excretion", "PK_Metabolism",
]


def truth_for_split(split: str) -> dict[str, str]:
    path = ROOT / f"outputs/eval_prompts/{split}_test_5000_stratified.manifest.jsonl"
    out = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["pair_id"]] = r["family"]
    return out


def final_family(rec: dict) -> str | None:
    fp = rec.get("final_prediction") or {}
    fam = fp.get("family")
    return fam if fam in FAMS else None


def load_preds(path: Path, keep: set[str]) -> dict[str, str]:
    out = {}
    try:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = r.get("pair_id")
                if pid not in keep or pid in out:
                    continue
                if (r.get("input_order") or "ab") not in {"ab", "abba", "n/a", None}:
                    # BA-only files still use canonical pair_id; keep them too because
                    # the final family is order-invariant. Do not skip.
                    pass
                fam = final_family(r)
                out[pid] = fam or "PD_Activity"
    except Exception:
        return {}
    return out


def candidate_files(split: str) -> list[Path]:
    patterns = [
        f"outputs/eval_prompts/pred_phase4*{split}*.jsonl",
        f"outputs/eval_prompts/pre_sft_greedy_baselines/pred_phase4_{split}_greedy.jsonl",
        f"outputs/student/trace_align/eval_after/pred_traceAlign_{split}_greedy.jsonl",
        f"outputs/student/trace_align/eval_after_v2/pred_traceAlign_v2_{split}_greedy.jsonl",
        f"outputs/diag2/**/*{split}*.jsonl",
        f"outputs/eval_prompts/pred_cpu_stack_{split}.jsonl",
    ]
    files: set[Path] = set()
    for pat in patterns:
        files.update(ROOT.glob(pat))
    # Exclude non-student/frontier judge traces and incomplete/auxiliary files.
    bad_parts = [
        "xjudge", "trace_quality", "pred_claude", "pred_gpt4o", "pred_gemini",
        "frontier", "med42", "openbiollm", "biomistral",
    ]
    out = []
    for p in sorted(files):
        s = str(p).lower()
        if any(b in s for b in bad_parts):
            continue
        if p.exists() and p.stat().st_size > 0:
            out.append(p)
    return out


def macro(yt: list[str], yp: list[str]) -> float:
    return float(f1_score(yt, yp, labels=FAMS, average="macro", zero_division=0))


def oracle_pred(yt_by_pid: dict[str, str], pred_maps: list[dict[str, str]], pids: list[str]) -> list[str]:
    out = []
    for pid in pids:
        gold = yt_by_pid[pid]
        cands = [m[pid] for m in pred_maps]
        out.append(gold if gold in cands else cands[0])
    return out


def majority_pred(pred_maps: list[dict[str, str]], pids: list[str]) -> list[str]:
    out = []
    for pid in pids:
        votes = [m[pid] for m in pred_maps]
        out.append(Counter(votes).most_common(1)[0][0])
    return out


def main() -> None:
    report = ["# Student Candidate Ceiling Inventory\n"]
    summary = {}
    for split in ["random_full", "drug_cold", "pair_cold"]:
        truth = truth_for_split(split)
        keep = set(truth)
        entries = []
        for path in candidate_files(split):
            pred = load_preds(path, keep)
            if len(pred) < 4500:
                continue
            pids = sorted(set(pred) & keep)
            yt = [truth[p] for p in pids]
            yp = [pred[p] for p in pids]
            entries.append({
                "path": str(path.relative_to(ROOT)),
                "pred": pred,
                "n": len(pids),
                "macro_f1": macro(yt, yp),
                "acc": sum(a == b for a, b in zip(yt, yp)) / len(yt),
            })
        entries.sort(key=lambda x: x["macro_f1"], reverse=True)

        report.append(f"\n## {split}\n")
        report.append(f"Complete candidate files (n>=4500): **{len(entries)}**\n\n")
        report.append("| Rank | Macro-F1 | Acc | n | File |\n")
        report.append("|---:|---:|---:|---:|---|\n")
        for i, e in enumerate(entries[:30], 1):
            report.append(f"| {i} | {e['macro_f1']:.4f} | {e['acc']:.4f} | {e['n']} | `{e['path']}` |\n")

        if not entries:
            continue

        common = sorted(set.intersection(*[set(e["pred"]) for e in entries]) & keep)
        yt = [truth[p] for p in common]
        maps = [e["pred"] for e in entries]
        all_oracle = oracle_pred(truth, maps, common)
        all_majority = majority_pred(maps, common)
        report.append("\n### Ceiling over all complete candidates\n")
        report.append(f"- Common pairs: **{len(common)}**\n")
        report.append(f"- Best single file: **{entries[0]['macro_f1']:.4f}** (`{entries[0]['path']}`)\n")
        report.append(f"- Majority vote over all candidates: **{macro(yt, all_majority):.4f}**\n")
        report.append(f"- ORACLE over all candidates: **{macro(yt, all_oracle):.4f}**\n")

        # Best pair/triple oracle: tells us if 2-3 candidate modes suffice.
        best_pair = (0.0, None)
        best_triple = (0.0, None)
        top_for_combo = entries[:20]
        for combo in combinations(range(len(top_for_combo)), 2):
            combo_maps = [top_for_combo[i]["pred"] for i in combo]
            pids2 = sorted(set.intersection(*[set(m) for m in combo_maps]) & keep)
            val = macro([truth[p] for p in pids2], oracle_pred(truth, combo_maps, pids2))
            if val > best_pair[0]:
                best_pair = (val, combo)
        for combo in combinations(range(len(top_for_combo)), 3):
            combo_maps = [top_for_combo[i]["pred"] for i in combo]
            pids3 = sorted(set.intersection(*[set(m) for m in combo_maps]) & keep)
            val = macro([truth[p] for p in pids3], oracle_pred(truth, combo_maps, pids3))
            if val > best_triple[0]:
                best_triple = (val, combo)
        if best_pair[1] is not None:
            files = [top_for_combo[i]["path"] for i in best_pair[1]]
            report.append(f"- Best 2-candidate oracle: **{best_pair[0]:.4f}**\n")
            for f in files:
                report.append(f"  - `{f}`\n")
        if best_triple[1] is not None:
            files = [top_for_combo[i]["path"] for i in best_triple[1]]
            report.append(f"- Best 3-candidate oracle: **{best_triple[0]:.4f}**\n")
            for f in files:
                report.append(f"  - `{f}`\n")

        summary[split] = {
            "n_candidates": len(entries),
            "best_single": {"f1": entries[0]["macro_f1"], "path": entries[0]["path"]},
            "majority_all": macro(yt, all_majority),
            "oracle_all": macro(yt, all_oracle),
            "best_pair_oracle": best_pair[0],
            "best_triple_oracle": best_triple[0],
        }

    (OUT / "student_ceiling_inventory.md").write_text("".join(report))
    (OUT / "student_ceiling_inventory.json").write_text(json.dumps(summary, indent=2))
    print((OUT / "student_ceiling_inventory.md"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
